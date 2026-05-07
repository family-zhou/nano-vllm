from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    大语言模型（LLM）推理引擎调度器。
    负责在每次模型前向传播前，挑选出合适的请求（Sequence）放入 GPU 计算，
    并管理其 KV Cache 物理块的分配与生命周期。
    """

    def __init__(self, config: Config):
        # --- 引擎基础配置 ---
        # 单次 batch 前向传播最多处理的序列（Sequence）数量上限
        self.max_num_seqs = config.max_num_seqs
        # 单次 batch 前向传播最多处理的 Token 总数上限（防止 OOM）
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 模型的 EOS (End of Sentence) token ID，用于判断生成是否结束
        self.eos = config.eos
        # 每个 KV Cache 物理块能容纳的 Token 数量（如 16 或 32）
        self.block_size = config.kvcache_block_size # 256
        
        # --- 显存/物理块管理器 ---
        # 核心组件：用于实现 PagedAttention，负责全局 KV Cache 块的分配与回收
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # --- 调度队列 ---
        # 使用双端队列(deque)，便于在两端进行高效的 O(1) 插入和删除
        # 等待队列：存放刚接收到的新请求，或者因显存不足被抢占退回的请求
        self.waiting: deque[Sequence] = deque()
        # 运行队列：存放正在进行 Decode (逐字生成) 的请求
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """判断调度器是否空闲（所有请求均处理完毕）"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """接收新请求，默认加入等待队列的最末尾"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度逻辑。
        每次前向传播前调用，返回本次需要送入模型的序列列表，以及当前是否为 Prefill 阶段。
        """
        scheduled_seqs = []       # 存放本次被选中的序列
        num_batched_tokens = 0    # 记录本次 batch 已经累计分配的 Token 数量

        # ==========================================
        # 阶段一：Prefill (预填充) 阶段调度
        # ==========================================
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 窥探等待队列的第一个序列
            seq = self.waiting[0]
            # 计算当前 batch 剩余可用的 token 额度
            remaining = self.max_num_batched_tokens - num_batched_tokens
            
            if remaining == 0:
                break  # Token 额度耗尽，停止调度新序列

            # 1. 计算该序列本次需要处理的 token 数量
            if not seq.block_table:
                # 纯新请求：询问 BlockManager 该序列的 Prompt 缓存命中了多少块物理块
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # -1 表示当前全局空闲物理块不足，无法为其分配，放弃本次调度
                # 需要处理的 token 数 = Prompt总长度 - 已经被前缀缓存(Prefix Cache)命中的长度
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已部分处理过的请求（之前被 Chunked 截断过）：计算还剩多少 token 没处理
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            
            # 2. Chunked Prefill (分块预填充) 拦截逻辑
            # 如果剩余的 token 额度不足以处理完这个序列，说明需要将该序列截断处理。
            # 为了保证 GPU 计算效率，只允许当前 batch 的*第一个*序列被截断（即 scheduled_seqs 为空时）
            if remaining < num_tokens and scheduled_seqs:
                break

            # 3. 物理块分配与额度扣除
            if not seq.block_table:
                # 为新序列正式分配 KV Cache 物理块
                self.block_manager.allocate(seq, num_cached_blocks)
            
            # 确定该序列本次实际调度的 token 数量（取需求量与剩余额度的最小值）
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            
            # 4. 状态流转判断
            # 如果该序列的所有 Prompt Token 都已经（或将在本次）处理完毕
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING  # 状态切换为运行中
                self.waiting.popleft()               # 正式从等待队列移除
                self.running.append(seq)             # 移入运行队列，准备后续的 Decode
            
            scheduled_seqs.append(seq)

        # 只要预填充阶段选中了序列，就直接返回并执行前向传播
        # 返回 True 代表当前是 Prefill 阶段。通常 Prefill 和 Decode 不混在一个 batch 中计算
        if scheduled_seqs:
            return scheduled_seqs, True

        # ==========================================
        # 阶段二：Decode (解码/逐字生成) 阶段调度
        # (只有当等待队列无可用任务，或者 Prefill 资源受限时才会进入)
        # ==========================================
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查为了生成下一个 Token，当前的 KV Cache 物理块容量是否足够
            while not self.block_manager.can_append(seq):
                # 显存不足，触发抢占 (Preemption) 机制！
                if self.running:
                    # 牺牲策略：从 running 队列的最尾端踢出一个序列（释放其显存给前面的序列用）
                    self.preempt(self.running.pop())
                else:
                    # 如果只剩当前序列自己，依然放不下，只能抢占自己并中止本序列的调度
                    self.preempt(seq)
                    break
            else:
                # while...else 语法：如果没有被 break（说明显存足够），则执行以下调度逻辑
                seq.num_scheduled_tokens = 1  # Decode 阶段每次前向传播只生成 1 个 Token
                seq.is_prefill = False        # 明确标记为非预填充阶段
                self.block_manager.may_append(seq) # 向 BlockManager 申请追加记录
                scheduled_seqs.append(seq)

        # 断言：如果是 Decode 阶段，必须保证至少有一个序列被调度，否则系统陷入死锁
        assert scheduled_seqs
        
        # 保持 FIFO 优先级：将本次调度的序列按原顺序重新放回 running 队列的最左侧
        self.running.extendleft(reversed(scheduled_seqs))
        # 返回 False 代表当前是 Decode 阶段
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占机制。当 Decode 阶段显存耗尽时调用。
        被抢占的序列会被剥夺显存，并退回等待队列，等待后续重新进行计算（Recomputation）。
        """
        seq.status = SequenceStatus.WAITING # 状态回退为等待中
        seq.is_prefill = True               # 标记为需要重新走 Prefill 阶段
        self.block_manager.deallocate(seq)  # 核心：彻底释放该序列当前占用的所有物理块
        self.waiting.appendleft(seq)        # 插队到等待队列的最前面，确保下次有资源时优先恢复它

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        后处理逻辑。在模型完成一次前向传播（Forward）后调用。
        负责更新缓存状态、保存新生成的 Token，并处理结束逻辑。
        """
        for seq, token_id in zip(seqs, token_ids):
            # 1. 前缀缓存处理：对新写入的 KV Cache 块计算 Hash，以便未来其他请求复用
            self.block_manager.hash_blocks(seq) 
            
            # 2. 更新 token 计数器
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0  # 重置单次调度计数，等待下一次 schedule
            
            # 3. 如果当前是 Prefill 阶段，且 Prompt 尚未处理完毕（Chunked 截断情况）
            # 则跳过 Token 追加逻辑，因为它还没开始真正生成新内容
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            
            # 4. 追加模型刚预测出的新 Token
            seq.append_token(token_id)
            
            # 5. 终止条件检查
            # 如果生成了结束符（且未设置忽略 EOS），或者生成长度达到了预设的最大 token 数
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED # 标记任务彻底完成
                self.block_manager.deallocate(seq)   # 回收其占用的所有 KV Cache 显存
                self.running.remove(seq)             # 将其从运行队列中剔除