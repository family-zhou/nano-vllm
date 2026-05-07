from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id      # 物理块的唯一标识符（类似内存条上的物理页号）
        self.ref_count = 0            # 引用计数：记录当前有多少个请求(Sequence)在共用这个块
        self.hash = -1                # 这个块所包含的 Token 序列的唯一特征码（用于前缀缓存）
        self.token_ids = []           # 实际存放在这个块里的 Token ID 列表

    def update(self, hash: int, token_ids: list[int]):
        # 当一个物理块被填满数据后，更新它的哈希值和包含的 tokens
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 当物理块被真正分配给一个新任务时，清空历史包袱
        self.ref_count = 1            # 刚分配时，引用计数初始化为 1
        self.hash = -1                # 清除历史哈希
        self.token_ids = []           # 清除历史数据


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size  # 每个物理块能容纳的 Token 数量（如 16）
        # 初始化所有的物理块对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        
        # 全局哈希表：建立 "Token序列哈希值 -> 物理块ID" 的映射。这是实现前缀共享的核心字典。
        self.hash_to_block_id: dict[int, int] = dict()
        
        # 空闲队列：存放当前没人用的物理块 ID (双端队列，方便做 LRU 淘汰)
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # 已用集合：存放正在被使用的物理块 ID
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # 链式哈希，将前一个blk的hash作为当前blk的hash的一部分
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # 更新当前blk的hash
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        # 选择一个最左侧（最老的）空闲块进行分配
        block_id = self.free_block_ids.popleft()
        # 获取Block对象
        block = self.blocks[block_id]
        assert block.ref_count == 0 # 确保这个块是空闲的
        # 重点：如果这个块身上还残留着历史的哈希值，且全局哈希表还指向它
        # 说明它之前是作为“备用缓存”存在的，现在要被别人覆盖了，必须把它从全局哈希表中除名 (Eviction)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset() # 重置这个块的引用计数、哈希值和token_ids
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # 确保这个块没有被使用
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        # 释放这个块，把它移到空闲队列的末尾
        # 这里并没有清空 block的hash和token_ids, 全局哈希表中还保留着它的信息
        # 这个块相当于变成了 备用缓存 ，只要显存没有耗尽，它就一直在队列里面
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0 # 记录当前已经缓存了多少个blk
        num_new_blocks = seq.num_blocks # 记录这个seq需要多少个blk
        # 遍历前 N-1 个完整的块，看能命中多少个历史缓存
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) # 计算当前blk的hash
            block_id = self.hash_to_block_id.get(h, -1) # 检查这个hash是否已经缓存过
            # 如果这个hash没有缓存过，或者这个blk的token_ids与缓存的不一致（这种情况是为了应对hash冲突）
            # 说明这个blk不能被复用，需要分配新的物理块
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1 # 命中一个历史缓存
            # 当我们确实命中了一个 used的 blk
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
            # 命中了一个blk，但是它实际上不是 used 但是其中存储的 kv tokens还没有被擦除
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 确保这个seq没有被分配过物理块
        assert not seq.block_table # 如果这个seq已经分配过物理块，则报错
        h = -1
        # 先复用命中块，注意 cached blks一定是连续的
        for i in range(num_cached_blocks):
            # 获取当前blk的token_ids
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) # 计算当前blk的hash
            block_id = self.hash_to_block_id[h] # 从hash表中查询 blk id
            block = self.blocks[block_id] # 获取blk对象
            if block_id in self.used_block_ids: # 如果这个blk正在被使用
                block.ref_count += 1 # 引用计数加1
            else: # 如果这个blk是空闲的
                block.ref_count = 1 # 引用计数初始化为1
                self.free_block_ids.remove(block_id) # 从空闲队列中移除
                self.used_block_ids.add(block_id) # 加入已用集合
            seq.block_table.append(block_id) # 将blk id添加到seq的block_table中
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block()) # 分配新的物理块
        seq.num_cached_tokens = num_cached_blocks * self.block_size # 更新seq的缓存token数量

    def deallocate(self, seq: Sequence):
        # 从后往前回收 blk , 最近使用的 blk 会被插到前面
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1 # 引用计数减1
            if block.ref_count == 0:
                self._deallocate_block(block_id) # 如果引用计数为0，则释放这个blk
        seq.num_cached_tokens = 0 # 清空seq的缓存token数量
        seq.block_table.clear() # 清空seq的block_table

    def can_append(self, seq: Sequence) -> bool:
        # Sequence对象的长度是token数量，如果token数量不能被block_size整除，则需要分配一个额外的blk
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 真的需要新的块的时候，申请一个新的块挂在页表最后
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # 计算缓存的 blk的hash值，并更新blk对象的hash值和token_ids
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
