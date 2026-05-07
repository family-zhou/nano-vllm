import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size # 将此表分开
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank # 当前进程的开始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition # 当前进程的结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0) # self.num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size # 当前进程的开始索引
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # 当前进程的权重
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight) # 直接使用 embedding 算子
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 将 mask 扩展到 1 维度， 然后与 y 相乘
            # 为什么可以用 sum 因为其他的查询都是0向量
            dist.all_reduce(y) # op默认为 ReduceOp.SUM
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight) # x @ W^T
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0) # 使用gather 将数据都放到 rank0 然后进入后续采样阶段。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
