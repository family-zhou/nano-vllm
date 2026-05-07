import torch
from torch import nn


# ==============================================================================
# Gumbel-Max Trick 推导与实现
# ==============================================================================
# 目标：从分布 P 中采样，等价于计算 argmax(log(P) + G)，其中 G ~ Gumbel(0, 1)。
#
# 1. Gumbel 分布与指数分布的关系：
#    若 E ~ Exponential(1) (标准指数分布)，则 G = -log(E) 服从标准 Gumbel 分布。
#
# 2. 代入 Gumbel-Max 公式：
#    argmax_i (log(P_i) + G_i)
#  = argmax_i (log(P_i) - log(E_i))
#  = argmax_i (log(P_i / E_i))
#
# 3. 利用 log(x) 的单调递增性质：
#    求 log(X) 的最大值等价于求 X 的最大值。
#  => argmax_i (P_i / E_i)
#
# 4. 代码实现对应：
#    probs.div_(exponential_noise).argmax()
# ==============================================================================

class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperature > 1 会让logits的差异变小，后续经过softmax之后 概率也会更平滑
        # temperature < 1 会让logits的差异变大，后续经过softmax之后 概率也会更尖锐
        # input temperatures (B, ) logits (B, V)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) 
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens # （B，）
