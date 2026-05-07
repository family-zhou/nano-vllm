import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    # SwiGLU 激活函数
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        # 将 x 分成两部分，一部分是 gate，一部分是 up
        # F.silu(x) = x * sigmoid(x)
        return F.silu(x) * y # 将 gate 和 up 相乘
