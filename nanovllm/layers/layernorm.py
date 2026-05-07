import torch
from torch import nn

# 残差链接和归一化
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        # 计算 均方
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 逐元素乘法
        # Root Mean Square Normalization
        # RMS(x) = sqrt(1/n * sum(x^2) + eps)
        # y = x / sqrt(var + eps) * wight 
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype) # 将 残差转为原始精度并拷贝到 x 中
        var = x.pow(2).mean(dim=-1, keepdim=True) # 计算 均方
        # 逐元素乘法
        # Root Mean Square Normalization
        # RMS(x) = sqrt(1/n * sum(x^2) + eps)
        # y = x / sqrt(var + eps) * wight 
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
