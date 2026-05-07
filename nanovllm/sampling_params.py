from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # 采样参数
    temperature: float = 1.0
    max_tokens: int = 64 # 最大生成的 token 数
    ignore_eos: bool = False # 是否忽略 EOS 

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
