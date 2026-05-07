import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size() # 总进程数
        self.total_num_heads = num_heads # 总Q头数
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size # 每个rank需要处理的Q头数
        self.total_num_kv_heads = num_kv_heads # 总kv头数
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size # 每个rank需要处理的KV头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads # 每个头的大小
        self.q_size = self.num_heads * self.head_dim # Q头数 * 每个头的大小
        self.kv_size = self.num_kv_heads * self.head_dim # KV头数 * 每个头的大小
        self.scaling = self.head_dim ** -0.5 # 缩放因子 sqrt(1/d)
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )# 按照 Q, K, V的顺序分别进行投影，并将输出拼接在一起
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )# 将计算出的 attention out 投影回 hidden_size
        # RoPE 旋转位置编码
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # 获取 RoPE embedding 对象
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 注意力机制
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # norm
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 将 hidden_states 投影为 Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        # 将 Q, K, V 重新排列为 [batch_size, num_heads, head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 将 Q, K 应用 RoPE 旋转位置编码
        # NOTE 位置编码在这儿， 再decoder layer内部
        q, k = self.rotary_emb(positions, q, k)
        # 注意力机制
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))# (bs, num_heads * head_dim) -> (bs, hidden_size)
        return output


class Qwen3MLP(nn.Module):
    '''
    这是SwiGLU FNN的实现
    '''
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # 这个类将 hidden_size 投影为 intermediate_size * 2
        # 一部分用于 gate，一部分用于 up
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 将 intermediate_size 投影回 hidden_size
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


"""
=========================================================================================
                            Qwen3DecoderLayer Data Flow
=========================================================================================

[Input]  hidden_states (N, d_model)
            residual      (N, d_model)  <-- initially None at Layer 0
                |             |
                v             v
+---------------------------------------------------------------------------------------+
|                      self.input_layernorm (RMSNorm)                                   |
|                                                                                       |
|  1. Fused Add : x = hidden_states + residual                                          |
|  2. Save Res  : residual = x                         ------------------------+        |
|  3. RMSNorm   : hidden_states = x / RMS(x) * weight                          |        |
+---------------------------------------------------------------------------------------+
                |                                                               |
                | (Normalized)                                                  |
                v                                                               |
+-------------------------------------------------------------+                |
|                      self.self_attn (Qwen3Attention)        |                |
|                                                             |                |
|  1. QKV Proj : hidden_states -> Q, K, V (qkv_proj)          |                | (Saved
|  2. RoPE     : Apply Rotary Positional Embeddings to Q, K   |                | Residual)
|  3. Attention: Softmax(Q*K^T / sqrt(d)) * V                 |                |
|  4. Out Proj : Output projection                            |                |
+-------------------------------------------------------------+                |
                |                                                               |
                | (Attention Output)                                            |
                v                                                               v
+---------------------------------------------------------------------------------------+
|                  self.post_attention_layernorm (RMSNorm)                              |
|                                                                                       |
|  1. Fused Add : x = hidden_states + residual                                          |
|  2. Save Res  : residual = x                         ------------------------+        |
|  3. RMSNorm   : hidden_states = x / RMS(x) * weight                          |        |
+---------------------------------------------------------------------------------------+
                |                                                               |
                | (Normalized)                                                  |
                v                                                               |
+-------------------------------------------------------------+                |
|                           self.mlp (Qwen3MLP)               |                | (Saved
|                                                             |                | Residual)
|  1. Gate/Up  : hidden_states -> W_gate, W_up (gate_up_proj) |                |
|  2. Act & Mul: hidden_act(W_gate) * W_up (SwiGLU logic)     |                |
|  3. Down Proj: Output -> W_down                             |                |
+-------------------------------------------------------------+                |
                |                                                               |
                | (MLP Output)                                                  |
                v                                                               v
[Output] hidden_states (N, d_model) ------------------------> Passes to next layer
            residual      (N, d_model) ------------------------> Passes to next layer
            
=========================================================================================
"""
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # attention
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # FFN
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # norm & residual
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input norm & residual
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # attention
        hidden_states = self.self_attn(positions, hidden_states)
        # post attention norm & residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # FFN
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # token id -> embedding
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # N 个 decoder layers
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终的 norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # embedding 是一个 (vocab_size, hidden_size) 的矩阵， 负责将 token id 映射为 embedding
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size) # 这个网络负责将 hidden_states 投影为 logits
        # 如果需要 tie word embeddings， 则将 lm_head 的权重与 embed_tokens 的权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # return hidden_states
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # return logits
        return self.lm_head(hidden_states)
