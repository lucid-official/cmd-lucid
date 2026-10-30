"""Cosmos-Reason1-7B's language model, as CMD uses it to embed a prompt.

The text-only path of ``Qwen2_5_VLForConditionalGeneration``: with no image
or video the three M-RoPE sections share one position, which is ordinary
1-D RoPE over positions ``0..511``. The reference passes no attention mask,
so the padding tokens are part of the causal sequence and of the output.

``CosmosTextEncoder.forward`` keeps every layer's output — Hugging Face's
``hidden_states[1:]``: layers 1..27 as they leave the residual stream and
layer 28 after the final RMSNorm — normalises each per token
(``(x - mean) / (std + 1e-8)``, unbiased std) and concatenates them into
28 x 3584 = 100,352 channels.

Parameter names match the released shards (``model.layers.N...``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

Tensor = lucid.Tensor


@dataclass(frozen=True)
class QwenConfig:
    hidden_size: int = 3584
    num_layers: int = 28
    num_heads: int = 28
    num_kv_heads: int = 4
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 152064

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class Attention(nn.Module):
    def __init__(self, cfg: QwenConfig) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = (
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
        )
        self.q_proj = nn.Linear(
            cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, n, _ = x.shape
        q = self.q_proj(x).reshape(b, n, self.heads, self.head_dim).swapaxes(1, 2)
        k = self.k_proj(x).reshape(b, n, self.kv_heads, self.head_dim).swapaxes(1, 2)
        v = self.v_proj(x).reshape(b, n, self.kv_heads, self.head_dim).swapaxes(1, 2)
        q, k = F.apply_rotary_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(out.swapaxes(1, 2).reshape(b, n, self.heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, cfg: QwenConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: QwenConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps
        )
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenTextModel(nn.Module):
    def __init__(self, cfg: QwenConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


def _mean_normalize(x: Tensor) -> Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)


class CosmosReasonEncoder(nn.Module):
    """``model.*`` of Cosmos-Reason1-7B and the Predict2.5 feature recipe."""

    def __init__(self, cfg: QwenConfig = QwenConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = QwenTextModel(cfg)

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def _rope(self, length: int, device: str) -> tuple[Tensor, Tensor]:
        """cos/sin computed in fp32 and cast to the model dtype, as HF does."""
        half = (
            lucid.arange(0, self.cfg.head_dim, 2, dtype=lucid.float32, device=device)
            / self.cfg.head_dim
        )
        base = lucid.full(
            tuple(half.shape), self.cfg.rope_theta, dtype=lucid.float32, device=device
        )
        inv_freq = 1.0 / lucid.pow(base, half)
        pos = lucid.arange(length, dtype=lucid.float32, device=device)
        freqs = pos.reshape(-1, 1) * inv_freq.reshape(1, -1)
        emb = lucid.cat([freqs, freqs], dim=-1)
        return lucid.cos(emb).to(self.dtype), lucid.sin(emb).to(self.dtype)

    def hidden_states(self, ids: Tensor) -> list[Tensor]:
        """Hugging Face's ``hidden_states[1:]`` for token ids ``(B, L)``."""
        x = self.model.embed_tokens(ids)
        cos, sin = self._rope(ids.shape[1], ids.device.type)
        states = []
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin)
            states.append(self.model.norm(x) if i == len(self.model.layers) - 1 else x)
        return states

    def forward(self, ids: Tensor) -> Tensor:
        """Prompt features ``(B, L, num_layers * hidden)`` in the model dtype."""
        return lucid.cat([_mean_normalize(s) for s in self.hidden_states(ids)], dim=-1)

    @classmethod
    def load(
        cls, directory: str | Path, device: str = "metal", dtype=lucid.bfloat16
    ) -> "CosmosReasonEncoder":
        """Build on ``device`` and fill it shard by shard (no random init is evaluated)."""
        directory = Path(directory)
        prev_device, prev_dtype = lucid.get_default_device(), lucid.get_default_dtype()
        lucid.set_default_device(device)
        lucid.set_default_dtype(dtype)
        try:
            model = cls()
        finally:
            lucid.set_default_device(prev_device)
            lucid.set_default_dtype(prev_dtype)
        expected = set(model.state_dict())
        seen: set[str] = set()
        for shard in sorted(directory.glob("model-*.safetensors")):
            weights = lucid.load_safetensors(str(shard), device=device)
            weights = {
                k: (v if v.dtype == dtype else v.to(dtype))
                for k, v in weights.items()
                if k in expected
            }
            model.load_state_dict(weights, strict=False, assign=True)
            seen |= set(weights)
            del weights
        missing = expected - seen
        if missing:
            raise RuntimeError(f"text encoder weights missing: {sorted(missing)[:5]}")
        return model.eval()
