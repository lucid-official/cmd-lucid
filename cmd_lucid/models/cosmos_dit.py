"""Cosmos-Predict2.5 2B DiT, causal, as CMD runs it.

A port of ``cosmos/minimal_v4_dit.py`` + ``cosmos/causal_model.py`` of the
reference. Module and parameter names match the released checkpoints, so a
``nvidia/cmd`` file loads with ``strict=True``.

Precision follows the reference run (bf16 weights, CUDA):

* the network runs in ``dtype`` (bf16 by default);
* every adaLN modulation, the final layer, and RoPE run in fp32 — the
  reference's ``use_wan_fp32_strategy`` (``autocast(dtype=float32)`` regions and
  q/k upcast before RoPE);
* in the streaming path the timestep MLP runs in ``dtype``; in the
  full-sequence path it sits inside the fp32 region.

Self-attention is handed an *attention strategy* by the caller, which is how
one set of blocks serves all three ways CMD uses the network:

* :class:`StreamingAttention` — one causal chunk at a time against a
  :class:`FrameKVCache` (inference, and the student rollout in training);
* :class:`MaskedAttention` — a whole sequence under a block-causal mask
  (teacher pre-training with Diffusion Forcing, packed CMD scoring),
  evaluated per query block so no token-level mask is ever built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

Tensor = lucid.Tensor


@dataclass(frozen=True)
class DiTConfig:
    """``CosmosDiffusionWrapper._model_kwargs`` of the reference."""

    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    mlp_ratio: float = 4.0
    adaln_lora_dim: int = 256
    crossattn_proj_in_channels: int = 100352
    crossattn_emb_channels: int = 1024
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    rope_h_extrapolation_ratio: float = 3.0
    rope_w_extrapolation_ratio: float = 3.0
    rope_t_extrapolation_ratio: float = 1.0
    timestep_scale: float = 0.001
    camera_dim: Optional[int] = None  # 6 * 16**2 for the camera models

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads

    @property
    def patch_in(self) -> int:
        # latent channels + condition mask + padding mask
        return (self.in_channels + 2) * self.patch_spatial**2 * self.patch_temporal


# --------------------------------------------------------------------------- #
# attention strategies
# --------------------------------------------------------------------------- #


class FrameKVCache:
    """Per-block K/V of committed latent frames, keyed by absolute frame index.

    Mirrors ``CausalCosmosAttention``'s cache: frames are written by a store
    pass, and a later query of ``n`` frames at ``current`` reads the
    ``local_attn - n`` most recent frames before it (plus ``sink`` leading
    frames). Frames no query can reach again are dropped, so memory stays at
    one attention window however long the rollout.
    """

    def __init__(self, num_blocks: int, local_attn: int = -1, sink: int = 0) -> None:
        if local_attn == 0 or local_attn < -1:
            raise ValueError("local_attn must be -1 or a positive frame count")
        self.local_attn = local_attn
        self.sink = sink
        self.frames: list[dict[int, tuple[Tensor, Tensor]]] = [
            {} for _ in range(num_blocks)
        ]

    def reset(self) -> None:
        for layer in self.frames:
            layer.clear()

    def history_indices(self, current: int, count: int) -> list[int]:
        """``CausalCosmosAttention._history_indices``."""
        if self.local_attn == -1:
            recent_start = self.sink
        else:
            recent_history = max(max(self.local_attn - self.sink, 0) - count, 0)
            recent_start = max(self.sink, current - recent_history)
        sink = range(min(self.sink, current, recent_start))
        return list(sink) + list(range(recent_start, current))

    def store(self, layer: int, frame: int, k: Tensor, v: Tensor) -> None:
        cache = self.frames[layer]
        # Contiguous copies: keeping the projections' transposed views alive
        # retains an extra (L, D) buffer per layer and frame (~180 MB/frame).
        cache[frame] = (k.contiguous(), v.contiguous())
        if self.local_attn > 0:
            # A query starting at frame f >= `frame` reads frames >= f - local_attn + 1.
            oldest = frame - self.local_attn + 1
            for old in [f for f in cache if self.sink <= f < oldest]:
                del cache[old]

    def read(self, layer: int, indices: list[int]) -> tuple[Tensor, Tensor] | None:
        if not indices:
            return None
        cache = self.frames[layer]
        missing = [i for i in indices if i not in cache]
        if missing:
            raise RuntimeError(f"KV cache is missing frames {missing[:4]}")
        ks = [cache[i][0] for i in indices]
        vs = [cache[i][1] for i in indices]
        if len(ks) == 1:
            return ks[0], vs[0]
        return lucid.cat(ks, dim=2), lucid.cat(vs, dim=2)

    @property
    def num_frames(self) -> int:
        return len(self.frames[0])


def block_causal_ranges(
    num_frames: int,
    frames_per_block: int,
    independent_first_frame: bool,
    local_attn: int = -1,
    sink: int = 0,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Query blocks and the key frames each may see, in frame units.

    The semantics of ``CausalCosmosAttention._block_causal_mask``'s
    ``mask_mod``: a query sees earlier blocks and its own, and with a local
    window only the ``local_attn - sink`` frames ending at its block's
    nominal end, plus ``sink`` leading frames. Every frame of a block shares
    the same keys, so the mask is a list of ``(q_start, q_end, key_ranges)``.
    """
    prefix = 1 if independent_first_frame else 0
    starts = ([0] if prefix else []) + list(range(prefix, num_frames, frames_per_block))
    out = []
    for i, q_start in enumerate(starts):
        nominal_end = q_start + (1 if (prefix and i == 0) else frames_per_block)
        q_end = min(nominal_end, num_frames)
        if local_attn > 0:
            window = max(local_attn - sink, 0)
            recent_start = max(nominal_end - window, 0)
            sink_end = min(q_start, sink, recent_start)
            recent_start = max(sink_end, recent_start)
            ranges = [(0, sink_end), (recent_start, q_end)]
        else:
            ranges = [(0, q_end)]
        merged: list[tuple[int, int]] = []
        for a, b in ranges:
            if a >= b:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        out.append((q_start, q_end, merged))
    return out


class MaskedAttention:
    """Whole-sequence block-causal attention, evaluated one query block at a time.

    Exact with respect to the reference mask, without materialising a
    token-level mask: each query block attends to the concatenation of its
    allowed key frames. ``detach_frames`` stops gradients through the K/V of
    the leading frames, as the reference's packed teacher-forcing pass does
    for its clean history.
    """

    def __init__(
        self,
        frames_per_block: int = 1,
        independent_first_frame: bool = False,
        local_attn: int = -1,
        sink: int = 0,
        detach_frames: int = 0,
    ) -> None:
        self.frames_per_block = frames_per_block
        self.independent_first_frame = independent_first_frame
        self.local_attn = local_attn
        self.sink = sink
        self.detach_frames = detach_frames

    def __call__(
        self, layer: int, q: Tensor, k: Tensor, v: Tensor, frames: int
    ) -> Tensor:
        tokens = q.shape[2] // frames
        if self.detach_frames:
            n = self.detach_frames * tokens
            k = lucid.cat([k[:, :, :n].detach(), k[:, :, n:]], dim=2)
            v = lucid.cat([v[:, :, :n].detach(), v[:, :, n:]], dim=2)
        plan = block_causal_ranges(
            frames,
            self.frames_per_block,
            self.independent_first_frame,
            self.local_attn,
            self.sink,
        )
        if len(plan) == 1 and plan[0][2] == [(0, frames)]:
            return F.scaled_dot_product_attention(q, k, v)
        outs = []
        for q_start, q_end, ranges in plan:
            qs = q[:, :, q_start * tokens : q_end * tokens]
            ks = [k[:, :, a * tokens : b * tokens] for a, b in ranges]
            vs = [v[:, :, a * tokens : b * tokens] for a, b in ranges]
            kk = ks[0] if len(ks) == 1 else lucid.cat(ks, dim=2)
            vv = vs[0] if len(vs) == 1 else lucid.cat(vs, dim=2)
            outs.append(F.scaled_dot_product_attention(qs, kk, vv))
        return outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)


class StreamingAttention:
    """One causal chunk against the frame cache (``forward_seq`` with KV)."""

    def __init__(
        self,
        cache: FrameKVCache,
        start: int,
        store: bool,
        frames_per_block: int = 1,
        independent_first_frame: bool = False,
    ) -> None:
        self.cache = cache
        self.start = start
        self.store = store
        self.frames_per_block = frames_per_block
        self.independent_first_frame = independent_first_frame

    def __call__(
        self, layer: int, q: Tensor, k: Tensor, v: Tensor, frames: int
    ) -> Tensor:
        tokens = k.shape[2] // frames
        if self.store and frames > 1:
            # Reference quirk kept on purpose: a multi-frame store pass writes
            # each frame, then attends within the chunk only (no history).
            for f in range(frames):
                sl = slice(f * tokens, (f + 1) * tokens)
                self.cache.store(layer, self.start + f, k[:, :, sl], v[:, :, sl])
            within = MaskedAttention(
                self.frames_per_block,
                self.independent_first_frame,
                self.cache.local_attn,
                self.cache.sink,
            )
            return within(layer, q, k, v, frames)
        if self.store:
            self.cache.store(layer, self.start, k, v)
        history = self.cache.read(layer, self.cache.history_indices(self.start, frames))
        if history is not None:
            hk, hv = history
            if hk.shape[0] != k.shape[0]:
                # A CFG batch [cond | uncond] reads the cache the conditional pass wrote.
                reps = k.shape[0] // hk.shape[0]
                hk = hk.repeat(reps, 1, 1, 1)
                hv = hv.repeat(reps, 1, 1, 1)
            k = lucid.cat([hk, k], dim=2)
            v = lucid.cat([hv, v], dim=2)
        return F.scaled_dot_product_attention(q, k, v)


# --------------------------------------------------------------------------- #
# modules (names follow the checkpoint)
# --------------------------------------------------------------------------- #


def _fp32_linear(x: Tensor, layer: nn.Linear) -> Tensor:
    """A linear layer inside the reference's ``autocast(float32)`` region."""
    bias = None if layer.bias is None else layer.bias.float()
    return F.linear(x.float(), layer.weight.float(), bias)


class Timesteps(nn.Module):
    """Sinusoidal timestep features, ``cat([cos, sin])``, in fp32."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: Tensor) -> Tensor:
        b, t = timesteps.shape
        half = self.num_channels // 2
        exponent = -math.log(10000) * lucid.arange(
            half, dtype=lucid.float32, device=timesteps.device.type
        )
        freqs = lucid.exp(exponent / half)
        emb = timesteps.reshape(-1, 1).float() * freqs.reshape(1, -1)
        emb = lucid.cat([lucid.cos(emb), lucid.sin(emb)], dim=-1)
        return emb.reshape(b, t, self.num_channels)


class TimestepEmbedding(nn.Module):
    """``linear_1 -> SiLU -> linear_2`` producing the adaLN-LoRA terms."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, sample: Tensor, fp32: bool = False) -> tuple[Tensor, Tensor]:
        sample = sample.to(self.linear_1.weight.dtype)
        if fp32:
            h = F.silu(_fp32_linear(sample, self.linear_1))
            lora = _fp32_linear(h, self.linear_2)
        else:
            lora = self.linear_2(self.activation(self.linear_1(sample)))
        return sample, lora


class PatchEmbed(nn.Module):
    """``b c (t r) (h m) (w n) -> b t h w (c r m n)`` then a linear map.

    Slot 0 of ``proj`` is the reference's einops ``Rearrange``; it holds no
    weights and is applied here by reshape/permute.
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.p = cfg.patch_spatial
        self.pt = cfg.patch_temporal
        self.proj = nn.Sequential(
            nn.Identity(), nn.Linear(cfg.patch_in, cfg.model_channels, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        p, pt = self.p, self.pt
        x = x.reshape(b, c, t // pt, pt, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
            b, t // pt, h // p, w // p, c * pt * p * p
        )
        return self.proj[1](x)


class Rope3D(nn.Module):
    """``VideoRopePosition3DEmb`` with explicit token positions.

    The buffers are loaded from the checkpoint, in its dtype, because the
    reference derives its frequencies from them after casting the model to
    bf16.
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        dim = cfg.head_dim
        self.dim_h = dim // 6 * 2
        self.dim_t = dim - 2 * self.dim_h
        length = max(
            cfg.max_img_h // cfg.patch_spatial,
            cfg.max_img_w // cfg.patch_spatial,
            cfg.max_frames,
        )
        self.register_buffer("seq", lucid.arange(length, dtype=lucid.float32))
        self.register_buffer(
            "dim_spatial_range",
            lucid.arange(0, self.dim_h, 2, dtype=lucid.float32) / self.dim_h,
        )
        self.register_buffer(
            "dim_temporal_range",
            lucid.arange(0, self.dim_t, 2, dtype=lucid.float32) / self.dim_t,
        )
        self.h_ntk = cfg.rope_h_extrapolation_ratio ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk = cfg.rope_w_extrapolation_ratio ** (self.dim_h / (self.dim_h - 2))
        self.t_ntk = cfg.rope_t_extrapolation_ratio ** (self.dim_t / (self.dim_t - 2))
        self._tables: dict[tuple, tuple[Tensor, Tensor]] = {}

    def _freqs(self, theta: float, dim_range: Tensor) -> Tensor:
        base = lucid.full(
            tuple(dim_range.shape),
            theta,
            dtype=lucid.float32,
            device=dim_range.device.type,
        )
        return 1.0 / lucid.pow(base, dim_range.float())

    def tables(
        self, t_start: int, t_count: int, h: int, w: int
    ) -> tuple[Tensor, Tensor]:
        """cos/sin tables ``(t_count*h*w, head_dim)`` for frames ``t_start..``."""
        key = (t_start, t_count, h, w, str(self.dim_spatial_range.device))
        hit = self._tables.get(key)
        if hit is not None:
            return hit
        dev = self.dim_spatial_range.device.type
        fh = self._freqs(10000.0 * self.h_ntk, self.dim_spatial_range)
        fw = self._freqs(10000.0 * self.w_ntk, self.dim_spatial_range)
        ft = self._freqs(10000.0 * self.t_ntk, self.dim_temporal_range)
        pt = lucid.arange(t_start, t_start + t_count, dtype=lucid.float32, device=dev)
        ph = lucid.arange(h, dtype=lucid.float32, device=dev)
        pw = lucid.arange(w, dtype=lucid.float32, device=dev)
        et = (
            (pt.reshape(-1, 1) * ft.reshape(1, -1))
            .reshape(t_count, 1, 1, -1)
            .expand(t_count, h, w, ft.shape[0])
        )
        eh = (
            (ph.reshape(-1, 1) * fh.reshape(1, -1))
            .reshape(1, h, 1, -1)
            .expand(t_count, h, w, fh.shape[0])
        )
        ew = (
            (pw.reshape(-1, 1) * fw.reshape(1, -1))
            .reshape(1, 1, w, -1)
            .expand(t_count, h, w, fw.shape[0])
        )
        half = lucid.cat([et, eh, ew], dim=-1).reshape(t_count * h * w, -1)
        angles = lucid.cat([half, half], dim=-1)
        out = (lucid.cos(angles), lucid.sin(angles))
        if len(self._tables) > 64:
            self._tables.clear()
        self._tables[key] = out
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden, bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class Attention(nn.Module):
    """Multi-head attention with RMS-normalised q/k (``q_norm``/``k_norm``)."""

    def __init__(
        self,
        dim: int,
        context_dim: int | None,
        heads: int,
        camera_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.is_self = context_dim is None
        ctx = dim if context_dim is None else context_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(ctx, dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(ctx, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        # Camera models keep the ray-token projection under `self_attn`.
        self.cam_encoder = (
            nn.Linear(camera_dim, dim, bias=False) if camera_dim else None
        )

    def _heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.head_dim)

    def qkv(
        self,
        x: Tensor,
        context: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ):
        """q, k, v as ``(B, heads, L, head_dim)``; RoPE in fp32 for self-attention."""
        context = x if context is None else context
        q = self.q_norm(self._heads(self.q_proj(x))).swapaxes(1, 2)
        k = self.k_norm(self._heads(self.k_proj(context))).swapaxes(1, 2)
        v = self._heads(self.v_proj(context)).swapaxes(1, 2)
        if self.is_self and rope is not None:
            dtype = q.dtype
            q, k = F.apply_rotary_emb(q.float(), k.float(), *rope)
            q, k = q.to(dtype), k.to(dtype)
        return q, k, v

    def out(self, attended: Tensor) -> Tensor:
        b, _, n, _ = attended.shape
        return self.output_proj(
            attended.swapaxes(1, 2).reshape(b, n, self.heads * self.head_dim)
        )


def _modulation(dim: int, lora: int, chunks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(dim, lora, bias=False),
        nn.Linear(lora, chunks * dim, bias=False),
    )


def _modulate_fp32(seq: nn.Sequential, emb: Tensor, lora: Tensor) -> Tensor:
    """``adaln_modulation(emb) + adaln_lora`` inside the fp32 region.

    SiLU runs in the input's dtype (autocast leaves it alone); both linears
    and the sum run in fp32.
    """
    h = F.silu(emb)
    return _fp32_linear(_fp32_linear(h, seq[1]), seq[2]) + lora.float()


class Block(nn.Module):
    """Self-attention, cross-attention and MLP, each adaLN-modulated and gated."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        d = cfg.model_channels
        self.layer_norm_self_attn = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        self.self_attn = Attention(d, None, cfg.num_heads, cfg.camera_dim)
        self.layer_norm_cross_attn = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        self.cross_attn = Attention(d, cfg.crossattn_emb_channels, cfg.num_heads)
        self.layer_norm_mlp = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        self.mlp = FeedForward(d, int(d * cfg.mlp_ratio))
        self.adaln_modulation_self_attn = _modulation(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = _modulation(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_mlp = _modulation(d, cfg.adaln_lora_dim, 3)

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        lora: Tensor,
        context: Tensor,
        rope: tuple[Tensor, Tensor],
        attend,
        layer: int,
        camera: Tensor | None = None,
    ) -> Tensor:
        b, t, h, w, d = x.shape
        dtype = x.dtype

        def mods(seq):
            m = _modulate_fp32(seq, emb, lora).to(dtype)
            return [c.reshape(b, t, 1, 1, d) for c in m.chunk(3, dim=-1)]

        shift_sa, scale_sa, gate_sa = mods(self.adaln_modulation_self_attn)
        shift_ca, scale_ca, gate_ca = mods(self.adaln_modulation_cross_attn)
        shift_mlp, scale_mlp, gate_mlp = mods(self.adaln_modulation_mlp)

        normed = self.layer_norm_self_attn(x) * (1 + scale_sa) + shift_sa
        if camera is not None:
            normed = normed + self.self_attn.cam_encoder(camera.to(dtype))
        q, k, v = self.self_attn.qkv(normed.reshape(b, t * h * w, d), rope=rope)
        attended = attend(layer, q, k, v, t)
        x = x + gate_sa * self.self_attn.out(attended).reshape(b, t, h, w, d)

        normed = self.layer_norm_cross_attn(x) * (1 + scale_ca) + shift_ca
        q, k, v = self.cross_attn.qkv(normed.reshape(b, t * h * w, d), context)
        attended = F.scaled_dot_product_attention(q, k, v)
        x = self.cross_attn.out(attended).reshape(b, t, h, w, d) * gate_ca + x

        normed = self.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(normed)


class FinalLayer(nn.Module):
    """adaLN-modulated LayerNorm and the patch projection, entirely in fp32."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        d = cfg.model_channels
        self.hidden = d
        self.layer_norm = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(
            d, cfg.patch_spatial**2 * cfg.patch_temporal * cfg.out_channels, bias=False
        )
        self.adaln_modulation = _modulation(d, cfg.adaln_lora_dim, 2)

    def forward(self, x: Tensor, emb: Tensor, lora: Tensor) -> Tensor:
        b, t = x.shape[:2]
        m = _modulate_fp32(self.adaln_modulation, emb, lora[..., : 2 * self.hidden])
        shift, scale = (c.reshape(b, t, 1, 1, self.hidden) for c in m.chunk(2, dim=-1))
        normed = F.layer_norm(x.float(), (self.hidden,), eps=1e-6) * (1 + scale) + shift
        return _fp32_linear(normed, self.linear)


class CosmosDiT(nn.Module):
    """``CausalCosmosModel`` (``MinimalV1LVGDiT`` with causal self-attention)."""

    def __init__(self, cfg: DiTConfig = DiTConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.model_channels
        self.x_embedder = PatchEmbed(cfg)
        self.pos_embedder = Rope3D(cfg)
        self.t_embedder = nn.Sequential(Timesteps(d), TimestepEmbedding(d))
        self.t_embedding_norm = nn.RMSNorm(d, eps=1e-6)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_blocks)])
        self.final_layer = FinalLayer(cfg)
        self.crossattn_proj = nn.Sequential(
            nn.Linear(
                cfg.crossattn_proj_in_channels, cfg.crossattn_emb_channels, bias=True
            ),
            nn.GELU(),
        )

    @property
    def dtype(self):
        return self.x_embedder.proj[1].weight.dtype

    # -- conditioning ------------------------------------------------------- #

    def project_context(self, prompt_embeds: Tensor) -> Tensor:
        """``crossattn_proj`` of the 100,352-dim text features.

        The reference applies it inside every forward; it depends on the
        prompt alone, so a caller may compute it once and reuse it.
        """
        return self.crossattn_proj(prompt_embeds.to(self.dtype))

    def _time(self, timesteps: Tensor, fp32: bool) -> tuple[Tensor, Tensor]:
        sample = self.t_embedder[0](timesteps.float() * self.cfg.timestep_scale)
        emb, lora = self.t_embedder[1](sample, fp32=fp32)
        return self.t_embedding_norm(emb), lora

    def _tokens(self, x: Tensor, cond_mask: Tensor) -> Tensor:
        """Concatenate the condition and (all-zero) padding masks, patchify."""
        b, _, t, h, w = x.shape
        dtype = self.dtype
        padding = lucid.zeros((b, 1, t, h, w), dtype=dtype, device=x.device.type)
        x = lucid.cat([x.to(dtype), cond_mask.to(dtype), padding], dim=1)
        return self.x_embedder(x)

    def _unpatchify(self, x: Tensor) -> Tensor:
        b, t, h, w, _ = x.shape
        p, pt, c = (
            self.cfg.patch_spatial,
            self.cfg.patch_temporal,
            self.cfg.out_channels,
        )
        x = x.reshape(b, t, h, w, p, p, pt, c).permute(0, 7, 1, 6, 2, 4, 3, 5)
        return x.reshape(b, c, t * pt, h * p, w * p)

    @staticmethod
    def _camera_tokens(camera: Tensor | None) -> Tensor | None:
        return None if camera is None else camera.permute(0, 2, 3, 4, 1)

    # -- forward passes ----------------------------------------------------- #

    def _run(
        self,
        tokens: Tensor,
        emb: Tensor,
        lora: Tensor,
        context: Tensor,
        rope: tuple[Tensor, Tensor],
        attend,
        camera: Tensor | None,
    ) -> Tensor:
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, emb, lora, context, rope, attend, i, camera)
        return tokens

    def forward_stream(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        cond_mask: Tensor,
        attention: StreamingAttention,
        camera: Tensor | None = None,
    ) -> Tensor:
        """``CausalCosmosModel.forward_seq``: one chunk at absolute frame ``attention.start``.

        ``x`` is ``(B, C, T, H, W)``; ``context`` is the projected prompt.
        Returns the flow prediction in fp32.
        """
        tokens = self._tokens(x, cond_mask)
        _, t, h, w, _ = tokens.shape
        emb, lora = self._time(timesteps, fp32=False)
        rope = self.pos_embedder.tables(attention.start, t, h, w)
        out = self._run(
            tokens, emb, lora, context, rope, attention, self._camera_tokens(camera)
        )
        return self._unpatchify(self.final_layer(out, emb, lora))

    def forward_full(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        cond_mask: Tensor,
        attention: MaskedAttention,
        camera: Tensor | None = None,
    ) -> Tensor:
        """``MinimalV1LVGDiT.forward``: the whole sequence at once (training)."""
        tokens = self._tokens(x, cond_mask)
        _, t, h, w, _ = tokens.shape
        emb, lora = self._time(timesteps, fp32=True)
        rope = self.pos_embedder.tables(0, t, h, w)
        out = self._run(
            tokens, emb, lora, context, rope, attention, self._camera_tokens(camera)
        )
        return self._unpatchify(self.final_layer(out, emb, lora))

    def forward_packed(
        self,
        noisy: Tensor,
        clean: Tensor,
        noisy_timesteps: Tensor,
        clean_timesteps: Tensor,
        context: Tensor,
        noisy_mask: Tensor,
        clean_mask: Tensor,
        attention: MaskedAttention,
        noisy_start: int,
        camera: Tensor | None = None,
    ) -> Tensor:
        """``forward_teacher_forcing``: score a noisy suffix after clean history.

        ``noisy`` covers the full sequence (frames before ``noisy_start`` are
        replaced by ``clean``); the history is computed without gradients, as
        in the reference. Returns the flow for every frame, zeros on the
        history.
        """
        b, _, total, _, _ = noisy.shape
        n_ctx = clean.shape[2]
        if n_ctx != noisy_start or not 0 < noisy_start < total:
            raise ValueError("packed scoring needs history before the noisy targets")
        noisy_tokens = self._tokens(noisy, noisy_mask)
        _, _, h, w, _ = noisy_tokens.shape
        with lucid.no_grad():
            clean_tokens = self._tokens(clean, clean_mask)
            clean_emb, clean_lora = self._time(clean_timesteps, fp32=False)
        noisy_emb, noisy_lora = self._time(noisy_timesteps, fp32=False)
        tokens = lucid.cat([clean_tokens, noisy_tokens[:, noisy_start:]], dim=1)
        emb = lucid.cat([clean_emb, noisy_emb[:, noisy_start:]], dim=1)
        lora = lucid.cat([clean_lora, noisy_lora[:, noisy_start:]], dim=1)
        rope = self.pos_embedder.tables(0, total, h, w)
        cam = self._camera_tokens(camera)
        attention = MaskedAttention(
            attention.frames_per_block,
            attention.independent_first_frame,
            attention.local_attn,
            attention.sink,
            detach_frames=n_ctx,
        )
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, emb, lora, context, rope, attention, i, cam)
            tokens = lucid.cat([tokens[:, :n_ctx].detach(), tokens[:, n_ctx:]], dim=1)
        out = self.final_layer(tokens[:, n_ctx:], emb[:, n_ctx:], lora[:, n_ctx:])
        suffix = self._unpatchify(out)
        prefix = lucid.zeros(
            (b, suffix.shape[1], noisy_start, suffix.shape[3], suffix.shape[4]),
            dtype=suffix.dtype,
            device=suffix.device.type,
        )
        return lucid.cat([prefix, suffix], dim=2)
