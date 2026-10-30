"""Wan2.1 video VAE (the ``tokenizer.pth`` of Cosmos-Predict2.5), streaming.

A port of ``wan/modules/vae.py``. Parameter names match ``Wan2.1_VAE.pth``.

The reference threads a ``feat_cache`` list through every causal 3-D
convolution while it walks a video chunk by chunk (1, 4, 4, ... frames when
encoding; one latent frame at a time when decoding). That bookkeeping is the
same as giving each temporal-kernel-3 convolution a two-frame *history* that
starts as zeros: its input is ``cat(history, x)`` and the new history is the
last two frames of that input. The temporal resamplers are the exceptions —
skipped on the first chunk, then fed one remembered frame (encoder) or a
zero-started history (decoder). :class:`StreamState` holds those histories,
so frames can be decoded as the DiT produces them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

Tensor = lucid.Tensor

LATENT_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]  # fmt: skip
LATENT_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]  # fmt: skip


@dataclass(frozen=True)
class VAEConfig:
    dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)


@dataclass
class StreamState:
    """Histories of the causal convolutions of one encoder or decoder stream."""

    hist: dict[int, Tensor] = field(default_factory=dict)
    first: bool = True  # the next call carries the first chunk


class CausalConv3d(nn.Conv3d):
    """``nn.Conv3d`` whose temporal front padding is the stream's history."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, padding=0, stride=1) -> None:
        k = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        p = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        super().__init__(in_ch, out_ch, k, stride=stride, padding=(0, p[1], p[2]))
        self.time_pad = 2 * p[0]
        self.slot = -1

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        if self.time_pad == 0:
            return self(x)
        hist = None if state is None else state.hist.get(self.slot)
        if hist is None:
            b, c, _, h, w = x.shape
            hist = lucid.zeros((b, c, self.time_pad, h, w), dtype=x.dtype, device=x.device.type)
        xin = lucid.cat([hist, x], dim=2)
        if state is not None:
            state.hist[self.slot] = xin[:, :, -self.time_pad :]
        return self(xin)


class RMSNorm(nn.Module):
    """Wan ``RMS_norm``: L2-normalise over channels, times ``sqrt(C) * gamma``."""

    def __init__(self, dim: int, images: bool = False) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(lucid.ones(dim, *((1, 1) if images else (1, 1, 1))))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim), nn.SiLU(), nn.Dropout(0.0), CausalConv3d(out_dim, out_dim, 3, padding=1),
        )  # fmt: skip
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer.stream(x, state) if isinstance(layer, CausalConv3d) else layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention over each frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def stream(self, x: Tensor, state: StreamState | None = None) -> Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        qkv = self.to_qkv(self.norm(x)).reshape(b * t, 1, c * 3, h * w).permute(0, 1, 3, 2)
        q, k, v = qkv[..., :c], qkv[..., c : 2 * c], qkv[..., 2 * c :]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.reshape(b * t, h * w, c).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def _per_frame(x: Tensor, fn) -> Tensor:
    b, c, t, h, w = x.shape
    y = fn(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
    c2, h2, w2 = y.shape[1:]
    return y.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4)


class Resample(nn.Module):
    """Spatial ×2 resampling, with a causal temporal ×2 for the ``*3d`` modes."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            # ZeroPad2d((0, 1, 0, 1)) then a stride-2 conv; slot 0 holds no weights.
            self.resample = nn.Sequential(nn.Identity(), nn.Conv2d(dim, dim, 3, stride=2))
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))

    def _spatial(self, x: Tensor) -> Tensor:
        if self.mode.startswith("upsample"):
            return self.resample(x)
        return self.resample[1](F.pad(x, (0, 1, 0, 1)))

    def stream(self, x: Tensor, state: StreamState | None) -> Tensor:
        b, c, t, h, w = x.shape
        first = state is None or state.first
        if self.mode == "upsample3d" and not first:
            x = self.time_conv.stream(x, state)
            x = x.reshape(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5).reshape(b, c, t * 2, h, w)
        x = _per_frame(x, self._spatial)
        if self.mode == "downsample3d":
            if first:
                if state is not None:
                    state.hist[self.time_conv.slot] = x[:, :, -1:]
            else:
                prev = state.hist[self.time_conv.slot]
                state.hist[self.time_conv.slot] = x[:, :, -1:]
                x = self.time_conv(lucid.cat([prev, x], dim=2))
        return x


def _run(layers: list[nn.Module], x: Tensor, state: StreamState | None) -> Tensor:
    for layer in layers:
        if isinstance(layer, nn.Sequential):  # head: norm, SiLU, conv
            for sub in layer:
                x = sub.stream(x, state) if isinstance(sub, CausalConv3d) else sub(x)
        else:
            x = layer.stream(x, state)
    return x


class Encoder3d(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        dims = [cfg.dim * u for u in [1, *cfg.dim_mult]]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        downs: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(cfg.num_res_blocks):
                downs.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(cfg.dim_mult) - 1:
                downs.append(Resample(out_dim, "downsample3d" if cfg.temperal_downsample[i] else "downsample2d"))
        self.downsamples = nn.Sequential(*downs)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim))
        self.head = nn.Sequential(RMSNorm(out_dim), nn.SiLU(), CausalConv3d(out_dim, cfg.z_dim * 2, 3, padding=1))

    def layers(self) -> list[nn.Module]:
        return [self.conv1, *self.downsamples, *self.middle, self.head]


class Decoder3d(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        mult = list(cfg.dim_mult)
        up_t = list(cfg.temperal_downsample)[::-1]
        dims = [cfg.dim * u for u in [mult[-1], *mult[::-1]]]
        self.conv1 = CausalConv3d(cfg.z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0]))
        ups: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim //= 2
            for _ in range(cfg.num_res_blocks + 1):
                ups.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(mult) - 1:
                ups.append(Resample(out_dim, "upsample3d" if up_t[i] else "upsample2d"))
        self.upsamples = nn.Sequential(*ups)
        self.head = nn.Sequential(RMSNorm(out_dim), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def layers(self) -> list[nn.Module]:
        return [self.conv1, *self.middle, *self.upsamples, self.head]


class WanVAE(nn.Module):
    """Encoder, decoder and the latent normalisation of ``CosmosVAEWrapper``.

    ``scale`` follows the reference exactly: ``mean`` and ``1 / std`` are
    built in the dtype of the latents being (de)normalised, so a bf16 run
    computes ``1 / std`` in bf16 as the reference does.
    """

    def __init__(self, cfg: VAEConfig = VAEConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder3d(cfg)
        self.conv1 = CausalConv3d(cfg.z_dim * 2, cfg.z_dim * 2, 1)
        self.conv2 = CausalConv3d(cfg.z_dim, cfg.z_dim, 1)
        self.decoder = Decoder3d(cfg)
        for part in (self.encoder, self.decoder):
            slot = 0
            for m in part.modules():
                if isinstance(m, CausalConv3d) and (m.time_pad or m.stride[0] == 2):
                    m.slot = slot
                    slot += 1

    @property
    def dtype(self):
        return self.conv2.weight.dtype

    def _scale(self, like: Tensor) -> tuple[Tensor, Tensor]:
        dev, dt = like.device.type, like.dtype
        mean = lucid.tensor(LATENT_MEAN, dtype=lucid.float32, device=dev).to(dt).reshape(1, -1, 1, 1, 1)
        std = lucid.tensor(LATENT_STD, dtype=lucid.float32, device=dev).to(dt)
        return mean, (1.0 / std).reshape(1, -1, 1, 1, 1)

    # -- encode ------------------------------------------------------------ #

    def encode(self, video: Tensor) -> Tensor:
        """Pixels ``(B, 3, T, H, W)`` in ``[-1, 1]`` -> normalised latents ``(B, 16, 1+(T-1)//4, H/8, W/8)``.

        Chunks of 1, 4, 4, ... frames, as ``WanVAE_.encode``. Returns the
        latents in the input's dtype (the reference returns ``mu`` before its
        caller's ``.float()``).
        """
        state = StreamState()
        t = video.shape[2]
        outs = []
        for i in range(1 + (t - 1) // 4):
            chunk = video[:, :, :1] if i == 0 else video[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            outs.append(_run(self.encoder.layers(), chunk.to(self.dtype), state))
            state.first = False
        out = outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)
        mu = self.conv1(out)[:, : self.cfg.z_dim]
        mean, inv_std = self._scale(mu)
        return (mu - mean) * inv_std

    # -- decode ------------------------------------------------------------ #

    def new_state(self) -> StreamState:
        return StreamState()

    def decode_stream(self, z: Tensor, state: StreamState) -> Tensor:
        """Normalised latent frames ``(B, 16, T, h, w)`` -> pixels in ``[-1, 1]``.

        Continues ``state``; returns ``1 + 4(T-1)`` frames on the first call
        and ``4T`` after. Feeding frames together or one by one is the same
        computation; the first latent frame always runs on its own.
        """
        mean, inv_std = self._scale(z)
        x = self.conv2((z / inv_std + mean).to(self.dtype))
        outs = []
        if state.first:
            outs.append(_run(self.decoder.layers(), x[:, :, :1], state))
            state.first = False
            x = x[:, :, 1:]
        for i in range(x.shape[2]):
            outs.append(_run(self.decoder.layers(), x[:, :, i : i + 1], state))
        out = outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)
        return out.float().clamp(-1.0, 1.0)

    def decode(self, z: Tensor) -> Tensor:
        """``WanVAE_.decode``: every latent frame, from a fresh state."""
        return self.decode_stream(z, self.new_state())
