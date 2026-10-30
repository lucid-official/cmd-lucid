"""``lucid.compile`` and ``lucid.coreml`` acceleration.

**Streaming VAE decoder (lucid.compile).** Decoding one latent frame is a
pure function of the frame and the two-frame histories of the decoder's ~30
causal convolutions. :class:`CompiledDecoder` traces that function twice —
the first frame (no temporal upsampling, zero histories) and the steady
state — so MPSGraph fuses the norm/SiLU/conv chains that dominate the
480x832 tail. Shapes never change, so each program compiles once.

**VAE encoder (lucid.coreml).** :func:`export_encoder` writes the
single-image encode as a Core ML package; :class:`CoreMLEncoder` runs it on the
Neural Engine (fp16), keeping the GPU free.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LUCID_COMPILE_MAX_CACHE", "256")

import lucid  # noqa: E402
import lucid.nn as nn  # noqa: E402

from cmd_lucid.models.wan_vae import (
    CausalConv3d,
    StreamState,
    WanVAE,
    _run,
)  # noqa: E402

Tensor = lucid.Tensor


def _history_slots(vae: WanVAE) -> list[CausalConv3d]:
    return [
        m for m in vae.decoder.modules() if isinstance(m, CausalConv3d) and m.time_pad
    ]


class _DecodeFirst(nn.Module):
    """First latent frame: zero histories, temporal upsamplers skipped. Returns pixels and histories."""

    def __init__(self, vae: WanVAE) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.slots = [m.slot for m in _history_slots(vae)]

    def forward(self, x: Tensor):
        state = StreamState(sync=False)
        out = _run(self.decoder.layers(), x, state)
        return (out, *[state.hist[s] for s in self.slots])


class _DecodeStep(nn.Module):
    """A later latent frame: histories in, pixels and new histories out."""

    def __init__(self, vae: WanVAE) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.slots = [m.slot for m in _history_slots(vae)]

    def forward(self, x: Tensor, *hists: Tensor):
        state = StreamState(hist=dict(zip(self.slots, hists)), first=False, sync=False)
        out = _run(self.decoder.layers(), x, state)
        return (out, *[state.hist[s] for s in self.slots])


class CompiledDecoder:
    """Drop-in for :class:`cmd_lucid.pipeline.StreamingDecoder`, through two compiled programs."""

    def __init__(self, vae: WanVAE) -> None:
        self.vae = vae
        self.first = lucid.compile(_DecodeFirst(vae).eval())
        self.step = lucid.compile(_DecodeStep(vae).eval())
        self.hists: list[Tensor] | None = None

    def reset(self) -> None:
        self.hists = None

    def __call__(self, latents: Tensor) -> Tensor:
        vae = self.vae
        with lucid.no_grad():
            mean, inv_std = vae._scale(latents)
            x = vae.conv2((latents / inv_std + mean).to(vae.dtype))
            outs = []
            for i in range(x.shape[2]):
                frame = x[:, :, i : i + 1]
                if self.hists is None:
                    out, *self.hists = self.first(frame)
                else:
                    out, *self.hists = self.step(frame, *self.hists)
                outs.append(out)
            pixels = outs[0] if len(outs) == 1 else lucid.cat(outs, dim=2)
            pixels = pixels.float().clamp(-1.0, 1.0)
            lucid.eval(pixels, *self.hists)
        return pixels[0]


# --------------------------------------------------------------------------- #
# Core ML: the VAE encoder on the Neural Engine
# --------------------------------------------------------------------------- #


class _EncodeImage(nn.Module):
    """Single-frame encode to normalised latents (the I2V conditioning path)."""

    def __init__(self, vae: WanVAE) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.conv1 = vae.conv1
        self.z = vae.cfg.z_dim
        self.vae = vae

    def forward(self, image: Tensor) -> Tensor:
        out = _run(self.encoder.layers(), image, None)
        mu = self.conv1(out)[:, : self.z]
        mean, inv_std = self.vae._scale(mu)
        return (mu - mean) * inv_std


def export_encoder(vae_fp32: WanVAE, path: Path) -> "lucid.coreml.CoreMLModel":
    """Write the image encoder as a float16 Core ML program."""
    import lucid.coreml as cml

    example = lucid.zeros(1, 3, 1, 480, 832)
    return cml.export(
        _EncodeImage(vae_fp32.to("cpu").float()).eval(),
        example,
        str(path),
        precision=cml.Precision.FLOAT16,
        compute_units=cml.ComputeUnits.CPU_AND_NE,
    )
