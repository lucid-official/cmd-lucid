"""Causal image-to-video inference: ``pipeline/causal_inference.py`` of the reference.

:meth:`CMDPipeline.stream` is ``CausalInferencePipeline.inference`` turned into
a generator: it yields each latent block the moment it is final, so a caller
can decode and show frames while later blocks are still being denoised. The
arithmetic and the order of random draws follow the reference:

1. the image latent is committed to the KV cache at ``t = 0`` with the
   condition mask set (frame 0);
2. each block of ``C`` latent frames starts from noise and is denoised at the
   warped timesteps ``[1000, 937.5, 833.3, 625]``: the network's flow is turned
   into ``x0`` and, between steps, ``x0`` is re-noised with fresh noise to the
   next timestep. Teachers instead integrate 35 UniPC steps with CFG 7;
3. the clean block is re-noised to ``t = 128`` (``context_noise``) and run once
   more with ``store`` set, which writes its K/V for later blocks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import lucid

from cmd_lucid import catalog
from cmd_lucid.catalog import Variant
from cmd_lucid.models.cosmos_dit import CosmosDiT, DiTConfig, FrameKVCache, StreamingAttention
from cmd_lucid.models.wan_vae import WanVAE
from cmd_lucid.scheduler import FlowMatchSchedule, FlowUniPC

Tensor = lucid.Tensor

LATENT_CHANNELS, LATENT_H, LATENT_W = 16, 60, 104
NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


def build_on(device: str, dtype, factory: Callable[[], lucid.nn.Module]) -> lucid.nn.Module:
    """Construct a module directly on ``device``/``dtype``; its random init is never evaluated."""
    prev = lucid.get_default_device(), lucid.get_default_dtype()
    lucid.set_default_device(device)
    lucid.set_default_dtype(dtype)
    try:
        return factory()
    finally:
        lucid.set_default_device(prev[0])
        lucid.set_default_dtype(prev[1])


def load_dit(variant: Variant, device: str = "metal", dtype=lucid.bfloat16) -> CosmosDiT:
    cfg = DiTConfig(camera_dim=6 * 16 * 16 if variant.camera else None)
    model = build_on(device, dtype, lambda: CosmosDiT(cfg))
    weights = lucid.load_safetensors(str(variant.path), device=device)
    weights = {k: (v if v.dtype == dtype else v.to(dtype)) for k, v in weights.items()}
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval()


def load_vae(device: str = "metal", dtype=lucid.bfloat16, path: Path | None = None) -> WanVAE:
    model = build_on(device, dtype, WanVAE)
    weights = lucid.load_safetensors(str(path or catalog.vae_path()), device=device)
    model.load_state_dict({k: v.to(dtype) for k, v in weights.items()}, strict=True, assign=True)
    return model.eval()


class NoiseSource:
    """The pipeline's random draws, in the reference's order.

    ``initial(shape)`` is the per-video noise (drawn in bf16 by the
    reference); ``like(x)`` the fp32 noise of each re-noising step. A
    parity test can pass recorded tensors instead.
    """

    def __init__(self, seed: int = 0, device: str = "metal") -> None:
        self.generator = lucid.Generator().manual_seed(seed)
        self.device = device

    def initial(self, shape: tuple[int, ...], dtype) -> Tensor:
        return lucid.randn(*shape, generator=self.generator).to(self.device).to(dtype)

    def like(self, x: Tensor) -> Tensor:
        """``torch.randn_like(x)``: same shape and dtype."""
        return lucid.randn(*x.shape, generator=self.generator).to(self.device).to(x.dtype)


class RecordedNoise(NoiseSource):
    """Replays a list of tensors (for comparisons against recorded runs)."""

    def __init__(self, tensors: list[Tensor], device: str = "metal") -> None:
        self.queue = list(tensors)
        self.device = device

    def initial(self, shape, dtype) -> Tensor:
        return self.queue.pop(0).to(self.device).to(dtype).reshape(shape)

    def like(self, x: Tensor) -> Tensor:
        return self.queue.pop(0).to(self.device).to(x.dtype).reshape(tuple(x.shape))


@dataclass
class Block:
    """One finished causal block."""

    start: int  # absolute latent frame index of its first frame
    latents: Tensor  # (1, 16, n, 60, 104), in the output dtype (bf16 by default)
    seconds: float  # wall time spent on it


@dataclass
class StreamSettings:
    """Per-run knobs; the defaults are the released configs."""

    denoising_steps: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    context_noise: int = 128
    teacher_steps: int = 35
    guidance_scale: float = 7.0
    timestep_shift: float = 5.0


class CMDPipeline:
    """A loaded CMD model (DiT + VAE) with the streaming causal sampler."""

    def __init__(self, variant: Variant, dit: CosmosDiT, vae: WanVAE, device: str = "metal") -> None:
        self.variant = variant
        self.dit = dit
        self.vae = vae
        self.device = device
        self.schedule = FlowMatchSchedule()

    @classmethod
    def load(cls, key: str, device: str = "metal", dtype=lucid.bfloat16, vae_dtype=None) -> "CMDPipeline":
        variant = catalog.VARIANTS[key]
        if not variant.path.exists():
            raise FileNotFoundError(f"{variant.path} is missing; run `python -m cmd_lucid.download {key}`")
        return cls(variant, load_dit(variant, device, dtype), load_vae(device, vae_dtype or dtype), device)

    @property
    def dtype(self):
        return self.dit.dtype

    # -- conditioning ------------------------------------------------------- #

    def encode_image(self, pixels: Tensor) -> Tensor:
        """``(1, 3, 1, 480, 832)`` in ``[-1, 1]`` -> image latent ``(1, 16, 1, 60, 104)`` in the model dtype."""
        with lucid.no_grad():
            latent = self.vae.encode(pixels.to(self.device).to(self.vae.dtype))
        return latent.float().to(self.dtype)

    def context(self, prompt_embeds: Tensor) -> Tensor:
        with lucid.no_grad():
            return self.dit.project_context(prompt_embeds.to(self.device))

    # -- sampling ----------------------------------------------------------- #

    def _flow(self, x, t, ctx, start, store, cache, camera, cond=None) -> Tensor:
        n = x.shape[2]
        mask = cond if cond is not None else lucid.zeros((x.shape[0], 1, n, LATENT_H, LATENT_W), dtype=self.dtype, device=self.device)
        steps = lucid.full((x.shape[0], n), float(t), dtype=lucid.float32, device=self.device)
        cam = None if camera is None else camera(start, n)
        attention = StreamingAttention(cache, start, store, self.variant.block)
        return self.dit.forward_stream(x, steps, ctx, mask, attention, camera=cam)

    def stream(
        self,
        image_latent: Tensor,
        prompt_embeds: Tensor,
        num_frames: int | None = None,
        noise: NoiseSource | None = None,
        camera: Callable[[int, int], Tensor] | None = None,
        negative_embeds: Tensor | None = None,
        settings: StreamSettings = StreamSettings(),
    ) -> Iterator[Block]:
        """Generate ``num_frames`` latent frames (image frame included) block by block.

        ``camera(start, count)`` returns ray tokens ``(1, 1536, count, 30, 52)``
        for camera models. ``negative_embeds`` is needed by teachers (CFG).
        ``num_frames`` defaults to the checkpoint's length; longer runs keep
        going with the same local attention window.
        """
        v = self.variant
        total = num_frames or v.latent_frames
        if (total - 1) % v.block:
            raise ValueError(f"{v.key}: frames after the image must be a multiple of {v.block}")
        if v.camera and camera is None:
            raise ValueError(f"{v.key} needs a camera trajectory")
        noise = noise or NoiseSource(device=self.device)
        cache = FrameKVCache(self.dit.cfg.num_blocks, local_attn=v.local_attn)
        sched = self.schedule
        with lucid.no_grad():
            ctx = self.context(prompt_embeds)
            if v.teacher:
                if negative_embeds is None:
                    raise ValueError("teachers sample with CFG and need negative_embeds")
                ctx = lucid.cat([ctx, self.context(negative_embeds)], dim=0)
            init_noise = noise.initial((1, LATENT_CHANNELS, total - 1, LATENT_H, LATENT_W), self.dtype)
            # 1. commit the image frame
            first = image_latent.to(self.device).to(self.dtype)
            cond = lucid.ones((1, 1, 1, LATENT_H, LATENT_W), dtype=self.dtype, device=self.device)
            self._flow(first, 0, ctx[:1], 0, True, cache, camera, cond=cond)
            lucid.eval(*[k for layer in cache.frames for (k, _) in layer.values()])
            yield Block(0, first, 0.0)
            # 2-3. causal blocks
            warped = sched.warp(settings.denoising_steps)
            start = 1
            while start < total:
                t0 = time.time()
                n = v.block
                x = init_noise[:, :, start - 1 : start - 1 + n]
                if v.teacher:
                    x0 = self._teacher_block(x, ctx, start, cache, camera, settings)
                else:
                    for i, t in enumerate(warped):
                        flow = self._flow(x, t, ctx, start, False, cache, camera)
                        x0 = sched.x0_from_flow(flow, x, t)
                        if i < len(warped) - 1:
                            x = sched.add_noise(x0, noise.like(x0), warped[i + 1])
                            lucid.eval(x)
                out = x0.to(self.dtype)
                # Commit at the context-noise level (both students and teachers use 128).
                t_ctx = settings.context_noise
                commit = sched.add_noise(x0, noise.like(x0), t_ctx) if t_ctx > 0 else x0
                self._flow(commit, t_ctx, ctx[:1], start, True, cache, camera)
                lucid.eval(out, *[k for (k, _) in cache.frames[-1].values()])
                yield Block(start, out, time.time() - t0)
                start += n

    def _teacher_block(self, x, ctx, start, cache, camera, settings: StreamSettings) -> Tensor:
        """The UniPC branch of ``CausalInferencePipeline.inference`` (velocity-space CFG)."""
        solver = FlowUniPC(settings.teacher_steps, shift=settings.timestep_shift)
        for t in solver.timesteps:
            both = lucid.cat([x, x], dim=0)
            flow = self._flow(both, t, ctx, start, False, cache, camera)
            cond, uncond = flow[:1], flow[1:]
            flow = cond + settings.guidance_scale * (cond - uncond)
            x = solver.step(flow, x)
            lucid.eval(x)
        return x  # stays in the noise dtype, as in the reference

    # -- decoding ----------------------------------------------------------- #

    def decoder(self) -> "StreamingDecoder":
        return StreamingDecoder(self.vae)


class StreamingDecoder:
    """Decodes latent blocks as they arrive; the first block yields 1 frame, later ones 4 per latent."""

    def __init__(self, vae: WanVAE) -> None:
        self.vae = vae
        self.state = vae.new_state()

    def __call__(self, latents: Tensor) -> Tensor:
        with lucid.no_grad():
            out = self.vae.decode_stream(latents.to(self.vae.dtype), self.state)
            lucid.eval(out)
        return out[0]  # (3, T, H, W)
