"""Context-Matched Distillation, the training side.

Ports of the reference's training objectives, on Lucid's autograd:

* :class:`ScoreModel` — ``CosmosDiffusionWrapper`` (causal): image-latent
  conditioning, then a streaming (KV), full-sequence or packed
  (teacher-forcing) pass, returning the flow and the implied ``x0``.
* :class:`SelfForcingRollout` — ``SelfForcingTrainingPipeline``: the student
  generates on policy, block by block, exactly as at inference; one randomly
  chosen denoising step per rollout keeps its graph.
* :class:`DiffusionForcingTeacher` — stage 1 (``CausalDiffusion``): the causal
  multi-step teacher learns flow matching with every block noised
  independently under a block-causal mask.
* :class:`ContextMatchedDistillation` — stage 2/3 (``ContextMatchedDistillation``):
  the DMD gradient ``s_fake - s_real`` where both scores are *causal* and see
  the scored targets packed after their history (the image latent and, across
  rollout chunks, the student's own prefix, corrupted with ``prefix_noise``),
  plus the critic's flow-matching loss.

Layouts follow the reference: videos are ``(B, F, C, H, W)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import lucid
import lucid.nn as nn

from cmd_lucid.models.cosmos_dit import (
    CosmosDiT,
    FrameKVCache,
    MaskedAttention,
    StreamingAttention,
)
from cmd_lucid.scheduler import FlowMatchSchedule

Tensor = lucid.Tensor


@dataclass
class DistillConfig:
    """The fields of ``configs/cosmos/*.yaml`` the objectives read."""

    num_frame_per_block: int = 1
    local_attn_size: int = 21
    sink_size: int = 0
    num_training_frames: int = 24
    scored_frames: int = 24  # image_or_video_shape[1]
    denoising_step_list: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    context_noise: int = 128
    prefix_noise: int = 256
    score_context_noise_frame_boundaries: list[int] = field(default_factory=list)
    score_context_noise_frame_values: list[int] = field(default_factory=list)
    real_guidance_scale: float = 4.0
    fake_guidance_scale: float = 0.0
    timestep_shift: float = 5.0
    num_train_timestep: int = 1000
    min_score_timestep: int = 0
    same_step_across_blocks: bool = True
    rollout_num_chunks: int = 1
    rollout_first_gt_latent_mean_loss_weight: float = 0.0
    rollout_first_gt_latent_std_loss_weight: float = 0.0
    seed: int = 0


class _Rng:
    """``torch.randint`` / ``torch.randn`` stand-ins on one seeded Lucid generator."""

    def __init__(self, seed: int, device: str) -> None:
        self.gen = lucid.Generator().manual_seed(seed)
        self.py = random.Random(seed)
        self.device = device

    def randint(self, low: int, high: int) -> int:
        return self.py.randrange(low, high)

    def randn(self, shape, dtype=lucid.float32) -> Tensor:
        return lucid.randn(*shape, generator=self.gen).to(self.device).to(dtype)


class ScoreModel(nn.Module):
    """``CosmosDiffusionWrapper(is_causal=True)`` around a :class:`CosmosDiT`."""

    def __init__(
        self,
        dit: CosmosDiT,
        schedule: FlowMatchSchedule,
        frames_per_block: int = 1,
        independent_first_frame: bool = True,
        local_attn: int = 21,
        sink: int = 0,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.schedule = schedule
        self.frames_per_block = frames_per_block
        self.independent_first_frame = independent_first_frame
        self.local_attn = local_attn
        self.sink = sink

    def new_cache(self) -> FrameKVCache:
        return FrameKVCache(self.dit.cfg.num_blocks, self.local_attn, self.sink)

    def _condition(self, x: Tensor, t: list[list[float]], cond: dict, apply: bool):
        """``_condition_inputs``: substitute the image latent at frame 0, set its mask, zero its timestep."""
        b, f = x.shape[:2]
        mask = lucid.zeros(
            (b, 1, f, x.shape[3], x.shape[4]), dtype=x.dtype, device=x.device.type
        )
        initial = cond.get("initial_latent")
        t = [list(row) for row in t]
        if initial is not None and apply:
            n = min(initial.shape[1], f)
            x = lucid.cat([initial[:, :n].to(x.dtype), x[:, n:]], dim=1)
            ones = lucid.ones(
                (b, 1, n, x.shape[3], x.shape[4]), dtype=x.dtype, device=x.device.type
            )
            mask = lucid.cat([ones, mask[:, :, n:]], dim=2)
            for row in t:
                row[:n] = [0.0] * n
        return x, t, mask

    def _x0(self, flow_bfchw: Tensor, xt: Tensor, t: list[list[float]]) -> Tensor:
        sig = lucid.tensor(
            [[self.schedule.sigma(v) for v in row] for row in t],
            dtype=lucid.float32,
            device=xt.device.type,
        ).reshape(xt.shape[0], xt.shape[1], 1, 1, 1)
        return xt.float() - sig * flow_bfchw

    def forward(
        self,
        noisy: Tensor,
        cond: dict,
        timestep: list[list[float]],
        kv_cache: FrameKVCache | None = None,
        current_start: int = 0,
        store_kv: bool = False,
        clean_x: Tensor | None = None,
        aug_t: list[list[float]] | None = None,
        teacher_forcing_start: int | None = None,
    ):
        """Returns ``(flow, x0)``, both ``(B, F, C, H, W)`` fp32.

        ``cond["prompt_embeds"]`` holds the text features; each model applies
        its own ``crossattn_proj`` to them, as in the reference.
        """
        ctx = self.dit.project_context(cond["prompt_embeds"])
        camera = cond.get("camera_condition")
        x, t, mask = self._condition(
            noisy, timestep, cond, apply=(kv_cache is None or current_start == 0)
        )
        t_tensor = lucid.tensor(t, dtype=lucid.float32, device=x.device.type)
        bcthw = x.permute(0, 2, 1, 3, 4)
        if kv_cache is not None:
            f = x.shape[1]
            cam = (
                None
                if camera is None
                else camera[:, :, current_start : current_start + f]
            )
            att = StreamingAttention(
                kv_cache,
                current_start,
                store_kv,
                self.frames_per_block,
                self.independent_first_frame,
            )
            flow = self.dit.forward_stream(bcthw, t_tensor, ctx, mask, att, camera=cam)
        elif clean_x is not None:
            cx, ct, cmask = self._condition(clean_x, aug_t, cond, apply=True)
            att = MaskedAttention(
                self.frames_per_block,
                self.independent_first_frame,
                self.local_attn,
                self.sink,
            )
            flow = self.dit.forward_packed(
                bcthw,
                cx.permute(0, 2, 1, 3, 4),
                t_tensor,
                lucid.tensor(ct, dtype=lucid.float32, device=x.device.type),
                ctx,
                mask,
                cmask,
                att,
                teacher_forcing_start,
                camera=camera,
            )
        else:
            att = MaskedAttention(
                self.frames_per_block,
                self.independent_first_frame,
                self.local_attn,
                self.sink,
            )
            flow = self.dit.forward_full(bcthw, t_tensor, ctx, mask, att, camera=camera)
        flow = flow.permute(0, 2, 1, 3, 4)
        return flow, self._x0(flow, x, t)


class SelfForcingRollout:
    """``SelfForcingTrainingPipeline.inference_with_trajectory``."""

    def __init__(self, generator: ScoreModel, cfg: DistillConfig, rng: _Rng) -> None:
        self.g = generator
        self.cfg = cfg
        self.rng = rng
        self.sched = generator.schedule
        self.steps = self.sched.warp(cfg.denoising_step_list)
        self.gradient_window = cfg.scored_frames - 1

    def __call__(
        self,
        noise: Tensor,
        cond: dict,
        initial_latent: Tensor | None = None,
        context_latents: Tensor | None = None,
        exit_step: int | None = None,
    ):
        cfg, sched, g = self.cfg, self.sched, self.g
        b, f = noise.shape[:2]
        cached = context_latents if context_latents is not None else initial_latent
        n_in = 0 if cached is None else cached.shape[1]
        n_out = f + n_in
        cache = g.new_cache()
        blocks: list[Tensor] = []
        if cached is not None:
            blocks.append(cached.to(noise.dtype))
            t_c = cfg.context_noise if context_latents is not None else 0
            inp = (
                cached
                if t_c == 0
                else sched.add_noise(
                    cached, self.rng.randn(cached.shape, cached.dtype), t_c
                )
            )
            with lucid.no_grad():
                g(
                    inp,
                    cond,
                    [[float(t_c)] * n_in] * b,
                    kv_cache=cache,
                    current_start=0,
                    store_kv=True,
                )
        fpb = cfg.num_frame_per_block
        sizes = [fpb] * (f // fpb)
        exit_flags = (
            [self.rng.randint(0, len(self.steps)) for _ in sizes]
            if exit_step is None
            else [exit_step] * len(sizes)
        )
        grad_from = (
            n_in if context_latents is not None else n_out - self.gradient_window
        )
        start = n_in
        for bi, n in enumerate(sizes):
            x = noise[:, start - n_in : start - n_in + n]
            ex = exit_flags[0] if cfg.same_step_across_blocks else exit_flags[bi]
            for i, t in enumerate(self.steps):
                tt = [[t] * n] * b
                if i != ex:
                    with lucid.no_grad():
                        _, x0 = g(x, cond, tt, kv_cache=cache, current_start=start)
                        x = sched.add_noise(
                            x0, self.rng.randn(x0.shape), self.steps[i + 1]
                        )
                    continue
                if start < grad_from:
                    with lucid.no_grad():
                        _, x0 = g(x, cond, tt, kv_cache=cache, current_start=start)
                else:
                    _, x0 = g(x, cond, tt, kv_cache=cache, current_start=start)
                break
            blocks.append(x0)
            commit = x0.detach()
            if cfg.context_noise > 0:
                commit = sched.add_noise(
                    commit, self.rng.randn(commit.shape), cfg.context_noise
                )
            with lucid.no_grad():
                g(
                    commit,
                    cond,
                    [[float(cfg.context_noise)] * n] * b,
                    kv_cache=cache,
                    current_start=start,
                    store_kv=True,
                )
            start += n
        output = lucid.cat(blocks, dim=1)
        ex = exit_flags[0]
        # denoised_timestep_{from,to} (used only when ts_schedule is on)
        idx = lambda v: 1000 - sched.index(v)  # noqa: E731
        if ex == len(self.steps) - 1:
            frm, to = idx(self.steps[ex]), 0
        else:
            frm, to = idx(self.steps[ex]), idx(self.steps[ex + 1])
        return output, frm, to, ex + 1


class DiffusionForcingTeacher:
    """``CausalDiffusion.generator_loss``: independent per-block noise, block-causal flow matching."""

    def __init__(self, model: ScoreModel, cfg: DistillConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.rng = _Rng(cfg.seed, model.dit.x_embedder.proj[1].weight.device.type)
        self.min_step = int(0.02 * cfg.num_train_timestep)
        self.max_step = int(0.98 * cfg.num_train_timestep)

    def _timesteps(self, b: int, f: int) -> list[list[float]]:
        """``_get_timestep(uniform=False)`` with an independent first frame, shifted and clamped."""
        rows = []
        fpb, s = self.cfg.num_frame_per_block, self.cfg.timestep_shift
        for _ in range(b):
            first = self.rng.randint(self.min_step, self.max_step)
            row = [first]
            for _ in range((f - 1) // fpb):
                row += [self.rng.randint(self.min_step, self.max_step)] * fpb
            row = [s * (v / 1000) / (1 + (s - 1) * (v / 1000)) * 1000 for v in row]
            rows.append([min(max(v, self.min_step), self.max_step) for v in row])
        return rows

    def loss(self, clean: Tensor, cond: dict) -> Tensor:
        b, f = clean.shape[:2]
        noise = self.rng.randn(clean.shape)
        t = self._timesteps(b, f)
        sched = self.model.schedule
        noisy = lucid.stack(
            [
                lucid.stack(
                    [
                        sched.add_noise(clean[i, j], noise[i, j], t[i][j])
                        for j in range(f)
                    ]
                )
                for i in range(b)
            ]
        )
        target = noise - clean.float()
        flow, _ = self.model(noisy, {**cond, "initial_latent": clean[:, :1]}, t)
        per_frame = ((flow - target) ** 2).mean(dim=(2, 3, 4))
        return per_frame[:, 1:].mean()


class ContextMatchedDistillation:
    """``ContextMatchedDistillation.generator_loss`` / ``critic_loss``."""

    def __init__(
        self,
        generator: ScoreModel,
        real: ScoreModel,
        fake: ScoreModel,
        cfg: DistillConfig,
    ) -> None:
        self.generator, self.real, self.fake = generator, real, fake
        self.cfg = cfg
        device = generator.dit.x_embedder.proj[1].weight.device.type
        self.rng = _Rng(cfg.seed, device)
        self.rollout = SelfForcingRollout(generator, cfg, self.rng)
        self.sched = generator.schedule
        self.min_step = int(0.02 * cfg.num_train_timestep)
        self.max_step = int(0.98 * cfg.num_train_timestep)
        self._rollout_states: dict[str, dict] = {}

    # -- rollout and splitting ---------------------------------------------- #

    def _generate(
        self, shape, cond: dict, initial: Tensor, chunk: int | None, phase: str
    ):
        b, _, c, h, w = shape
        if chunk is not None:
            return self._rollout_chunk(shape, cond, initial, chunk, phase)
        frames = self.cfg.num_training_frames - 1
        noise = self.rng.randn((b, frames, c, h, w))
        out, frm, to, _ = self.rollout(noise, cond, initial_latent=initial)
        scored = self.cfg.scored_frames - 1
        suffix_start = out.shape[1] - scored
        prefix = out[:, 1:suffix_start].detach()
        return out[:, suffix_start:], prefix, 1

    def _rollout_chunk(self, shape, cond, initial, chunk: int, phase: str):
        """``_run_rollout_generator``: continue a long rollout from the cached student history."""
        b, f, c, h, w = shape
        if chunk == 0:
            self._rollout_states[phase] = {"history": None, "exit_step": None}
        state = self._rollout_states[phase]
        history = state["history"]
        window = f - 1
        if history is None:
            prefix, prefix_start, context, chunk_initial, frames = (
                initial[:, :0],
                1,
                None,
                initial,
                f - 1,
            )
        else:
            prefix = history[:, 1:][:, -window:].detach()
            prefix_start = history.shape[1]
            context = history[:, -window:].detach()
            chunk_initial, frames = None, f
        noise = self.rng.randn((b, frames, c, h, w))
        out, frm, to, sim = self.rollout(
            noise,
            cond,
            initial_latent=chunk_initial,
            context_latents=context,
            exit_step=state["exit_step"],
        )
        state["exit_step"] = sim - 1
        generated = out[:, -frames:]
        state["history"] = (
            lucid.cat([initial, generated.detach()], dim=1)
            if history is None
            else lucid.cat([history, generated.detach()], dim=1)
        )
        return generated, prefix, prefix_start

    # -- scoring ------------------------------------------------------------- #

    def _prefix_timesteps(self, n: int, start: int) -> list[float]:
        bounds, values = (
            self.cfg.score_context_noise_frame_boundaries,
            self.cfg.score_context_noise_frame_values,
        )
        if not bounds:
            return [float(self.cfg.prefix_noise)] * n
        out = []
        for i in range(start, start + n):
            bucket = sum(1 for bnd in bounds if i >= bnd)  # torch.bucketize(right=True)
            out.append(float(values[bucket]))
        return out

    def _noise_prefix(self, prefix: Tensor, start: int):
        """Prefix Corruption: ``(1 - rho) x + rho eps`` at ``prefix_noise`` (or the per-frame schedule)."""
        b, n = prefix.shape[:2]
        ts = self._prefix_timesteps(n, start)
        if n == 0:
            return prefix, [[] for _ in range(b)]
        noise = self.rng.randn(prefix.shape, prefix.dtype)
        noised = lucid.stack(
            [
                lucid.stack(
                    [
                        self.sched.add_noise(prefix[i, j], noise[i, j], ts[j])
                        for j in range(n)
                    ]
                )
                for i in range(b)
            ]
        )
        return noised, [list(ts) for _ in range(b)]

    def _score(
        self,
        model: ScoreModel,
        noisy: Tensor,
        t: list[list[float]],
        cond: dict,
        history: Tensor,
        history_t: list[list[float]],
    ):
        """``_score_with_prefix``: pack [image latent | noised prefix | noisy targets], score the targets."""
        initial = cond["initial_latent"].to(noisy.dtype)
        context = lucid.cat([initial, history.to(noisy.dtype)], dim=1)
        context_t = [[0.0] + list(h) for h in history_t]
        start = context.shape[1]
        flow, x0 = model(
            lucid.cat([context, noisy], dim=1),
            cond,
            [c + list(r) for c, r in zip(context_t, t)],
            clean_x=context,
            aug_t=context_t,
            teacher_forcing_start=start,
        )
        return flow[:, start:], x0[:, start:]

    def _score_cfg(self, model, noisy, t, cond, uncond, history, history_t):
        both = {k: lucid.cat([v, uncond[k]], dim=0) for k, v in cond.items()}
        _, pred = self._score(
            model,
            lucid.cat([noisy, noisy], dim=0),
            t + t,
            both,
            lucid.cat([history, history], dim=0),
            history_t + history_t,
        )
        b = noisy.shape[0]
        return pred[:b], pred[b:]

    def _timestep(self, b: int, f: int) -> list[list[float]]:
        """Uniform DMD timestep: ``randint(0, 1000)``, shifted, clamped to ``[20, 980]``."""
        s = self.cfg.timestep_shift
        out = []
        for _ in range(b):
            v = self.rng.randint(
                self.cfg.min_score_timestep, self.cfg.num_train_timestep
            )
            if s > 1:
                v = s * (v / 1000) / (1 + (s - 1) * (v / 1000)) * 1000
            out.append([min(max(float(v), self.min_step), self.max_step)] * f)
        return out

    def _add_noise(self, x: Tensor, noise: Tensor, t: list[list[float]]) -> Tensor:
        return lucid.stack(
            [
                lucid.stack(
                    [
                        self.sched.add_noise(x[i, j], noise[i, j], t[i][j])
                        for j in range(x.shape[1])
                    ]
                )
                for i in range(x.shape[0])
            ]
        )

    # -- losses -------------------------------------------------------------- #

    def generator_loss(
        self,
        shape,
        cond: dict,
        uncond: dict,
        initial: Tensor,
        clean: Tensor | None = None,
        chunk: int | None = None,
    ):
        pred, prefix, prefix_start = self._generate(
            shape, cond, initial, chunk, "generator"
        )
        b, f = pred.shape[:2]
        with lucid.no_grad():
            t = self._timestep(b, f)
            noisy = self._add_noise(pred.detach(), self.rng.randn(pred.shape), t)
            history, history_t = self._noise_prefix(prefix, prefix_start)
            if self.cfg.fake_guidance_scale != 0.0:
                fc, fu = self._score_cfg(
                    self.fake, noisy, t, cond, uncond, history, history_t
                )
                fake = fc + (fc - fu) * self.cfg.fake_guidance_scale
            else:
                _, fake = self._score(self.fake, noisy, t, cond, history, history_t)
            rc, ru = self._score_cfg(
                self.real, noisy, t, cond, uncond, history, history_t
            )
            real = rc + (rc - ru) * self.cfg.real_guidance_scale
            grad = fake - real
            normalizer = (
                (pred.detach().float() - real)
                .abs()
                .mean(dim=(1, 2, 3, 4), keepdim=True)
            )
            grad = lucid.nan_to_num(grad / normalizer)
        target = (pred.float() - grad).detach()
        loss = 0.5 * ((pred.float() - target) ** 2).mean()
        if (
            chunk is not None
            and clean is not None
            and (
                self.cfg.rollout_first_gt_latent_mean_loss_weight
                or self.cfg.rollout_first_gt_latent_std_loss_weight
            )
        ):
            loss = loss + self._anchor_loss(pred, clean)
        return loss, {"dmd_grad_norm": grad.abs().mean().item(), "timestep": t[0][0]}

    def _anchor_loss(self, generated: Tensor, reference: Tensor) -> Tensor:
        dims = (1, 3, 4)
        g = generated.float()
        r = reference[:, 1 : 1 + g.shape[1]].detach().float()
        mean_l = (g.mean(dim=dims) - r.mean(dim=dims)).abs().mean()
        std_l = (
            (
                (g.var(dim=dims, unbiased=False) + 1e-6).sqrt()
                - (r.var(dim=dims, unbiased=False) + 1e-6).sqrt()
            )
            .abs()
            .mean()
        )
        return (
            self.cfg.rollout_first_gt_latent_mean_loss_weight * mean_l
            + self.cfg.rollout_first_gt_latent_std_loss_weight * std_l
        )

    def critic_loss(
        self, shape, cond: dict, uncond: dict, initial: Tensor, chunk: int | None = None
    ):
        with lucid.no_grad():
            gen, prefix, prefix_start = self._generate(
                shape, cond, initial, chunk, "critic"
            )
            gen = gen.detach()
            b, f = gen.shape[:2]
            t = self._timestep(b, f)
            noise = self.rng.randn(gen.shape)
            noisy = self._add_noise(gen, noise, t)
            history, history_t = self._noise_prefix(prefix, prefix_start)
        flow, _ = self._score(self.fake, noisy, t, cond, history, history_t)
        loss = ((flow - (noise - gen.float())) ** 2).mean()
        return loss, {"critic_timestep": t[0][0]}
