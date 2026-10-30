"""CMD end to end at toy scale: teacher pre-training, then context-matched distillation.

    python -m cmd_lucid.distill.toy --out outputs/toy

The released models were distilled on 8 GPUs from three 2B networks plus a
7B text encoder; that does not fit a 36 GB Mac. This runs the *same* code
(``cmd_lucid.distill``) on a 4-block DiT and synthetic latent videos: a
Gaussian blob drifting one cell per frame. The "prompt" (a fixed random
embedding) says *horizontal* or *vertical*; the direction along that axis is a
coin flip, so every future is one of two modes. A few-step sampler that has
not been distilled blends the modes; distribution matching is meant to fix
exactly that.

The metric is the mode error: the RMSE of a sample against the nearer of its
two valid futures (the image frame excluded). Reported for the multi-step
causal teacher (35 UniPC steps), the four-step student right after
initialisation from the teacher, and the student after CMD. The toy teacher
is trained without prompt dropout, so no CFG is used anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import lucid

from cmd_lucid.distill.cmd import DistillConfig, ScoreModel, SelfForcingRollout, _Rng
from cmd_lucid.distill.trainer import DistillationTrainer, OptimConfig, TeacherTrainer
from cmd_lucid.models.cosmos_dit import CosmosDiT, DiTConfig
from cmd_lucid.scheduler import FlowMatchSchedule, FlowUniPC

H, W, C, FRAMES, TOKENS, TEXT_DIM = 8, 12, 16, 7, 4, 96
AXES = [[(0, 1), (0, -1)], [(1, 0), (-1, 0)]]  # horizontal: right/left, vertical: down/up


class BlobWorld:
    """Latent videos of a drifting blob, and fixed embeddings for the four directions."""

    def __init__(self, seed: int = 0, device: str = "metal") -> None:
        g = lucid.Generator().manual_seed(seed)
        self.device = device
        self.palette = lucid.randn(C, generator=g).to(device)
        self.background = (lucid.randn(C, generator=g) * 0.2).to(device)
        self.prompts = [(lucid.randn(1, TOKENS, TEXT_DIM, generator=g)).to(device) for _ in AXES]
        self.negative = lucid.randn(1, TOKENS, TEXT_DIM, generator=g).to(device)
        self.rng = _Rng(seed + 1, device)

    def video(self, y0: float, x0: float, axis: int, sign: int) -> lucid.Tensor:
        dy, dx = AXES[axis][sign]
        ys = lucid.arange(H, dtype=lucid.float32, device=self.device).reshape(H, 1)
        xs = lucid.arange(W, dtype=lucid.float32, device=self.device).reshape(1, W)
        frames = []
        for t in range(FRAMES):
            cy, cx = (y0 + dy * t) % H, (x0 + dx * t) % W
            ddy = lucid.minimum((ys - cy).abs(), H - (ys - cy).abs())
            ddx = lucid.minimum((xs - cx).abs(), W - (xs - cx).abs())
            blob = lucid.exp(-(ddy**2 + ddx**2) / (2 * 1.1**2))
            frames.append(blob.reshape(1, H, W) * self.palette.reshape(C, 1, 1) * 2.0 + self.background.reshape(C, 1, 1))
        return lucid.stack(frames)  # (F, C, H, W)

    def batch(self, size: int) -> dict:
        vids, alts, axes = [], [], []
        for _ in range(size):
            axis, sign = self.rng.randint(0, 2), self.rng.randint(0, 2)
            y, x = self.rng.randint(0, H), self.rng.randint(0, W)
            vids.append(self.video(y, x, axis, sign))
            alts.append(self.video(y, x, axis, 1 - sign))
            axes.append(axis)
        return {
            "latents": lucid.stack(vids),
            "alternate": lucid.stack(alts),
            "cond": {"prompt_embeds": lucid.cat([self.prompts[a] for a in axes])},
            "uncond": {"prompt_embeds": self.negative.expand(size, TOKENS, TEXT_DIM)},
        }


def tiny_dit(seed: int, device: str) -> CosmosDiT:
    lucid.manual_seed(seed)
    cfg = DiTConfig(model_channels=128, num_blocks=4, num_heads=2, adaln_lora_dim=32,
                    crossattn_proj_in_channels=TEXT_DIM, crossattn_emb_channels=64)
    return CosmosDiT(cfg).to(device)


def student_sample(model: ScoreModel, cfg: DistillConfig, batch: dict, seed: int) -> lucid.Tensor:
    """Four-step causal rollout at inference settings (the training rollout with the last exit step)."""
    rng = _Rng(seed, batch["latents"].device.type)
    rollout = SelfForcingRollout(model, cfg, rng)
    lat = batch["latents"]
    cond = {**batch["cond"], "initial_latent": lat[:, :1]}
    with lucid.no_grad():
        out, *_ = rollout(rng.randn((lat.shape[0], FRAMES - 1, C, H, W)), cond, initial_latent=lat[:, :1],
                          exit_step=len(cfg.denoising_step_list) - 1)
    return out


def teacher_sample(model: ScoreModel, cfg: DistillConfig, batch: dict, seed: int, steps: int = 35, guidance: float = 0.0):
    """Causal multi-step sampling: UniPC (with velocity-space CFG if ``guidance``), then a context-noised commit per frame."""
    rng = _Rng(seed, batch["latents"].device.type)
    lat = batch["latents"]
    b = lat.shape[0]
    sched = model.schedule
    cond = {**batch["cond"], "initial_latent": lat[:, :1]}
    both = {"prompt_embeds": lucid.cat([batch["cond"]["prompt_embeds"], batch["uncond"]["prompt_embeds"]]),
            "initial_latent": lucid.cat([lat[:, :1], lat[:, :1]])}
    cache = model.new_cache()
    frames = [lat[:, :1]]
    with lucid.no_grad():
        model(lat[:, :1], cond, [[0.0]] * b, kv_cache=cache, current_start=0, store_kv=True)
        for f in range(1, FRAMES):
            x = rng.randn((b, 1, C, H, W))
            solver = FlowUniPC(steps)
            for t in solver.timesteps:
                flow, _ = model(lucid.cat([x, x]), both, [[float(t)]] * (2 * b), kv_cache=cache, current_start=f)
                v = flow[:b] + guidance * (flow[:b] - flow[b:])
                x = solver.step(v, x)
            frames.append(x)
            commit = sched.add_noise(x, rng.randn(x.shape), cfg.context_noise)
            model(commit, cond, [[float(cfg.context_noise)]] * b, kv_cache=cache, current_start=f, store_kv=True)
    return lucid.cat(frames, dim=1)


def error(sample: lucid.Tensor, batch: dict) -> float:
    """Mode error: per-sample RMSE to the nearer valid future, averaged (image frame excluded)."""
    s = sample[:, 1:].float()
    a = ((s - batch["latents"][:, 1:].float()) ** 2).mean(dim=(1, 2, 3, 4))
    b = ((s - batch["alternate"][:, 1:].float()) ** 2).mean(dim=(1, 2, 3, 4))
    return lucid.minimum(a, b).sqrt().mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-steps", type=int, default=3000)
    p.add_argument("--distill-steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-critic", type=float, default=5e-5)
    p.add_argument("--device", default="metal")
    p.add_argument("--out", default="outputs/toy")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = BlobWorld(0, args.device)
    cfg = DistillConfig(num_frame_per_block=1, local_attn_size=FRAMES, num_training_frames=FRAMES,
                        scored_frames=FRAMES, real_guidance_scale=0.0)
    sched = FlowMatchSchedule()
    teacher = ScoreModel(tiny_dit(0, args.device), sched, 1, True, cfg.local_attn_size)
    evaluation = world.batch(32)
    report: dict = {"config": vars(args)}

    t0 = time.time()
    trainer = TeacherTrainer(teacher, cfg, OptimConfig(lr=3e-4, max_grad_norm=1.0))
    for step in range(args.teacher_steps):
        log = trainer.train_step(world.batch(args.batch))
        if step % 250 == 0 or step == args.teacher_steps - 1:
            print(f"teacher {step:5d}  flow loss {log['loss']:.4f}", flush=True)
    report["teacher_train_seconds"] = time.time() - t0
    report["teacher_mode_error"] = error(teacher_sample(teacher, cfg, evaluation, 1), evaluation)
    print(f"teacher (35 UniPC steps) mode error {report['teacher_mode_error']:.4f}", flush=True)

    distill = DistillationTrainer.from_teacher(
        teacher, cfg, OptimConfig(lr=args.lr, lr_critic=args.lr_critic, max_grad_norm=1.0, max_grad_norm_critic=1.0)
    )
    report["student_init_mode_error"] = error(student_sample(distill.cmd.generator, cfg, evaluation, 2), evaluation)
    print(f"student at init (4 steps) mode error {report['student_init_mode_error']:.4f}", flush=True)
    history = []
    t0 = time.time()
    for step in range(args.distill_steps):
        log = distill.train_step(lambda: world.batch(args.batch))
        if log["phase"] == "generator" and step % 50 == 0:
            err = error(student_sample(distill.cmd.generator, cfg, evaluation, 2), evaluation)
            history.append({"step": step, "mode_error": err, "dmd_grad": log["dmd_grad_norm"]})
            print(f"distill {step:5d}  student mode error {err:.4f}  |dmd grad| {log['dmd_grad_norm']:.4f}", flush=True)
    report["distill_seconds"] = time.time() - t0
    report["student_mode_error"] = error(student_sample(distill.cmd.generator, cfg, evaluation, 2), evaluation)
    report["history"] = history
    print(f"student after CMD (4 steps) mode error {report['student_mode_error']:.4f}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
