"""Training loops: stage 1 (causal teacher) and stage 2/3 (context-matched distillation).

Mirrors ``trainer/diffusion.py`` and ``trainer/distillation.py``: AdamW with
``beta1 = 0``, gradient-norm clipping, and in distillation one generator
update every ``dfake_gen_update_ratio`` steps with critic updates in between.
A long-video run (``rollout_num_chunks > 1``) walks the same batch through
consecutive rollout chunks before drawing the next one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterator

import lucid
import lucid.nn as nn
import lucid.optim as optim

from cmd_lucid.distill.cmd import (
    ContextMatchedDistillation,
    DiffusionForcingTeacher,
    DistillConfig,
    ScoreModel,
)

Batch = dict  # {"latents": (B, F, C, H, W), "cond": {...}, "uncond": {...}}


@dataclass
class OptimConfig:
    lr: float = 2e-6
    lr_critic: float = 4e-7
    betas: tuple[float, float] = (0.0, 0.999)
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0
    max_grad_norm_critic: float = 10.0
    dfake_gen_update_ratio: int = 5
    ema_weight: float = 0.0
    ema_start_step: int = 200


class EMA:
    """``EMA_FSDP`` without sharding: ``shadow = w * shadow + (1 - w) * param``."""

    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in module.named_parameters()}

    def update(self, module: nn.Module) -> None:
        with lucid.no_grad():
            for n, p in module.named_parameters():
                self.shadow[n] = self.shadow[n] * self.decay + p.detach() * (
                    1.0 - self.decay
                )


def _params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


class TeacherTrainer:
    """Stage 1: Diffusion-Forcing flow matching of the causal multi-step teacher."""

    def __init__(
        self, teacher: ScoreModel, cfg: DistillConfig, opt: OptimConfig
    ) -> None:
        self.objective = DiffusionForcingTeacher(teacher, cfg)
        self.model = teacher
        self.opt = opt
        self.optimizer = optim.AdamW(
            _params(teacher.dit),
            lr=opt.lr,
            betas=opt.betas,
            weight_decay=opt.weight_decay,
        )
        self.step = 0

    def train_step(self, batch: Batch) -> dict:
        self.optimizer.zero_grad()
        loss = self.objective.loss(batch["latents"], batch["cond"])
        loss.backward()
        norm = nn.utils.clip_grad_norm_(_params(self.model.dit), self.opt.max_grad_norm)
        self.optimizer.step()
        lucid.eval(*_params(self.model.dit))
        self.step += 1
        return {"loss": loss.item(), "grad_norm": float(norm.item())}


class DistillationTrainer:
    """Stage 2/3: on-policy student rollouts scored by the causal teacher (real) and a critic (fake)."""

    def __init__(
        self,
        generator: ScoreModel,
        real: ScoreModel,
        fake: ScoreModel,
        cfg: DistillConfig,
        opt: OptimConfig,
    ) -> None:
        self.cmd = ContextMatchedDistillation(generator, real, fake, cfg)
        self.cfg, self.opt = cfg, opt
        real.dit.requires_grad_(False)
        self.g_opt = optim.AdamW(
            _params(generator.dit),
            lr=opt.lr,
            betas=opt.betas,
            weight_decay=opt.weight_decay,
        )
        self.c_opt = optim.AdamW(
            _params(fake.dit),
            lr=opt.lr_critic,
            betas=opt.betas,
            weight_decay=opt.weight_decay,
        )
        self.ema: EMA | None = None
        self.step = 0
        self.chunk = 0
        self._batch: dict[str, Batch] = {}

    @classmethod
    def from_teacher(
        cls, teacher: ScoreModel, cfg: DistillConfig, opt: OptimConfig
    ) -> "DistillationTrainer":
        """Generator, real and fake scores all start from the causal teacher (``generator_ckpt == teacher_ckpt``)."""

        def clone() -> ScoreModel:
            m = copy.deepcopy(teacher)
            m.dit.requires_grad_(True)
            return m

        return cls(clone(), clone(), clone(), cfg, opt)

    def train_step(self, next_batch: Callable[[], Batch]) -> dict:
        chunks = self.cfg.rollout_num_chunks
        train_generator = (self.step // chunks) % self.opt.dfake_gen_update_ratio == 0
        phase = "generator" if train_generator else "critic"
        if chunks == 1 or self.chunk == 0:
            self._batch[phase] = next_batch()
        batch = self._batch[phase]
        latents = batch["latents"]
        shape = (latents.shape[0], self.cfg.scored_frames, *latents.shape[2:])
        cond = {**batch["cond"], "initial_latent": latents[:, :1]}
        uncond = {**batch["uncond"], "initial_latent": latents[:, :1]}
        chunk = self.chunk if chunks > 1 else None
        if train_generator:
            self.g_opt.zero_grad()
            loss, log = self.cmd.generator_loss(
                shape, cond, uncond, latents[:, :1], latents, chunk
            )
            loss.backward()
            model, opt, cap = self.cmd.generator.dit, self.g_opt, self.opt.max_grad_norm
        else:
            self.c_opt.zero_grad()
            loss, log = self.cmd.critic_loss(shape, cond, uncond, latents[:, :1], chunk)
            loss.backward()
            model, opt, cap = (
                self.cmd.fake.dit,
                self.c_opt,
                self.opt.max_grad_norm_critic,
            )
        norm = nn.utils.clip_grad_norm_(_params(model), cap)
        opt.step()
        lucid.eval(*_params(model))
        if train_generator and self.ema is not None:
            self.ema.update(self.cmd.generator.dit)
        self.step += 1
        if chunks > 1:
            self.chunk = (self.chunk + 1) % chunks
            if self.chunk == 0:
                self.cmd._rollout_states.clear()
        if (
            self.ema is None
            and self.opt.ema_weight > 0
            and self.step >= self.opt.ema_start_step
        ):
            self.ema = EMA(self.cmd.generator.dit, self.opt.ema_weight)
        return {
            "phase": phase,
            "loss": loss.item(),
            "grad_norm": float(norm.item()),
            **log,
        }


def batches(sample: Callable[[], Batch]) -> Iterator[Batch]:
    while True:
        yield sample()
