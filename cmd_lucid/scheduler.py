"""Noise schedules of the CMD pipelines.

:class:`FlowMatchSchedule` is ``utils.scheduler.FlowMatchScheduler(shift=5,
sigma_min=0, extra_one_step=True)`` after ``set_timesteps(1000)``: the
1000-entry table the few-step student, the KV-cache context noise and the
DMD/critic losses all look timesteps up in. The table is computed with the
same float32 operations as the reference (``torch.linspace`` then the shift),
on the CPU, so lookups land on the same entries.

:class:`FlowUniPC` is the multi-step solver the causal teachers sample with
(``wan.utils.fm_solvers_unipc.FlowUniPCMultistepScheduler``: UniPC bh2,
order 2, x0 prediction from the flow, lower order on the last step).
"""

from __future__ import annotations

import math
import struct

import lucid

Tensor = lucid.Tensor


def _f32(v: float) -> float:
    """Round a Python float to the nearest float32 (correct rounding for one +, -, * or /)."""
    return struct.unpack("f", struct.pack("f", v))[0]


def _linspace_f32(start: float, end: float, steps: int) -> list[float]:
    """``torch.linspace`` on the CPU in float32, bit for bit.

    The first half counts up from ``start`` and the second down from ``end``;
    each element is one fused multiply-add (``start + step * i`` rounded
    once), which the product being exact in double precision reproduces.
    """
    step = _f32((_f32(end) - _f32(start)) / (steps - 1))
    half = steps // 2
    return [
        _f32(start + step * i) if i < half else _f32(end - step * (steps - 1 - i))
        for i in range(steps)
    ]


class FlowMatchSchedule:
    """The shifted rectified-flow table: ``sigma_i`` and ``timestep_i = 1000 sigma_i``."""

    def __init__(self, shift: float = 5.0, num_train_timesteps: int = 1000) -> None:
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        f, sh = _f32, _f32(shift)
        lin = _linspace_f32(1.0, 0.0, num_train_timesteps + 1)[:-1]
        # shift * s / (1 + (shift - 1) * s), one float32 rounding per operation.
        self._s = [f(f(sh * s) / f(1.0 + f(f(sh - 1.0) * s))) for s in lin]
        self._t = [f(s * num_train_timesteps) for s in self._s]
        self.sigmas = lucid.tensor(self._s, dtype=lucid.float32)
        self.timesteps = lucid.tensor(self._t, dtype=lucid.float32)

    def warp(self, steps: list[int]) -> list[float]:
        """``warp_denoising_step``: ``[1000, 750, 500, 250]`` -> ``[1000, 937.5, 833.3, 625]``."""
        table = self._t + [0.0]
        return [table[self.num_train_timesteps - s] for s in steps]

    def index(self, timestep: float) -> int:
        """``argmin |timesteps - t|`` (first index on ties, as ``torch.argmin``)."""
        best, best_d = 0, math.inf
        for i, v in enumerate(self._t):
            d = abs(v - timestep)
            if d < best_d:
                best, best_d = i, d
        return best

    def sigma(self, timestep: float) -> float:
        return self._s[self.index(timestep)]

    def add_noise(self, x: Tensor, noise: Tensor, timestep: float) -> Tensor:
        """``(1 - sigma) x + sigma noise`` in the dtype of ``noise`` (``type_as(noise)``)."""
        s = self.sigma(timestep)
        return ((1.0 - s) * x.float() + s * noise.float()).to(noise.dtype)

    def x0_from_flow(self, flow: Tensor, xt: Tensor, timestep: float) -> Tensor:
        """``_convert_flow_pred_to_x0``: ``xt - sigma_t * flow`` in fp32."""
        return xt.float() - self.sigma(timestep) * flow.float()

    def flow_from_x0(self, x0: Tensor, xt: Tensor, timestep: float) -> Tensor:
        return (xt.float() - x0.float()) / self.sigma(timestep)


class FlowUniPC:
    """``FlowUniPCMultistepScheduler`` (solver_order=2, bh2, predict_x0, final sigma 0)."""

    def __init__(
        self,
        num_inference_steps: int,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ) -> None:
        # Constructed with shift=1: sigma_max = 1 - 1/1000 (float32), sigma_min = 0.
        sigma_max = _f32(1.0 - 1.0 / num_train_timesteps)
        # np.linspace(sigma_max, 0, n + 1)[:-1]: i * step + start.
        step = (0.0 - sigma_max) / num_inference_steps
        sigmas = [i * step + sigma_max for i in range(num_inference_steps)]
        sigmas = [shift * s / (1 + (shift - 1) * s) for s in sigmas]
        # np.float64 -> int64 truncates toward zero.
        self.timesteps = [int(s * num_train_timesteps) for s in sigmas]
        self.sigmas = lucid.tensor(sigmas + [0.0], dtype=lucid.float32).tolist()
        self.order = 2
        self.model_outputs: list[Tensor | None] = [None, None]
        self.lower_order_nums = 0
        self.last_sample: Tensor | None = None
        self.step_index = 0
        self.this_order = 1

    def _lambda(self, sigma: float) -> float:
        alpha = 1.0 - sigma
        return _f32(math.log(_f32(alpha)) - math.log(sigma)) if sigma > 0 else math.inf

    def _coeffs(
        self, sigma_t: float, sigma_s0: float, rk_sigmas: list[float], order: int
    ):
        lam_t, lam_s0 = self._lambda(sigma_t), self._lambda(sigma_s0)
        h = lam_t - lam_s0
        rks = [(self._lambda(s) - lam_s0) / h for s in rk_sigmas] + [1.0]
        hh = -h
        h_phi_1 = math.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1 if math.isfinite(hh) else -1.0
        b_h = math.expm1(hh)
        factorial = 1
        b = []
        for i in range(1, order + 1):
            b.append(h_phi_k * factorial / b_h)
            factorial *= i + 1
            h_phi_k = (
                (h_phi_k / hh - 1 / factorial) if math.isfinite(hh) else -1 / factorial
            )
        return h_phi_1, b_h, rks, b

    def step(self, flow: Tensor, sample: Tensor) -> Tensor:
        """One solver step at ``timesteps[step_index]``; returns the next sample (dtype of ``sample``)."""
        i = self.step_index
        x0 = sample.float() - self.sigmas[i] * flow.float()
        if i > 0 and self.last_sample is not None:
            sample = self._corrector(x0, self.last_sample, sample, self.this_order)
        self.model_outputs = [self.model_outputs[1], x0]
        this_order = min(self.order, len(self.timesteps) - i)
        self.this_order = min(this_order, self.lower_order_nums + 1)
        self.last_sample = sample
        out = self._predictor(sample, self.this_order)
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1
        self.step_index += 1
        return out

    def _predictor(self, x: Tensor, order: int) -> Tensor:
        i = self.step_index
        m0 = self.model_outputs[-1]
        sigma_t, sigma_s0 = self.sigmas[i + 1], self.sigmas[i]
        h_phi_1, b_h, rks, _ = self._coeffs(
            sigma_t, sigma_s0, [self.sigmas[i - k] for k in range(1, order)], order
        )
        alpha_t = 1.0 - sigma_t
        x_t = (sigma_t / sigma_s0) * x.float() - (alpha_t * h_phi_1) * m0
        if order == 2:
            d1 = (self.model_outputs[-2] - m0) / rks[0]
            x_t = x_t - (alpha_t * b_h * 0.5) * d1
        return x_t.to(x.dtype)

    def _corrector(
        self, model_t: Tensor, last: Tensor, this: Tensor, order: int
    ) -> Tensor:
        i = self.step_index
        m0 = self.model_outputs[-1]
        sigma_t, sigma_s0 = self.sigmas[i], self.sigmas[i - 1]
        h_phi_1, b_h, rks, b = self._coeffs(
            sigma_t, sigma_s0, [self.sigmas[i - 1 - k] for k in range(1, order)], order
        )
        alpha_t = 1.0 - sigma_t
        x_t = (sigma_t / sigma_s0) * last.float() - (alpha_t * h_phi_1) * m0
        if order == 1:
            rhos = [0.5]
        else:
            # Solve R rho = b with R = [[1, 1], [r, 1]].
            r = rks[0]
            det = 1.0 - r
            rhos = [(b[0] - b[1]) / det, (b[1] - r * b[0]) / det]
        corr = 0.0
        if order == 2:
            d1 = (self.model_outputs[-2] - m0) / rks[0]
            corr = rhos[0] * d1
        x_t = x_t - (alpha_t * b_h) * (corr + rhos[-1] * (model_t - m0))
        return x_t.to(this.dtype)
