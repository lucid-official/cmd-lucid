"""Schedules: the flow-matching table bit for bit, and the UniPC solver's behaviour."""

import lucid

from cmd_lucid.scheduler import FlowMatchSchedule, FlowUniPC
from conftest import load_fixture


def test_flow_match_table_is_bit_exact():
    fx = load_fixture("schedules_torch.safetensors")
    sched = FlowMatchSchedule()
    assert sched.timesteps.tolist() == fx["timesteps"].tolist()
    assert sched.sigmas.tolist() == fx["sigmas"].tolist()


def test_warped_student_steps():
    sched = FlowMatchSchedule()
    warped = sched.warp([1000, 750, 500, 250])
    assert warped[0] == 1000.0 and warped[1] == 937.5 and warped[3] == 625.0
    assert abs(warped[2] - 833.3333) < 1e-3
    # Context noise 128 lands on the nearest table entry, sigma ~ 0.1299.
    assert abs(sched.sigma(128) - 0.12993) < 1e-4


def test_unipc_recovers_a_straight_flow():
    """For x_t = (1 - s) x0 + s eps the flow is constant; every x0 estimate is exact."""
    lucid.manual_seed(0)
    x0, eps = lucid.randn(2, 8), lucid.randn(2, 8)
    solver = FlowUniPC(35)
    assert solver.timesteps[0] == 999 and len(solver.timesteps) == 35
    s0 = solver.sigmas[0]
    x = (1 - s0) * x0 + s0 * eps
    for _ in solver.timesteps:
        x = solver.step(eps - x0, x)
    assert (x - x0).abs().max().item() < 1e-5
