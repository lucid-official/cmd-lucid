"""The flow-matching table as torch computes it (``.venv-ref``).

    .venv-ref/bin/python tests/ref/schedules_torch.py

Same six float32 operations as ``FlowMatchScheduler.set_timesteps(1000)``
with ``shift=5, sigma_min=0, extra_one_step=True``.
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "schedules_torch.safetensors"

sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
sigmas = 5.0 * sigmas / (1 + 4.0 * sigmas)
timesteps = sigmas * 1000
save_file({"sigmas": sigmas, "timesteps": timesteps}, str(OUT))
print("wrote", OUT, timesteps[[0, 250, 500, 750]].tolist())
