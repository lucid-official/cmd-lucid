"""The released CMD checkpoints and the settings each one runs with.

Mirrors ``examples/run_examples.sh`` and ``configs/cosmos/*.yaml`` of the
reference: the latent frame count (``t``), the causal chunk size and the
local attention window (``l``) belong to the checkpoint, not to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CMD_REPO = "nvidia/cmd"
TEXT_ENCODER_REPO = "nvidia/Cosmos-Reason1-7B"
# Cosmos-Predict2.5 packages the Wan2.1 VAE as `tokenizer.pth` in a gated
# repository; the file is the same size (507,609,880 bytes) and uses the same
# latent statistics as the ungated Wan release, which is read instead.
VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
VAE_FILE = "Wan2.1_VAE.pth"

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = Path(
    __import__("os").environ.get("CMD_LUCID_CHECKPOINTS", ROOT / "checkpoints")
)


@dataclass(frozen=True)
class Variant:
    """One released CMD model.

    Attributes
    ----------
    key : str
        Short name used on the command line (``chunk1-short``...).
    filename : str
        File in ``nvidia/cmd``.
    latent_frames : int
        Latent frames the reference generates, the clean image latent included.
        ``4 * (t - 1) + 1`` pixel frames.
    block : int
        Latent frames denoised together (the causal chunk).
    local_attn : int
        Frames a query may attend to, its own block included.
    camera : bool
        Whether the model takes a camera trajectory.
    teacher : bool
        Multi-step causal teacher (35 UniPC steps with CFG) instead of a
        four-step student.
    """

    key: str
    filename: str
    latent_frames: int
    block: int
    local_attn: int
    camera: bool = False
    teacher: bool = False

    @property
    def pixel_frames(self) -> int:
        return 4 * (self.latent_frames - 1) + 1

    @property
    def path(self) -> Path:
        return CHECKPOINTS / "cmd" / self.filename


VARIANTS: dict[str, Variant] = {
    v.key: v
    for v in [
        Variant("chunk1-short", "chunk1_short_t24_l21.safetensors", 24, 1, 21),
        Variant("chunk4-short", "chunk4_short_t21_l16.safetensors", 21, 4, 16),
        Variant("chunk1-long", "chunk1_long_t126_l21.safetensors", 126, 1, 21),
        Variant("chunk4-long", "chunk4_long_t121_l16.safetensors", 121, 4, 16),
        Variant(
            "chunk1-camera",
            "chunk1_camera_control_t32_l21.safetensors",
            32,
            1,
            21,
            camera=True,
        ),
        Variant(
            "chunk4-camera",
            "chunk4_camera_control_t29_l24.safetensors",
            29,
            4,
            24,
            camera=True,
        ),
        Variant(
            "chunk1-teacher",
            "chunk1_teacher_t24_l21.safetensors",
            24,
            1,
            21,
            teacher=True,
        ),
        Variant(
            "chunk4-teacher",
            "chunk4_teacher_t21_l16.safetensors",
            21,
            4,
            16,
            teacher=True,
        ),
        Variant(
            "chunk1-camera-teacher",
            "chunk1_teacher_t32_l21_camera.safetensors",
            32,
            1,
            21,
            camera=True,
            teacher=True,
        ),
        Variant(
            "chunk4-camera-teacher",
            "chunk4_teacher_t29_l24_camera.safetensors",
            29,
            4,
            24,
            camera=True,
            teacher=True,
        ),
    ]
}


def text_encoder_dir() -> Path:
    return CHECKPOINTS / "cosmos-reason1-7b"


def vae_path() -> Path:
    return CHECKPOINTS / "wan2.1_vae.safetensors"
