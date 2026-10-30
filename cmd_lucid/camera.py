"""Camera-ray conditioning (``cosmos/camera_conditioning.py``) and an interactive controller.

A camera trajectory is given per *pixel* frame (camera-to-world ``E_t`` and
intrinsics ``K_t``, pixel units at 480x832). Every fourth camera lines up with
a latent frame. Each latent frame is conditioned on its pose relative to the
start of its causal block — the previous frame for chunk-1 models — so the
signal is ``Delta E_t = E_anchor^-1 E_t``. For every pixel a unit ray is cast
through ``K_t``, rotated by ``Delta E_t``, and stored with the camera origin
(6 channels); 16x16 pixel patches become the 1536-channel tokens that line up
with the DiT's 30x52 token grid.

Cameras follow the OpenCV convention: +x right, +y down, +z forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import lucid
import lucid.nn.functional as F

from cmd_lucid.io.npy import load_npz

Tensor = lucid.Tensor
CAMERA_FEATURE_DIM = 6


def camera_frame_indices(num_pixel_frames: int, frame_stride: int = 4) -> list[int]:
    if (num_pixel_frames - 1) % frame_stride:
        raise ValueError(f"camera sequence length must be 1 + k * {frame_stride}, got {num_pixel_frames}")
    return list(range(0, num_pixel_frames, frame_stride))


def frame_relative(camera_to_world: Tensor, frames_per_block: int = 1) -> Tensor:
    """Each pose relative to its block's anchor (frame ``floor((t-1)/C) * C``)."""
    t = camera_to_world.shape[1]
    anchors = [max(i - 1, 0) // frames_per_block * frames_per_block for i in range(t)]
    anchor_poses = camera_to_world[:, anchors]
    return lucid.linalg.solve(anchor_poses, camera_to_world)


def camera_rays(camera_to_world: Tensor, intrinsics: Tensor, height: int, width: int) -> Tensor:
    """Ray origins and unit directions, ``(B, T, H, W, 6)``, float32."""
    poses = camera_to_world.float()
    k = intrinsics.float()
    b, t = poses.shape[:2]
    dev = poses.device.type
    ys = lucid.arange(height, dtype=lucid.float32, device=dev) + 0.5
    xs = lucid.arange(width, dtype=lucid.float32, device=dev) + 0.5
    py = ys.reshape(1, 1, height, 1)
    px = xs.reshape(1, 1, 1, width)
    fx, fy = k[..., 0, 0].reshape(b, t, 1, 1), k[..., 1, 1].reshape(b, t, 1, 1)
    cx, cy = k[..., 0, 2].reshape(b, t, 1, 1), k[..., 1, 2].reshape(b, t, 1, 1)
    dx = ((px - cx) / fx).expand(b, t, height, width)
    dy = ((py - cy) / fy).expand(b, t, height, width)
    direction = F.normalize(lucid.stack([dx, dy, lucid.ones_like(dx)], dim=-1), dim=-1)
    rotation = poses[..., :3, :3]
    world = (direction.reshape(b, t, height * width, 3) @ rotation.swapaxes(-1, -2)).reshape(b, t, height, width, 3)
    origin = poses[..., :3, 3].reshape(b, t, 1, 1, 3).expand(b, t, height, width, 3)
    return lucid.cat([origin, world], dim=-1)


def patchify_rays(rays: Tensor, patch: int = 16) -> Tensor:
    """``(B, T, H, W, 6)`` -> ``(B, 6 * patch^2, T, H/patch, W/patch)``, channels ordered ``(c, py, px)``."""
    b, t, h, w, c = rays.shape
    x = rays.permute(0, 4, 1, 2, 3).reshape(b, c, t, h // patch, patch, w // patch, patch)
    return x.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * patch * patch, t, h // patch, w // patch)


def build_camera_conditioning(
    camera_to_world: Tensor,
    intrinsics: Tensor,
    *,
    height: int = 480,
    width: int = 832,
    frame_stride: int = 4,
    patch: int = 16,
    frames_per_block: int = 1,
    expected_latent_frames: int | None = None,
) -> Tensor:
    """Pixel-frame cameras ``(B, T, 4, 4)`` / ``(B, T, 3, 3)`` -> ray tokens ``(B, 1536, T_latent, 30, 52)``."""
    idx = camera_frame_indices(camera_to_world.shape[1], frame_stride)
    if expected_latent_frames is not None and len(idx) != expected_latent_frames:
        raise ValueError(f"cameras give {len(idx)} latent frames, expected {expected_latent_frames}")
    poses = camera_to_world[:, idx]
    k = intrinsics if intrinsics.ndim == 4 else intrinsics.reshape(intrinsics.shape[0], 1, 3, 3)
    k = k[:, idx] if k.shape[1] == camera_to_world.shape[1] else k.expand(k.shape[0], len(idx), 3, 3)
    rays = camera_rays(frame_relative(poses, frames_per_block), k, height, width)
    return patchify_rays(rays, patch)


def load_trajectory(path, latent_frames: int, frame_stride: int = 4) -> tuple[Tensor, Tensor]:
    """``inference.load_camera_inputs``: world-to-camera -> camera-to-world, cut to the video."""
    arrays = load_npz(path)
    pixel_frames = 1 + (latent_frames - 1) * frame_stride
    w2c = arrays["target_w2c"].float()
    k = arrays["target_intrinsics"].float()
    if w2c.shape[0] < pixel_frames:
        raise ValueError(f"camera file has {w2c.shape[0]} frames, need {pixel_frames}")
    c2w = lucid.linalg.inv(w2c[:pixel_frames])
    return c2w.reshape(1, pixel_frames, 4, 4), k[:pixel_frames].reshape(1, pixel_frames, 3, 3)


# --------------------------------------------------------------------------- #
# interactive control
# --------------------------------------------------------------------------- #


def _matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _rotation(yaw: float, pitch: float, roll: float = 0.0) -> list[list[float]]:
    """Camera-frame rotation: yaw about +y (down), pitch about +x (right), roll about +z."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]]
    rx = [[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]]
    rz = [[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return _matmul4(_matmul4(ry, rx), rz)


@dataclass
class CameraAction:
    """Motion over one latent frame (four video frames), in the current camera's frame.

    ``forward``/``right``/``up`` are in scene units (the example trajectory
    moves ~0.16 per latent frame); ``yaw``/``pitch`` in degrees (positive yaw
    turns right, positive pitch looks up).
    """

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    @classmethod
    def from_keys(cls, keys: str, speed: float = 0.16, turn: float = 3.0) -> "CameraAction":
        """``w a s d`` move, ``q e`` rise/sink, arrows (``i j k l``) look."""
        k = set(keys.lower())
        return cls(
            forward=speed * (("w" in k) - ("s" in k)),
            right=speed * (("d" in k) - ("a" in k)),
            up=speed * (("q" in k) - ("e" in k)),
            yaw=turn * (("l" in k) - ("j" in k)),
            pitch=turn * (("i" in k) - ("k" in k)),
        )


@dataclass
class CameraController:
    """Accumulates per-latent-frame actions into camera-to-world poses.

    ``poses[0]`` is the identity camera of the input image. Each
    :meth:`advance` appends the pose of the next latent frame; the ray
    tokens of any frame are then exactly those :func:`build_camera_conditioning`
    builds for the same trajectory.
    """

    intrinsics: list[list[float]] = field(
        default_factory=lambda: [[529.088, 0.0, 416.032], [0.0, 529.187, 239.365], [0.0, 0.0, 1.0]]
    )
    poses: list[list[list[float]]] = field(
        default_factory=lambda: [[[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]]
    )

    def advance(self, action: CameraAction) -> None:
        delta = _rotation(math.radians(action.yaw), math.radians(action.pitch))
        delta[0][3], delta[1][3], delta[2][3] = action.right, -action.up, action.forward
        self.poses.append(_matmul4(self.poses[-1], delta))

    def tokens(self, start: int, count: int, frames_per_block: int = 1, height: int = 480, width: int = 832) -> Tensor:
        """Ray tokens of latent frames ``start .. start+count-1``, ``(1, 1536, count, 30, 52)``."""
        frames = list(range(start, start + count))
        anchors = [max(i - 1, 0) // frames_per_block * frames_per_block for i in frames]
        c2w = lucid.tensor([self.poses[i] for i in frames], dtype=lucid.float32).reshape(1, count, 4, 4)
        anc = lucid.tensor([self.poses[i] for i in anchors], dtype=lucid.float32).reshape(1, count, 4, 4)
        k = lucid.tensor([self.intrinsics] * count, dtype=lucid.float32).reshape(1, count, 3, 3)
        rel = lucid.linalg.solve(anc, c2w)
        return patchify_rays(camera_rays(rel, k, height, width))
