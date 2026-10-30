"""Reference outputs for the Wan2.1 VAE, from diffusers' ``AutoencoderKLWan``.

Runs in ``.venv-ref``:  .venv-ref/bin/python tests/ref/vae_diffusers.py

Loads ``checkpoints/wan2.1_vae.safetensors`` (the released weights, original
naming), converts them with diffusers' own single-file converter, and records
an encode and a streaming-length decode in fp32.
"""

from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "vae_diffusers.safetensors"
MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def main() -> None:
    torch.manual_seed(0)
    sd = load_file(str(ROOT / "checkpoints" / "wan2.1_vae.safetensors"))
    vae = AutoencoderKLWan()
    missing, unexpected = vae.load_state_dict(convert_wan_vae_to_diffusers(dict(sd)), strict=True), None
    vae = vae.eval().float()
    mean = torch.tensor(MEAN).view(1, 16, 1, 1, 1)
    std = torch.tensor(STD).view(1, 16, 1, 1, 1)
    # A smooth-ish video in [-1, 1]: 9 frames -> 3 latent frames.
    video = torch.tanh(torch.randn(1, 3, 9, 64, 96).cumsum(-1) * 0.2)
    z = torch.randn(1, 16, 4, 8, 12)
    with torch.no_grad():
        mu = vae.encode(video).latent_dist.mean
        decoded = vae.decode(z * std + mean).sample
    save_file(
        {"in.video": video, "out.latent": ((mu - mean) / std).contiguous(), "in.z": z, "out.pixels": decoded.contiguous()},
        str(OUT),
    )
    print("wrote", OUT, tuple(mu.shape), tuple(decoded.shape))


if __name__ == "__main__":
    main()
