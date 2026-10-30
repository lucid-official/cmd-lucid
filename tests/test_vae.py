"""The Lucid Wan2.1 VAE against diffusers' AutoencoderKLWan, on the released weights."""

import lucid
import pytest

from cmd_lucid import catalog
from cmd_lucid.models.wan_vae import WanVAE
from conftest import DEVICES, load_fixture, max_rel


def _vae(device):
    if not catalog.vae_path().exists():
        pytest.skip("VAE weights not downloaded")
    vae = WanVAE()
    vae.load_state_dict(lucid.load_safetensors(str(catalog.vae_path())), strict=True)
    return vae.to(device).eval()


@pytest.mark.parametrize("device", DEVICES)
def test_encode_and_decode_match_diffusers(device):
    fx = load_fixture("vae_diffusers.safetensors")
    vae = _vae(device)
    with lucid.no_grad():
        latent = vae.encode(fx["in.video"].to(device))
        pixels = vae.decode(fx["in.z"].to(device))
    assert latent.shape == fx["out.latent"].shape
    assert max_rel(latent, fx["out.latent"]) < 1e-4
    assert pixels.shape == fx["out.pixels"].shape
    assert max_rel(pixels, fx["out.pixels"]) < 1e-4


def test_streaming_decode_equals_full_decode():
    vae = _vae("metal")
    lucid.manual_seed(0)
    z = lucid.randn(1, 16, 4, 8, 12).to("metal")
    with lucid.no_grad():
        full = vae.decode(z)
        state = vae.new_state()
        parts = [vae.decode_stream(z[:, :, i : i + 1], state) for i in range(4)]
    assert (lucid.cat(parts, dim=2) - full).abs().max().item() == 0.0
