"""The Lucid DiT against diffusers' Cosmos-Predict2.5 transformer, and its own causal paths."""

import lucid
import pytest

from cmd_lucid.models.cosmos_dit import (
    CosmosDiT, DiTConfig, FrameKVCache, MaskedAttention, StreamingAttention,
)
from conftest import DEVICES, load_fixture, max_rel

TINY = DiTConfig(
    model_channels=128, num_blocks=2, num_heads=2, adaln_lora_dim=16,
    crossattn_proj_in_channels=96, crossattn_emb_channels=32,
)


def _model(fx, device, camera_dim=None):
    cfg = DiTConfig(**{**TINY.__dict__, "camera_dim": camera_dim})
    model = CosmosDiT(cfg)
    weights = {k[2:]: v for k, v in fx.items() if k.startswith("w.")}
    if camera_dim:
        lucid.manual_seed(3)
        for i in range(cfg.num_blocks):
            weights[f"blocks.{i}.self_attn.cam_encoder.weight"] = lucid.randn(cfg.model_channels, camera_dim) * 0.02
    missing = model.load_state_dict(weights, strict=False)
    assert all(k.startswith("pos_embedder.") for k in missing.missing_keys), missing
    assert not missing.unexpected_keys
    return model.to(device).eval()


@pytest.mark.parametrize("device", DEVICES)
def test_bidirectional_matches_diffusers(device):
    fx = load_fixture("dit_diffusers.safetensors")
    model = _model(fx, device)
    x, cond, t = fx["in.x"].to(device), fx["in.cond"].to(device), fx["in.t"].to(device)
    ctx = model.project_context(fx["in.prompt"].to(device))
    frames = x.shape[2]
    with lucid.no_grad():
        out = model.forward_full(x, t, ctx, cond, MaskedAttention(frames_per_block=frames))
    assert max_rel(out, fx["out.flow"]) < 2e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block,local", [(1, 3), (2, 4), (1, -1)])
def test_streaming_equals_block_causal_full_sequence(device, block, local):
    """KV-cached chunks against one block-causal pass over the same frames.

    With one frame per chunk the store pass sees exactly the frames a
    block-causal query sees, so streaming must equal the full masked forward.
    """
    fx = load_fixture("dit_diffusers.safetensors")
    model = _model(fx, device, camera_dim=1536)
    lucid.manual_seed(0)
    frames = 1 + 3 * block
    x = lucid.randn(1, 16, frames, 8, 12).to(device)
    cam = lucid.randn(1, 1536, frames, 4, 6).to(device)
    cond = lucid.zeros(1, 1, frames, 8, 12).to(device)
    t = lucid.full((1, frames), 400.0).to(device)
    ctx = model.project_context(lucid.randn(1, 5, 96).to(device))
    with lucid.no_grad():
        full = model.forward_full(x, t, ctx, cond, MaskedAttention(block, True, local), camera=cam)
        cache = FrameKVCache(TINY.num_blocks, local_attn=local)
        start, chunks = 0, [1] + [block] * 3
        outs = []
        for n in chunks:
            sl = slice(start, start + n)
            args = (x[:, :, sl], t[:, sl], ctx, cond[:, :, sl])
            # Denoising pass (reads history), then the store pass for that chunk.
            outs.append(model.forward_stream(*args, StreamingAttention(cache, start, False, block), camera=cam[:, :, sl]))
            model.forward_stream(*args, StreamingAttention(cache, start, True, block), camera=cam[:, :, sl])
            start += n
        stream = lucid.cat(outs, dim=2)
    if block == 1:
        assert max_rel(stream, full) < 2e-5
    else:
        # Reference quirk (CausalCosmosAttention.forward): a multi-frame store
        # pass attends within its chunk only, so K/V cached for the first
        # multi-frame chunk never saw the image frame. Chunks up to and
        # including that one agree; later ones read the history-blind cache.
        upto = 1 + block
        assert max_rel(stream[:, :, :upto], full[:, :, :upto]) < 2e-5
        assert max_rel(stream[:, :, upto:], full[:, :, upto:]) > 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_packed_scoring_equals_masked_full_sequence(device):
    """forward_packed (teacher forcing) equals a block-causal pass over [history | targets]."""
    fx = load_fixture("dit_diffusers.safetensors")
    model = _model(fx, device)
    lucid.manual_seed(1)
    n_ctx, n_tgt = 3, 4
    clean = lucid.randn(1, 16, n_ctx, 8, 12).to(device)
    noisy = lucid.randn(1, 16, n_ctx + n_tgt, 8, 12).to(device)
    t_clean = lucid.tensor([[0.0, 256.0, 256.0]]).to(device)
    t_noisy = lucid.full((1, n_ctx + n_tgt), 700.0).to(device)
    cmask = lucid.zeros(1, 1, n_ctx, 8, 12).to(device)
    nmask = lucid.zeros(1, 1, n_ctx + n_tgt, 8, 12).to(device)
    ctx = model.project_context(lucid.randn(1, 5, 96).to(device))
    att = MaskedAttention(1, True, 5)
    with lucid.no_grad():
        packed = model.forward_packed(noisy, clean, t_noisy, t_clean, ctx, nmask, cmask, att, n_ctx)
        seq = lucid.cat([clean, noisy[:, :, n_ctx:]], dim=2)
        tt = lucid.cat([t_clean, t_noisy[:, n_ctx:]], dim=1)
        full = model.forward_full(seq, tt, ctx, nmask, att)
    # forward_full puts the timestep MLP in fp32; in fp32 runs the two coincide.
    assert max_rel(packed[:, :, n_ctx:], full[:, :, n_ctx:]) < 2e-5
    assert packed[:, :, :n_ctx].abs().max().item() == 0.0
