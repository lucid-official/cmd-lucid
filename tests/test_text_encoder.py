"""The Lucid Qwen2.5-VL text path against transformers, on the released weights (first 4 layers, fp32)."""

import lucid
import pytest

from cmd_lucid import catalog
from cmd_lucid.text.qwen import CosmosReasonEncoder, QwenConfig
from conftest import DEVICES, load_fixture, max_rel

LAYERS = 4


@pytest.mark.parametrize("device", DEVICES)
def test_hidden_states_and_features_match_transformers(device):
    fx = load_fixture("qwen_hf_4layers.safetensors")
    shard = catalog.text_encoder_dir() / "model-00001-of-00004.safetensors"
    if not shard.exists():
        pytest.skip("text encoder shard 1 not downloaded")
    model = CosmosReasonEncoder(QwenConfig(num_layers=LAYERS))
    weights = lucid.load_safetensors(str(shard))
    keep = {k: v.float() for k, v in weights.items() if k in model.state_dict()}
    keep["model.norm.weight"] = lucid.ones(QwenConfig().hidden_size)
    model.load_state_dict(keep, strict=True)
    model = model.to(device).eval()
    ids = fx["ids"].reshape(1, -1).to(device)
    with lucid.no_grad():
        states = model.hidden_states(ids)
        features = model(ids)
    for i, s in enumerate(states):
        assert max_rel(s[0], fx[f"state.{i}"]) < 1e-4, i
    assert max_rel(features[0], fx["features"]) < 1e-4
