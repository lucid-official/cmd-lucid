"""Reference hidden states from transformers' Qwen2.5-VL text model (``.venv-ref``).

    .venv-ref/bin/python tests/ref/qwen_hf.py

The first 4 decoder layers of Cosmos-Reason1-7B (real weights, shard 1) in
fp32 on the CPU, with the final norm's weight set to ones. Records
``hidden_states[1:]`` for the example prompt's 512 token ids, plus the
Predict2.5 feature recipe applied to them.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "checkpoints" / "cosmos-reason1-7b"
LAYERS = 4


def main() -> None:
    cfg = AutoConfig.from_pretrained(str(DIR)).text_config
    cfg.num_hidden_layers = LAYERS
    cfg._attn_implementation = "sdpa"
    model = Qwen2_5_VLTextModel(cfg).float().eval()
    shard = load_file(str(DIR / "model-00001-of-00004.safetensors"))
    keep = {"model.embed_tokens.weight"} | {k for k in shard if any(k.startswith(f"model.layers.{i}.") for i in range(LAYERS))}
    sd = {k.removeprefix("model."): shard[k].float() for k in keep}
    sd["norm.weight"] = torch.ones(cfg.hidden_size)
    model.load_state_dict(sd, strict=True)
    ids = json.loads((ROOT / "tests/fixtures/tokenizer_hf.json").read_text())[0]["ids"]
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
        states = out.hidden_states[1:]
        norm = [(s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-8) for s in states]
    tensors = {f"state.{i}": s[0].contiguous() for i, s in enumerate(states)}
    tensors["features"] = torch.cat(norm, dim=-1)[0].contiguous()
    tensors["ids"] = input_ids[0].to(torch.int64)
    path = ROOT / "tests/fixtures/qwen_hf_4layers.safetensors"
    save_file(tensors, str(path))
    print("wrote", path, len(states), tuple(tensors["features"].shape))


if __name__ == "__main__":
    main()
