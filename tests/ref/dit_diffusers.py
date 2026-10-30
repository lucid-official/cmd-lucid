"""Reference outputs for the Cosmos-Predict2.5 DiT, from diffusers.

Runs in the torch reference venv (``.venv-ref``), never in the Lucid one:

    .venv-ref/bin/python tests/ref/dit_diffusers.py

Builds diffusers' ``CosmosTransformer3DModel`` (which implements
Cosmos-Predict2.5 with ``use_crossattn_projection``) at a tiny size with
random weights, runs a bidirectional forward, and writes the weights (in the
Cosmos checkpoint naming CMD uses), inputs and output to
``tests/fixtures/dit_diffusers.safetensors``.
"""

from pathlib import Path

import torch
from diffusers import CosmosTransformer3DModel
from safetensors.torch import save_file

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "dit_diffusers.safetensors"

# Cosmos checkpoint name -> diffusers name (``convert_cosmos_transformer_checkpoint_to_diffusers``).
RENAME = {
    "t_embedder.1": "time_embed.t_embedder",
    "t_embedding_norm": "time_embed.norm",
    "blocks": "transformer_blocks",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn": "attn1",
    "cross_attn": "attn2",
    "q_proj": "to_q",
    "k_proj": "to_k",
    "v_proj": "to_v",
    "output_proj": "to_out.0",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}


def to_diffusers(key: str) -> str:
    for a, b in RENAME.items():
        key = key.replace(a, b)
    return key


def main() -> None:
    torch.manual_seed(0)
    heads, head_dim, layers = 2, 64, 2
    model = CosmosTransformer3DModel(
        in_channels=17,  # latent + condition mask
        out_channels=16,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        num_layers=layers,
        mlp_ratio=4.0,
        text_embed_dim=32,
        adaln_lora_dim=16,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 3.0, 3.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=96,
        encoder_hidden_states_channels=32,
    ).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    diff_sd = model.state_dict()

    # Every Cosmos-named key of the Lucid model, mapped to diffusers.
    names = [
        "x_embedder.proj.1.weight",
        "t_embedder.1.linear_1.weight",
        "t_embedder.1.linear_2.weight",
        "t_embedding_norm.weight",
        "final_layer.linear.weight",
        "final_layer.adaln_modulation.1.weight",
        "final_layer.adaln_modulation.2.weight",
        "crossattn_proj.0.weight",
        "crossattn_proj.0.bias",
    ]
    for i in range(layers):
        for part in ["self_attn", "cross_attn"]:
            for w in ["q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm"]:
                names.append(f"blocks.{i}.{part}.{w}.weight")
        for m in ["self_attn", "cross_attn", "mlp"]:
            names += [
                f"blocks.{i}.adaln_modulation_{m}.1.weight",
                f"blocks.{i}.adaln_modulation_{m}.2.weight",
            ]
        names += [f"blocks.{i}.mlp.layer1.weight", f"blocks.{i}.mlp.layer2.weight"]
    tensors = {}
    for name in names:
        tensors["w." + name] = diff_sd[to_diffusers(name)].contiguous()
    assert len(names) == len(diff_sd), (
        len(names),
        len(diff_sd),
        set(diff_sd) - {to_diffusers(n) for n in names},
    )

    b, t, h, w = 1, 3, 8, 12
    x = torch.randn(b, 16, t, h, w)
    cond = torch.zeros(b, 1, t, h, w)
    cond[:, :, 0] = 1
    raw_t = torch.tensor([[0.0, 625.0, 937.5]])
    prompt = torch.randn(b, 7, 96)
    with torch.no_grad():
        out = model(
            hidden_states=x,
            timestep=(raw_t * 0.001).reshape(b, 1, t, 1, 1),
            encoder_hidden_states=prompt,
            condition_mask=cond,
            padding_mask=torch.zeros(b, 1, h, w),
            return_dict=False,
        )[0]
    tensors.update(
        {
            "in.x": x,
            "in.cond": cond,
            "in.t": raw_t,
            "in.prompt": prompt,
            "out.flow": out,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(OUT))
    print("wrote", OUT, tuple(out.shape))


if __name__ == "__main__":
    main()
