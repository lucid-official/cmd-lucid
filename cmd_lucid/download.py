"""Fetch the released weights into ``checkpoints/``.

    python -m cmd_lucid.download                   # everything, most useful first
    python -m cmd_lucid.download chunk1-camera     # one CMD model (+ shared parts)

Each CMD DiT is stored in bf16, the dtype the reference casts it to before
inference. The text encoder keeps only its language model (the vision tower
and LM head are never used to embed a prompt). The VAE is converted from its
``torch.save`` archive without torch.
"""

from __future__ import annotations

import argparse
import shutil

from cmd_lucid import catalog, hub

_TEXT_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json"]
_TEXT_SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
_ORDER = [
    "chunk1-camera", "chunk1-short", "chunk4-short", "chunk4-camera",
    "chunk1-long", "chunk4-long", "chunk1-teacher", "chunk4-teacher",
    "chunk1-camera-teacher", "chunk4-camera-teacher",
]
_MIN_FREE_GB = 12


def fetch_shared() -> None:
    raw = catalog.CHECKPOINTS / "_raw" / catalog.VAE_FILE
    if not catalog.vae_path().exists():
        hub.fetch_file(catalog.VAE_REPO, catalog.VAE_FILE, raw)
        from cmd_lucid.io.torch_zip import convert_wan_vae

        convert_wan_vae(raw, catalog.vae_path())
        raw.unlink()
    target = catalog.text_encoder_dir()
    for name in _TEXT_FILES:
        hub.fetch_file(catalog.TEXT_ENCODER_REPO, name, target / name)
    for shard in _TEXT_SHARDS:
        remote = hub.RemoteSafetensors(catalog.TEXT_ENCODER_REPO, shard)
        hub.fetch_passthrough(remote, target / shard, select=lambda n: n.startswith("model."))


def fetch_variant(key: str) -> None:
    variant = catalog.VARIANTS[key]
    if variant.path.exists():
        return
    free = shutil.disk_usage(catalog.CHECKPOINTS).free / 1e9
    if free < _MIN_FREE_GB:
        raise SystemExit(f"only {free:.0f} GB free; not fetching {key}")
    remote = hub.RemoteSafetensors(catalog.CMD_REPO, variant.filename)
    if all(t.dtype == "BF16" for t in remote.tensors):
        hub.fetch_passthrough(remote, variant.path)
    else:
        hub.fetch_cast(remote, variant.path, "BF16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help=f"default: all, in order ({', '.join(catalog.VARIANTS)})")
    parser.add_argument("--no-shared", action="store_true", help="skip the text encoder and VAE")
    args = parser.parse_args()
    catalog.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    keys = args.variants or _ORDER
    unknown = [k for k in keys if k not in catalog.VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {unknown}")
    first, rest = keys[:1], keys[1:]
    # The first model before the 14 GB text encoder: a usable set sooner.
    for key in first:
        print(f"[{key}]", flush=True)
        fetch_variant(key)
    if not args.no_shared:
        print("[shared: VAE + Cosmos-Reason1-7B language model]", flush=True)
        fetch_shared()
    for key in rest:
        print(f"[{key}]", flush=True)
        fetch_variant(key)


if __name__ == "__main__":
    main()
