"""Read ``torch.save`` archives without torch, and rewrite them as SafeTensors.

A ``torch.save`` file (zip format) holds ``<root>/data.pkl`` — a pickle whose
tensors are persistent references — and one uncompressed ``<root>/data/<key>``
entry per storage. The pickle is read by a restricted unpickler that knows
only the handful of names a state dict needs, so no code from the archive is
ever executed. Tensors come out as raw little-endian bytes, which a SafeTensors
header makes loadable by :func:`lucid.load_safetensors` (bf16 included).
"""

from __future__ import annotations

import collections
import json
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

_STORAGE_DTYPE = {
    "FloatStorage": ("F32", 4),
    "HalfStorage": ("F16", 2),
    "BFloat16Storage": ("BF16", 2),
    "DoubleStorage": ("F64", 8),
    "LongStorage": ("I64", 8),
    "IntStorage": ("I32", 4),
}


@dataclass
class _Storage:
    key: str
    dtype: str
    itemsize: int


@dataclass
class TensorRef:
    storage: _Storage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def is_contiguous(self) -> bool:
        expected = 1
        for size, step in zip(reversed(self.shape), reversed(self.stride)):
            if size != 1 and step != expected:
                return False
            expected *= size
        return True


class _StorageType:
    def __init__(self, name: str) -> None:
        self.name = name


def _rebuild_tensor_v2(storage, offset, shape, stride, *_args):
    return TensorRef(storage, int(offset), tuple(shape), tuple(stride))


class _Unpickler(pickle.Unpickler):
    _ALLOWED = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    }

    def find_class(self, module: str, name: str):
        if module == "torch" and name in _STORAGE_DTYPE:
            return _StorageType(name)
        try:
            return self._ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}") from None

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {kind!r}")
        dtype, itemsize = _STORAGE_DTYPE[storage_type.name]
        return _Storage(str(key), dtype, itemsize)


def read_state_dict(path: Path) -> tuple[zipfile.ZipFile, str, dict[str, TensorRef]]:
    archive = zipfile.ZipFile(path)
    pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
    root = pkl[: -len("/data.pkl")]
    with archive.open(pkl) as fh:
        state = _Unpickler(fh).load()
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    return archive, root, {k: v for k, v in state.items() if isinstance(v, TensorRef)}


def to_safetensors(src: Path, dest: Path, rename=lambda name: name) -> None:
    """Rewrite a ``torch.save`` state dict as a SafeTensors file, byte for byte."""
    archive, root, refs = read_state_dict(src)
    entries, header, offset = [], {}, 0
    for name, ref in refs.items():
        if not ref.is_contiguous():
            raise ValueError(f"{name}: non-contiguous tensors are not supported")
        nbytes = ref.numel * ref.storage.itemsize
        header[rename(name)] = {
            "dtype": ref.storage.dtype,
            "shape": list(ref.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        entries.append(ref)
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    cache: dict[str, bytes] = {}
    with open(part, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for ref in entries:
            blob = cache.get(ref.storage.key)
            if blob is None:
                blob = cache[ref.storage.key] = archive.read(
                    f"{root}/data/{ref.storage.key}"
                )
            start = ref.offset * ref.storage.itemsize
            fh.write(blob[start : start + ref.numel * ref.storage.itemsize])
    part.replace(dest)


def convert_wan_vae(src: Path, dest: Path) -> None:
    """The Wan2.1 VAE archive as SafeTensors, fp32 as released."""
    to_safetensors(src, dest)
    print(f"  wrote {dest}", flush=True)
