"""Read ``.npy`` / ``.npz`` arrays into Lucid tensors with the standard library."""

from __future__ import annotations

import ast
import struct
import zipfile
from pathlib import Path

import lucid

_DTYPES = {
    "<f4": ("f", lucid.float32),
    "<f8": ("d", lucid.float64),
    "<i4": ("i", lucid.int32),
    "<i8": ("q", lucid.int64),
}


def parse_npy(raw: bytes) -> lucid.Tensor | None:
    """One ``.npy`` payload as a tensor, or ``None`` for dtypes Lucid has no use for (strings, objects)."""
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy payload")
    major = raw[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", raw[8:10])
        off = 10
    else:
        (hlen,) = struct.unpack("<I", raw[8:12])
        off = 12
    header = ast.literal_eval(raw[off : off + hlen].decode("latin1"))
    spec = _DTYPES.get(header["descr"])
    if spec is None or header["fortran_order"]:
        return None
    code, dtype = spec
    shape = tuple(header["shape"])
    data = memoryview(raw[off + hlen :])
    n = 1
    for d in shape:
        n *= d
    values = data[: n * struct.calcsize(code)].cast(code)
    return lucid.tensor(values, dtype=dtype).reshape(shape)


def load_npz(path: str | Path) -> dict[str, lucid.Tensor]:
    """Numeric arrays of an ``.npz`` archive, keyed by name without ``.npy``."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            tensor = parse_npy(archive.read(name))
            if tensor is not None:
                out[name.removesuffix(".npy")] = tensor
    return out
