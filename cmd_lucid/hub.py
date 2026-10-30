"""Hugging Face downloads straight into Lucid-ready SafeTensors.

The released files are large (8 GB per fp32 DiT, 16.6 GB for the text encoder)
and only part of each is needed, so nothing is downloaded whole. A remote
SafeTensors file is read by HTTP range requests: the header first, then only
the selected tensors. Two ways to write them out:

* ``passthrough`` copies the selected tensors' bytes verbatim. Used for files
  already stored in the dtype the reference runs in (bf16), and resumable.
* ``cast`` builds each tensor in Lucid and casts it on the CPU, where the cast
  rounds to nearest-even with subnormals kept, exactly like ``Tensor.to`` in
  the reference. The reference casts every DiT to bf16 before inference
  (``pipeline.to(dtype=torch.bfloat16)``), so storing bf16 loses nothing.

Only the standard library and Lucid are used.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lucid

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
_CHUNK = 32 << 20
_ST_ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "BOOL": 1}
_ST_DTYPE = {"F32": lucid.float32, "F16": lucid.float16, "BF16": lucid.bfloat16}


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{filename}"


def _request(url: str, start: int | None = None, end: int | None = None, retries: int = 8):
    headers = {"User-Agent": "cmd-lucid"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end - 1}"
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise
            wait = min(60, 2**attempt)
            print(f"  retry in {wait}s after {exc}", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def read_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``[start, end)`` of ``url``, re-requesting the remainder on a dropped connection."""
    out = bytearray()
    while len(out) < end - start:
        try:
            with _request(url, start + len(out), end) as response:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out += block
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"  resume after {exc}", file=sys.stderr, flush=True)
            time.sleep(2)
    return bytes(out)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class RemoteSafetensors:
    """A SafeTensors file on the Hub, read by range requests."""

    def __init__(self, repo: str, filename: str, revision: str = "main") -> None:
        self.url = hf_url(repo, filename, revision)
        self.name = f"{repo}/{filename}"
        (length,) = struct.unpack("<Q", read_range(self.url, 0, 8))
        header = json.loads(read_range(self.url, 8, 8 + length))
        self.metadata = header.pop("__metadata__", None) or {}
        self.base = 8 + length
        self.tensors = sorted(
            (
                TensorInfo(k, v["dtype"], tuple(v["shape"]), *v["data_offsets"])
                for k, v in header.items()
            ),
            key=lambda t: t.start,
        )

    def read(self, info: TensorInfo) -> bytes:
        return read_range(self.url, self.base + info.start, self.base + info.end)


def _write_header(fh, entries: list[tuple[str, str, tuple[int, ...], int]]) -> int:
    header, offset = {}, 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    fh.write(struct.pack("<Q", len(encoded)))
    fh.write(encoded)
    return 8 + len(encoded)


def fetch_passthrough(
    remote: RemoteSafetensors,
    dest: Path,
    select: Callable[[str], bool] = lambda name: True,
) -> None:
    """Copy the selected tensors byte for byte. Resumes an interrupted ``.part`` file."""
    if dest.exists():
        return
    picked = [t for t in remote.tensors if select(t.name)]
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = sum(t.nbytes for t in picked)
    mode = "r+b" if part.exists() else "wb"
    with open(part, mode) as fh:
        data_start = _write_header(fh, [(t.name, t.dtype, t.shape, t.nbytes) for t in picked])
        done = max(0, os.path.getsize(part) - data_start)
        fh.seek(data_start + done)
        # Coalesce runs of tensors that sit back to back in the remote file.
        runs: list[list[TensorInfo]] = []
        for t in picked:
            if runs and runs[-1][-1].end == t.start:
                runs[-1].append(t)
            else:
                runs.append([t])
        written, t0 = 0, time.time()
        for run in runs:
            size = run[-1].end - run[0].start
            if written + size <= done:
                written += size
                continue
            skip = max(0, done - written)
            pos = remote.base + run[0].start + skip
            stop = remote.base + run[-1].end
            while pos < stop:
                block = read_range(remote.url, pos, min(stop, pos + _CHUNK))
                fh.write(block)
                pos += len(block)
                _progress(remote.name, data_start, fh.tell() - data_start, total, t0, done)
            written += size
    os.replace(part, dest)
    print(f"\n  wrote {dest} ({total / 1e9:.2f} GB)", flush=True)


def fetch_cast(
    remote: RemoteSafetensors,
    dest: Path,
    dtype: str = "BF16",
    select: Callable[[str], bool] = lambda name: True,
) -> None:
    """Download the selected tensors, cast floating ones to ``dtype`` on the CPU, save."""
    if dest.exists():
        return
    target = _ST_DTYPE[dtype]
    picked = [t for t in remote.tensors if select(t.name)]
    total = sum(t.nbytes for t in picked)
    out: dict[str, lucid.Tensor] = {}
    got, t0 = 0, time.time()
    for t in picked:
        raw = remote.read(t)
        got += len(raw)
        if t.dtype == "F32":
            value = lucid.tensor(memoryview(raw).cast("f")).reshape(t.shape)
        elif t.dtype in _ST_DTYPE:
            value = _from_raw(raw, t)
        else:
            raise ValueError(f"{t.name}: unexpected dtype {t.dtype}")
        out[t.name] = value.to(target) if value.dtype != target else value
        _progress(remote.name, 0, got, total, t0, 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    lucid.save_safetensors(out, str(part))
    os.replace(part, dest)
    print(f"\n  wrote {dest}", flush=True)


def _from_raw(raw: bytes, info: TensorInfo) -> lucid.Tensor:
    """A non-fp32 SafeTensors entry as a Lucid tensor, through a one-entry file."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
        _write_header(fh, [(info.name, info.dtype, info.shape, len(raw))])
        fh.write(raw)
        path = fh.name
    try:
        return lucid.load_safetensors(path)[info.name]
    finally:
        os.unlink(path)


def fetch_file(repo: str, filename: str, dest: Path, revision: str = "main") -> None:
    """Download a whole (small or non-SafeTensors) file, resuming a ``.part``."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = hf_url(repo, filename, revision)
    with _request(url) as head:
        total = int(head.headers.get("Content-Length", 0))
    done = part.stat().st_size if part.exists() else 0
    t0 = time.time()
    with open(part, "ab") as fh:
        while done < total:
            block = read_range(url, done, min(total, done + _CHUNK))
            fh.write(block)
            done += len(block)
            _progress(f"{repo}/{filename}", 0, done, total, t0, 0)
    os.replace(part, dest)
    if total > _CHUNK:
        print(f"\n  wrote {dest}", flush=True)


def _progress(name: str, _base: int, done: int, total: int, t0: float, resumed: int) -> None:
    rate = (done - resumed) / max(time.time() - t0, 1e-6) / 1e6
    sys.stdout.write(f"\r  {name}: {done / 1e9:6.2f}/{total / 1e9:.2f} GB  {rate:5.1f} MB/s")
    sys.stdout.flush()
