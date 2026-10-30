"""Shared test helpers. Tests run in the Lucid venv; fixtures come from tests/ref/ (torch venv)."""

from pathlib import Path

import lucid
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEVICES = ["cpu", "metal"]


def load_fixture(name: str) -> dict[str, lucid.Tensor]:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(
            f"fixture {name} missing; run the matching tests/ref/ script in .venv-ref"
        )
    return lucid.load_safetensors(str(path))


def max_rel(a: lucid.Tensor, b: lucid.Tensor) -> float:
    a, b = a.float().to("cpu"), b.float().to("cpu")
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)
