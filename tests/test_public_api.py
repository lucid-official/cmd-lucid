"""The implementation uses Lucid's public API and the standard library, nothing else."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "cmd_lucid"
FORBIDDEN_TOP = {"torch", "numpy", "np", "scipy", "PIL", "cv2", "einops", "safetensors", "transformers", "diffusers", "mlx"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


def test_no_third_party_or_private_lucid_imports():
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        for name in _imports(path):
            parts = name.split(".")
            if parts[0] in FORBIDDEN_TOP:
                offenders.append((path.name, name))
            if parts[0] == "lucid" and any(p.startswith("_") for p in parts[1:]):
                offenders.append((path.name, name))
    assert not offenders, offenders
