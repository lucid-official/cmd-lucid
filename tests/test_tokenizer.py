"""The Lucid-built Qwen2 tokenizer against Hugging Face's, id for id."""

import json

import pytest

from cmd_lucid import catalog
from cmd_lucid.text.tokenizer import CosmosTokenizer
from conftest import FIXTURES


@pytest.fixture(scope="module")
def tok():
    if not (catalog.text_encoder_dir() / "tokenizer.json").exists():
        pytest.skip("tokenizer not downloaded")
    return CosmosTokenizer(catalog.text_encoder_dir())


def _cases():
    path = FIXTURES / "tokenizer_hf.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["prompt"][:24])
def test_matches_hugging_face(tok, case):
    assert tok.render(case["prompt"]) == case["text"]
    assert tok(case["prompt"]) == case["ids"]
