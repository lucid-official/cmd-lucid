"""Cosmos-Reason1 (Qwen2) tokenizer and the prompt template CMD wraps around a prompt.

Lucid's ``BPETokenizerFast`` runs the merges. Two parts of Qwen2's
``tokenizer.json`` pipeline need care, and are supplied here through the
public ``PreTokenizer`` interface:

* the split regex uses ``\\p{L}`` / ``\\p{N}``, which Python's ``re`` lacks;
  they are expanded into explicit code-point classes (and ``\\s`` into the
  Unicode ``White_Space`` set Oniguruma uses);
* the byte-level step must *not* re-split with GPT-2's regex
  (``"use_regex": false``), so it is applied here as a pure byte mapping.

Added tokens (``<|im_start|>`` ...) are matched on the raw text before
normalisation, like Hugging Face's added-token pass with
``split_special_tokens=False``.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

from lucid.utils.tokenizer import BPETokenizerFast, SpecialTokens
from lucid.utils.tokenizer import normalizers, pre_tokenizers

SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."
MAX_LENGTH = 512

# Unicode White_Space (what Oniguruma's \s matches for UTF-8 text).
_WHITE_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680),
                (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def _class(ranges: list[tuple[int, int]]) -> str:
    def esc(c: int) -> str:
        return f"\\U{c:08x}"
    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in ranges)


@lru_cache(maxsize=1)
def _categories() -> tuple[str, str]:
    """Code-point classes for ``\\p{L}`` and ``\\p{N}`` from ``unicodedata``."""
    found: dict[str, list[tuple[int, int]]] = {"L": [], "N": []}
    for cp in range(sys.maxunicode + 1):
        major = unicodedata.category(chr(cp))[0]
        if major in found:
            ranges = found[major]
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1] = (ranges[-1][0], cp)
            else:
                ranges.append((cp, cp))
    return _class(found["L"]), _class(found["N"])


@lru_cache(maxsize=1)
def qwen2_split_pattern() -> re.Pattern[str]:
    """Qwen2's pre-tokenizer regex, with Unicode properties spelled out."""
    L, N = _categories()
    S = _class(_WHITE_SPACE)
    return re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|[^\r\n{L}{N}]?[{L}]+"
        rf"|[{N}]"
        rf"| ?[^{S}{L}{N}]+[\r\n]*"
        rf"|[{S}]*[\r\n]+"
        rf"|[{S}]+(?![^{S}])"
        rf"|[{S}]+"
    )


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's printable byte alphabet."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Qwen2PreTokenizer(pre_tokenizers.PreTokenizer):
    """``Split(regex, isolated)`` then ``ByteLevel(use_regex=False)``."""

    def __init__(self) -> None:
        self.pattern = qwen2_split_pattern()
        self.byte_map = _bytes_to_unicode()

    def pre_tokenize(self, text: str):
        chunks = []
        last = 0
        for m in self.pattern.finditer(text):
            if m.start() > last:  # isolated split keeps unmatched text as chunks
                chunks.append((text[last : m.start()], (last, m.start())))
            chunks.append((m.group(), m.span()))
            last = m.end()
        if last < len(text):
            chunks.append((text[last:], (last, len(text))))
        return [("".join(self.byte_map[b] for b in c.encode("utf-8")), span) for c, span in chunks]


class CosmosTokenizer:
    """Chat-template rendering + Qwen2 tokenization, padded to 512 ids."""

    def __init__(self, directory: str | Path, max_length: int = MAX_LENGTH) -> None:
        directory = Path(directory)
        config = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        self.added = {t["content"]: t["id"] for t in config["added_tokens"]}
        self.pad_id = self.added["<|endoftext|>"]
        self.max_length = max_length
        self.bpe = BPETokenizerFast.from_file(
            str(directory),
            normalizer=normalizers.NFC(),
            pre_tokenizer=Qwen2PreTokenizer(),
            special_tokens=SpecialTokens(pad="<|endoftext|>"),
        )
        longest_first = sorted(self.added, key=len, reverse=True)
        self._added_re = re.compile("|".join(re.escape(t) for t in longest_first))

    @staticmethod
    def render(prompt: str, system: str = SYSTEM_PROMPT) -> str:
        """``processor.apply_chat_template(..., add_generation_prompt=False)`` for one turn."""
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        last = 0
        for m in self._added_re.finditer(text):
            if m.start() > last:
                ids += self.bpe.encode(text[last : m.start()], add_special_tokens=False)
            ids.append(self.added[m.group()])
            last = m.end()
        if last < len(text):
            ids += self.bpe.encode(text[last:], add_special_tokens=False)
        return ids

    def __call__(self, prompt: str) -> list[int]:
        """Token ids of the rendered prompt, truncated and right-padded to ``max_length``."""
        ids = self.encode(self.render(prompt))[: self.max_length]
        return ids + [self.pad_id] * (self.max_length - len(ids))
