"""Prompt -> 100,352-dim Cosmos features, cached on disk.

The 7B text encoder is 14 GB in bf16. It is loaded only when a prompt has no
cached features, used, and released before the video model runs, so the two
never share memory. Features are stored per prompt (bf16, ~100 MB) under
``checkpoints/_prompt_cache``.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import lucid

from cmd_lucid import catalog
from cmd_lucid.text.qwen import CosmosReasonEncoder
from cmd_lucid.text.tokenizer import CosmosTokenizer, SYSTEM_PROMPT


class PromptEncoder:
    """Turns prompts into ``prompt_embeds`` ``(1, 512, 100352)`` the way ``CosmosTextEncoder`` does."""

    def __init__(
        self,
        directory: Path | None = None,
        device: str = "metal",
        dtype=lucid.bfloat16,
        keep_loaded: bool = False,
    ) -> None:
        self.directory = Path(directory or catalog.text_encoder_dir())
        self.device = device
        self.dtype = dtype
        self.keep_loaded = keep_loaded
        self.cache_dir = catalog.CHECKPOINTS / "_prompt_cache"
        self.tokenizer = CosmosTokenizer(self.directory)
        self._model: CosmosReasonEncoder | None = None

    def _key(self, prompt: str) -> Path:
        digest = hashlib.sha256(
            f"{SYSTEM_PROMPT}\x00{prompt}\x00{self.dtype}".encode()
        ).hexdigest()[:24]
        return self.cache_dir / f"{digest}.safetensors"

    def cached(self, prompt: str) -> bool:
        return self._key(prompt).exists()

    def __call__(self, prompt: str) -> lucid.Tensor:
        path = self._key(prompt)
        if path.exists():
            return lucid.load_safetensors(str(path), device=self.device)[
                "prompt_embeds"
            ]
        if self._model is None:
            self._model = CosmosReasonEncoder.load(
                self.directory, self.device, self.dtype
            )
        ids = lucid.tensor(
            [self.tokenizer(prompt)], dtype=lucid.int64, device=self.device
        )
        with lucid.no_grad():
            embeds = self._model(ids)
            lucid.eval(embeds)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        lucid.save_safetensors({"prompt_embeds": embeds.to("cpu")}, str(part))
        part.replace(path)
        if not self.keep_loaded:
            self.release()
        return embeds

    def release(self) -> None:
        self._model = None
        gc.collect()
        lucid.metal.empty_cache()
