"""Reference token ids from Hugging Face's Qwen2 tokenizer (``.venv-ref``).

    .venv-ref/bin/python tests/ref/tokenizer_hf.py

Renders each prompt with Cosmos-Reason1's chat template exactly as
``CosmosTextEncoder.forward`` does, tokenizes with padding to 512, and saves
the rendered text and ids to ``tests/fixtures/tokenizer_hf.json``.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "You are a helpful assistant who will provide prompts to an image generator."
NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)
PROMPTS = [
    (ROOT / "tests/fixtures/example_prompt.txt").read_text(encoding="utf-8").strip(),
    (ROOT / "tests/fixtures/example_camera_prompt.txt")
    .read_text(encoding="utf-8")
    .strip(),
    NEGATIVE,
    "A corgi wearing sunglasses rides a surfboard, golden hour",
    "It's 2026; we've got 1234567 apples, 3.14159 pies & they're GONE!!!",
    "  leading spaces,\ttabs\nnewlines\r\n\r\nand   trailing   ",
    "Ünïcödé café naïve — “quotes” … 東京の夜景、カメラがゆっくり前進する。서울의 밤 🚀🌕",
    "<|im_start|> literal specials <|endoftext|> inside text",
    "word " * 700,
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/cosmos-reason1-7b"))
    out = []
    for prompt in PROMPTS:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        text = tok.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        ids = tok(text, padding="max_length", truncation=True, max_length=512)[
            "input_ids"
        ]
        out.append({"prompt": prompt, "text": text, "ids": ids})
    path = ROOT / "tests/fixtures/tokenizer_hf.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        "wrote", path, [len(o["ids"]) for o in out], "padding side:", tok.padding_side
    )


if __name__ == "__main__":
    main()
