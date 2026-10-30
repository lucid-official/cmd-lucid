"""Generate a video with a released CMD model.

    python -m cmd_lucid.generate chunk1-short \\
        --image assets/examples/image.png --prompt-file assets/examples/prompt.txt --seed 22

    python -m cmd_lucid.generate chunk4-camera \\
        --image assets/examples/camera_image.png --prompt-file assets/examples/camera_prompt.txt \\
        --camera assets/examples/camera.npz

The defaults reproduce ``examples/run_examples.sh`` of the reference: each
model's own length, chunk size and attention window, 480x832 at 16 fps.
Frames are decoded and written while later blocks are still generating.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lucid
from tqdm import tqdm

from cmd_lucid import catalog
from cmd_lucid.camera import build_camera_conditioning, load_trajectory
from cmd_lucid.io.media import VideoWriter, image_to_model, read_image, to_uint8_frames
from cmd_lucid.pipeline import NEGATIVE_PROMPT, CMDPipeline, NoiseSource
from cmd_lucid.text.encoder import PromptEncoder

_DTYPES = {"bf16": lucid.bfloat16, "fp16": lucid.float16, "fp32": lucid.float32}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model", choices=list(catalog.VARIANTS))
    p.add_argument("--image", required=True)
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument(
        "--camera",
        help="trajectory .npz (target_w2c, target_intrinsics) for camera models",
    )
    p.add_argument(
        "--frames",
        type=int,
        help="latent frames incl. the image (default: the model's)",
    )
    p.add_argument("--seed", type=int, default=22)
    p.add_argument(
        "--dtype",
        choices=list(_DTYPES),
        default="bf16",
        help="DiT/VAE compute dtype (reference: bf16)",
    )
    p.add_argument("--out", help="output .mp4 (default: outputs/<model>-<seed>.mp4)")
    args = p.parse_args()

    prompt = (
        args.prompt
        if args.prompt is not None
        else Path(args.prompt_file).read_text(encoding="utf-8").strip()
    )
    variant = catalog.VARIANTS[args.model]
    frames = args.frames or variant.latent_frames
    out = Path(args.out or f"outputs/{args.model}-{args.seed}.mp4")

    t0 = time.time()
    encoder = PromptEncoder()
    embeds = encoder(prompt)
    negative = encoder(NEGATIVE_PROMPT) if variant.teacher else None
    print(
        f"prompt features {tuple(embeds.shape)} in {time.time() - t0:.1f}s", flush=True
    )

    t0 = time.time()
    pipe = CMDPipeline.load(args.model, dtype=_DTYPES[args.dtype])
    print(f"loaded {args.model} in {time.time() - t0:.1f}s", flush=True)

    camera = None
    if variant.camera:
        if not args.camera:
            p.error(f"{args.model} needs --camera")
        c2w, k = load_trajectory(args.camera, frames)
        tokens = build_camera_conditioning(
            c2w, k, frames_per_block=variant.block, expected_latent_frames=frames
        )
        tokens = tokens.to(pipe.device).to(pipe.dtype)
        camera = lambda start, n: tokens[:, :, start : start + n]  # noqa: E731

    latent = pipe.encode_image(image_to_model(read_image(args.image)))
    decode = pipe.decoder()
    writer = VideoWriter(out, 832, 480)
    t0, written = time.time(), 0
    bar = tqdm(
        total=4 * (frames - 1) + 1,
        unit="frame",
        desc=f"{args.model} -> {out.name}",
        dynamic_ncols=True,
    )
    for block in pipe.stream(
        latent, embeds, frames, NoiseSource(args.seed), camera, negative
    ):
        pixels = decode(block.latents)
        writer.write(to_uint8_frames(pixels))
        written += pixels.shape[1]
        bar.update(pixels.shape[1])
        bar.set_postfix_str(
            f"latent {block.start}/{frames - 1}, {block.seconds:.1f}s/block, video {written / 16:.1f}s"
        )
    bar.close()
    writer.close()
    print(
        f"wrote {out}: {written} frames ({written / 16:.1f}s of video) in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
