<h1 align="center">CMD on Lucid</h1>

<p align="center">
  <b>Causal, camera-controllable image-to-video generation, natively on Apple Silicon.</b><br>
  A torch-free re-implementation of <a href="https://arxiv.org/abs/2608.13391">Context-Matched Distillation</a>
  on <a href="https://github.com/lucid-official/lucid">Lucid</a>. All ten released Cosmos-Predict2.5 2B checkpoints
  run unchanged on Metal, streaming video block by block as it is generated.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.13391"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2608.13391-b31b1b.svg"></a>
  <a href="https://github.com/nv-tlabs/cmd"><img alt="Upstream" src="https://img.shields.io/badge/upstream-nv--tlabs%2Fcmd-black.svg"></a>
  <a href="https://huggingface.co/nvidia/cmd"><img alt="Weights" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-nvidia%2Fcmd-orange.svg"></a>
  <a href="https://pypi.org/project/lucid-dl/3.16.2/"><img alt="Lucid" src="https://img.shields.io/badge/Lucid-3.16.2-4c8bf5.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-NVIDIA%20noncommercial-green.svg"></a>
</p>

<p align="center">
  <img src="assets/chunk1-short.gif" width="50%">
</p>

<p align="center">
  <sub>The upstream example, run here: <code>chunk1-short</code> with <code>examples/image.png</code>, its prompt and
  seed 22. 93 frames at 832×480, generated one latent frame (four video frames) at a time on a Mac Studio M4 Max.
  The seed does not reproduce upstream's video, whose noise comes from CUDA's generator.</sub>
</p>

---

## Highlights

- **The authors' weights, unchanged.** Every tensor of all ten `nvidia/cmd` checkpoints (six students, four causal
  teachers) loads with `strict=True` into this port's DiT. Each component matches an independent implementation:
  the DiT matches diffusers' Cosmos-Predict2.5 transformer to **2e-5**, the VAE matches diffusers to **1e-4**, the
  text encoder matches transformers' Qwen2.5-VL to **1e-4**, and the tokenizer matches Hugging Face's id for id.
- **Streaming, causal generation.** Each latent frame is denoised in four steps against a KV cache of the previous
  21, then decoded by a streaming VAE while the next one starts. The first frames appear after ~30 s.
- **Camera control from the keyboard.** The camera checkpoints take a ray map per latent frame. A web demo turns
  held W/A/S/D/Q/E and arrow keys into those poses live, and `--keys 'wl*16,w*224'` scripts the same from the
  command line.
- **The training side too.** The causal teacher's Diffusion-Forcing loss, the Self-Forcing rollout, and the
  context-matched DMD objective with Prefix Scoring and Prefix Corruption are all written on Lucid's autograd.
- **Written as a Lucid user would write it.** Models, pipeline and training use only Lucid's public API and the
  standard library. There is no torch and no numpy, and [`tests/test_public_api.py`](tests/test_public_api.py)
  enforces this.

## What this is

Context-Matched Distillation (Bandyopadhyay et al., 2026) distills a multi-step video model into a four-step
**causal** student, so a video can be extended frame by frame while new controls arrive. Earlier distillation
pipelines score each student frame with a *bidirectional* teacher, which can see future frames and future camera
poses the student did not have when it generated that frame. CMD scores every target with a *causal* teacher,
under the history the student actually generated (Prefix Scoring), and lightly noises that history while early
rollouts are still unreliable (Prefix Corruption).

This repository contains:

- **The inference stack:**
  - the Cosmos-Predict2.5 2B DiT, with block-causal attention and a sliding KV cache;
  - the Wan2.1 VAE, with streaming encode and decode;
  - the Cosmos-Reason1-7B (Qwen2.5-VL) text encoder, which turns a prompt into 28 normalised hidden states
    (100,352 channels);
  - the flow-matching and UniPC schedules;
  - camera-ray conditioning;
  - `CausalInferencePipeline` as a generator that yields each finished block.
- **The released models,** fetched from Hugging Face by HTTP range requests. Only the tensors the pipeline needs
  are kept, in the bf16 the reference casts them to.
- **The training objectives and loops** of the paper's three stages.
- **A live web demo,** reachable over Tailscale, with an initial frame, a prompt, keyboard camera control and
  streamed frames.

## Results

### Long video: 31 seconds, one rollout

<p align="center">
  <img src="assets/forward-31s-4x.gif" width="70%">
</p>

<p align="center">
  <sub><code>chunk1-long</code> (distilled for 126 latent frames) from the same image, with the example prompt
  rewritten to keep moving forward instead of halting. 501 frames, shown at 4× speed. Generated in 22.8 min.</sub>
</p>

<p align="center">
  <img src="assets/forward-31s-strip.jpg" width="100%">
</p>

<p align="center">
  <sub>0, 6, 12, 18, 24 and 31 s. The rollout keeps moving through gate after gate and stays coherent. The first
  ~12 s are photographic; after that the look drifts towards flatter, more saturated, game-like textures. The model
  card names this, gradual quality drift over long rollouts, as a known limitation.</sub>
</p>

## Speed on an M4 Max

Mac Studio, M4 Max (40-core GPU), 36 GB. DiT in bf16, as the reference runs it; VAE in fp32 (see
[Lucid findings](#lucid-findings)).

| stage | time |
|---|---|
| prompt → features (Cosmos-Reason1-7B, 14 GB) | ~21 s the first time, including the load; cached on disk afterwards |
| DiT, one latent frame (4 denoising passes + 1 KV commit), `chunk1-*` | 3.3 s at the start, **9.0 s** once the 21-frame window is full |
| VAE decode, one latent frame → 4 video frames | **2.1 s** |
| 31 s video (`chunk1-long`, 501 frames) | 22.8 min |

At steady state that is about **11 s for 0.25 s of video**, roughly 45× slower than real time. Interactive camera
control therefore works turn by turn, not like a game. The DiT runs at ~10 TFLOPS, close to what this GPU
sustains, so even one network pass per latent frame (0.6 s) would exceed the 0.25 s budget.

Two fixes behind these numbers, found by profiling a 28-latent run:

- **Cached K/V are stored as contiguous copies.** Views of the projections kept an extra buffer alive per layer and
  frame (+180 MB per frame).
- **The 240×416 and 480×832 decoder stages take their four frames one at a time.** This is identical output, since
  every layer there is causal in time.

Together they cut memory in use from 26.6 to 18.2 GB (peak 30.3 → 19.8 GB) and made a run 1.7× faster end to end
(430 → 249 s), 2.4× at steady state.

**Lucid vs PyTorch (MPS) at this model's shapes:**

| op | Lucid 3.16.2 | torch 2.9.1, MPS |
|---|---|---|
| linear 1560×2048→8192, bf16 | 5.5 ms | **4.7 ms** |
| linear 1560×2048→8192, fp16 | **4.8 ms** | **4.7 ms** |
| attention, q 1560 × kv 32,760, bf16 | **42 ms** | 120 ms |

Attention is ~60% of a DiT pass, so the same model on PyTorch's MPS backend would be about 2× slower.

## Numerical parity

Running the upstream repository's own code was not possible in this setup, so each part is checked against an
independent, widely used implementation, in fp32 on both CPU and Metal (`python -m pytest`, 28 tests). Reference
outputs come from `tests/ref/*.py`, run in a separate torch environment.

| check | against | result |
|---|---|---|
| Cosmos DiT (random weights, Predict2.5 config) | diffusers `CosmosTransformer3DModel` | < 2e-5 rel. |
| streaming KV-cache chunks | the same model, one block-causal pass | < 2e-5 rel. |
| packed teacher-forcing scoring (CMD) | a block-causal pass over [history ‖ targets] | < 2e-5 rel. |
| Wan2.1 VAE encode + decode (released weights) | diffusers `AutoencoderKLWan` | < 1e-4 rel. |
| streaming VAE decode | full decode | bit-identical |
| Qwen2.5-VL text path, first 4 layers (released weights) + feature recipe | transformers `Qwen2_5_VLTextModel` | < 1e-4 rel. |
| Qwen2 tokenizer + chat template (CJK, emoji, digits, specials, truncation) | Hugging Face `tokenizers` | 9 / 9 prompts, every id |
| flow-matching table (1000 σ, t) | torch float32 arithmetic | bit-exact |
| image preprocessing (Pillow bilinear, 8-bit fixed point) | reproduced from Pillow's algorithm | exact integer arithmetic |

Reproduced numerics of the reference run:
- bf16 weights and activations, with fp32 islands where the reference uses them (adaLN modulation, the final
  layer, RoPE).
- RoPE frequencies derived from bf16-rounded buffers, because the reference casts them too.
- A quirk kept on purpose: a multi-frame KV commit attends only within its own chunk, as `CausalCosmosAttention`
  does.

## Lucid findings

Found while building this, each with a measured repro:

- **Half-precision `conv2d` on Metal loses accuracy over large channel reductions.** On the VAE decoder's shapes, the
  relative error is **16% in bf16** and 2% in fp16, against ~0.4% expected from rounding alone. `conv3d`,
  `linear` and SDPA are accurate. The VAE therefore runs in fp32.
- **A lazily built 480×832 residual block can spill to swap.** Evaluating the whole block at once kept every
  multi-GB intermediate alive and took ~50 s per latent frame. Evaluating after each causal conv brought it to ~2 s.
- **Since 3.15.2, fixed:**
  - BF16 SafeTensors load and save;
  - bf16 scalar ops;
  - `transpose(a, b)`;
  - `lucid.compile` with bf16;
  - the GIL is released during GPU evaluation, so the demo serves HTTP from threads in the model's process.

## Getting started

### Install

An Apple Silicon Mac, Python 3.14, and `ffmpeg` (`brew install ffmpeg`) for image and video I/O.

```bash
git clone https://github.com/lucid-official/cmd-lucid && cd cmd-lucid
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -e ".[test]"
python -m pytest -q
```

### Download the weights

```bash
python -m cmd_lucid.download chunk1-short   # one model + the shared VAE and text encoder (~19 GB)
python -m cmd_lucid.download                # all ten models (53 GB on disk)
```

Downloads are range requests. Only the language model of Cosmos-Reason1-7B is kept, and the fp32 DiTs are stored
in bf16. The VAE is read from the ungated Wan2.1 release. It has the same size and the same latent statistics as
the copy Cosmos-Predict2.5 ships behind a gate.

### Generate

```bash
# the upstream examples (examples/run_examples.sh)
python -m cmd_lucid.generate chunk1-short --image assets/examples/image.png --prompt-file assets/examples/prompt.txt
python -m cmd_lucid.generate chunk1-long  --image assets/examples/image.png --prompt-file assets/examples/prompt.txt
python -m cmd_lucid.generate chunk4-camera --image assets/examples/camera_image.png \
    --prompt-file assets/examples/camera_prompt.txt --camera assets/examples/camera.npz

# camera models driven by held keys: W+→ for 16 latent frames (4 s), then W for 56 s
python -m cmd_lucid.generate chunk1-camera --image my.png --prompt "..." --keys 'wl*16,w*224' --turn 2.5
```

A progress bar shows frames written, the latent index and the seconds of video so far. Frames are decoded and
written while later blocks are still generating. `--dtype fp16` is ~5–10% faster than the reference's bf16.

### Web demo

```bash
python -m cmd_lucid.server        # binds the Tailscale IP if there is one, else 127.0.0.1; port 7860
```

Choose a model, drop or paste an image, write a prompt, press **Generate**. With a camera model, hold W/A/S/D (move),
Q/E (rise / sink) and the arrow keys (look) while it generates. Physical keys are read, so any input language
works, and a tap between frames is applied to the next one. The page shows the keys each latent frame used, a
top-down map of the camera path, and an MP4 to download at the end.

## Training

`cmd_lucid/distill/` ports the three stages of the paper:

| stage | reference | here |
|---|---|---|
| 1. causal teacher | `model/diffusion.py`: Diffusion Forcing, independent noise per block, block-causal mask | `DiffusionForcingTeacher`, `TeacherTrainer` |
| 2. context-matched distillation | `model/context_matched_distillation.py`: causal real/fake scores over [image ‖ noised prefix ‖ DMD targets] in one packed pass, critic flow loss | `ContextMatchedDistillation`, `DistillationTrainer` |
| 3. long video | rollout chunks with cached student history and a per-frame prefix-noise schedule | same classes, `rollout_num_chunks > 1` |

The same code runs at any size, but the released configuration needs three 2B networks plus optimizer state
(~80 GB), beyond a 36 GB Mac. `python -m cmd_lucid.distill.toy` runs it end to end on a tiny DiT with synthetic
bimodal "videos". **So far the toy run does not show the improvement it was built to demonstrate.** The toy teacher
was trained without prompt dropout, so there is no useful guidance signal, and with real and fake scores starting
equal the DMD gradient is ~0. The student drifts under Adam instead of improving. The experiment needs redesigning
before it says anything about CMD.

One reading of the released code: in the short-video configs the scored window covers the whole rollout, so the
"prefix" packed before the targets is just the image latent. Prefix Scoring with a generated prefix and Prefix
Corruption act across rollout chunks, in the long-video stage.

## Repository layout

```
cmd_lucid/
  models/cosmos_dit.py   Cosmos-Predict2.5 2B DiT, KV cache, block-causal / streaming / packed attention
  models/wan_vae.py      Wan2.1 VAE, streaming encode and decode
  text/                  Qwen2 tokenizer (exact pre-tokenizer), Qwen2.5-VL text model, cached prompt encoder
  scheduler.py           flow-matching table (bit-exact), UniPC for the teachers
  camera.py              ray-map conditioning, keyboard camera controller
  pipeline.py            CausalInferencePipeline as a block generator (students and teachers)
  generate.py            CLI
  distill/               training objectives, trainers, toy experiment
  server/                web demo (stdlib HTTP + Server-Sent Events)
  hub.py, download.py    range-request downloads into bf16 SafeTensors
  io/                    torch.save reader (no unpickling of code), .npy reader, ffmpeg media I/O
  accel.py               lucid.compile streaming-decoder programs
tests/                   parity and behaviour tests; tests/ref/ makes the reference outputs in a torch venv
assets/                  README media and the upstream example inputs
```

## Limitations

- **Not checked against the upstream code itself.** The components match diffusers and transformers. The end-to-end
  output also cannot match upstream's videos frame for frame, because their noise comes from CUDA's generator.
- **No benchmark numbers.** VBench and the camera-error metrics of the paper were not computed here.
- **Far from real time.** About 11 s per 0.25 s of video at steady state on an M4 Max.
- **`lucid.compile` and `lucid.coreml` do not speed anything up yet.** The compiled VAE decoder matches eager but is
  not faster, attention-bound DiT passes keep MLX's fused attention, and no Core ML program is used by default.
- **Camera rollouts are short-horizon.** The camera checkpoints were distilled on ~8 s clips; rollouts far past
  that drift and can collapse.
- **The training toy run is not yet a successful demonstration** (see [Training](#training)).

## Citation

```bibtex
@article{bandyopadhyay2026context,
  title   = {Context-Matched Distillation: Teacher Causality for Autoregressive Video Distillation},
  author  = {Bandyopadhyay, Hmrishav and Ren, Xuanchi and Huang, Zijian and Wu, Jay Zhangjie and Cao, Tianshi
             and Li, Ruilong and Chu, Bryan and Fidler, Sanja and Song, Yi-Zhe and Wang, Zian},
  journal = {arXiv preprint arXiv:2608.13391},
  year    = {2026}
}
```

## Acknowledgements

- [CMD](https://github.com/nv-tlabs/cmd), for the method, the released code that settles what the paper leaves
  open, and the [checkpoints](https://huggingface.co/nvidia/cmd).
- [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) and
  [Cosmos-Reason1](https://huggingface.co/nvidia/Cosmos-Reason1-7B), for the backbone and the text encoder.
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing), which CMD builds on, and
  [Wan2.1](https://github.com/Wan-Video/Wan2.1), for the VAE.
- [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers),
  the independent implementations the parity tests compare against.
- [Lucid](https://github.com/lucid-official/lucid), for the Apple Silicon runtime.

No model weights are redistributed here; they are downloaded by the user. The videos in [`assets/`](assets/) were
generated by the released checkpoints with this code.

## License

The code ports NVIDIA's CMD implementation and is distributed under the
[NVIDIA OneWay Noncommercial License](LICENSE) (non-commercial research and education only). See [NOTICE](NOTICE)
for the Apache-2.0 projects it also derives from. The released weights carry the same license.
