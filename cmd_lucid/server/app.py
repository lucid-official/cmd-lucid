"""Live demo: image + prompt (+ camera keys) in, streaming video out.

    python -m cmd_lucid.server                 # binds the Tailscale IP, port 7860
    python -m cmd_lucid.server --host 0.0.0.0  # any interface

Standard library only (``http.server`` + Server-Sent Events). One worker
thread owns the models and generates one session at a time; HTTP threads stay
responsive because Lucid releases the GIL while the GPU works. For camera
models the browser streams the held keys (W/A/S/D move, Q/E rise/sink, arrows
look), and each latent frame takes the keys held when it starts — every
frame for chunk-1 models, every fourth for chunk-4 models.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import lucid

from cmd_lucid import catalog
from cmd_lucid.camera import CameraAction, CameraController
from cmd_lucid.io.media import VideoWriter, encode_jpeg, image_from_bytes, image_to_model, read_image, to_uint8_frames
from cmd_lucid.pipeline import NEGATIVE_PROMPT, CMDPipeline, NoiseSource
from cmd_lucid.text.encoder import PromptEncoder

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = catalog.ROOT / "assets" / "examples"
OUTPUTS = catalog.ROOT / "outputs" / "web"
_DTYPES = {"bf16": lucid.bfloat16, "fp16": lucid.float16}


@dataclass
class Session:
    id: str
    model: str
    prompt: str
    image: bytes | None
    example: str | None
    seed: int
    max_frames: int
    speed: float
    turn: float
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    keys: str = ""  # held right now
    latched: set = field(default_factory=set)  # pressed since the last frame took its keys
    stop: bool = False
    video: Path | None = None
    poses: list = field(default_factory=list)

    def emit(self, kind: str, **data) -> None:
        self.events.put({"type": kind, "t": time.time(), **data})


class Engine:
    """Owns the prompt encoder and one loaded CMD model; runs sessions on its own thread."""

    def __init__(self, dtype: str = "bf16") -> None:
        self.dtype = _DTYPES[dtype]
        self.encoder: PromptEncoder | None = None
        self.pipe: CMDPipeline | None = None
        self.jobs: "queue.Queue[Session]" = queue.Queue()
        self.current: Session | None = None
        threading.Thread(target=self._loop, daemon=True, name="cmd-engine").start()

    def submit(self, session: Session) -> None:
        if self.current is not None:
            self.current.stop = True
        self.jobs.put(session)

    def _loop(self) -> None:
        while True:
            session = self.jobs.get()
            if session.stop:
                continue
            self.current = session
            try:
                self._run(session)
            except Exception as exc:  # report to the browser, keep serving
                traceback.print_exc()
                session.emit("error", message=f"{type(exc).__name__}: {exc}")
            finally:
                session.emit("end")
                self.current = None

    def _ensure(self, key: str, session: Session) -> CMDPipeline:
        if self.pipe is None or self.pipe.variant.key != key:
            session.emit("status", message=f"loading {key} ...")
            self.pipe = None
            lucid.metal.empty_cache()
            t0 = time.time()
            self.pipe = CMDPipeline.load(key, dtype=self.dtype)
            session.emit("status", message=f"loaded {key} in {time.time() - t0:.0f}s")
        return self.pipe

    def _run(self, s: Session) -> None:
        variant = catalog.VARIANTS[s.model]
        if self.encoder is None:
            self.encoder = PromptEncoder()
        if not self.encoder.cached(s.prompt) or (variant.teacher and not self.encoder.cached(NEGATIVE_PROMPT)):
            s.emit("status", message="encoding the prompt with Cosmos-Reason1-7B ...")
            if self.pipe is not None:  # make room for the 14 GB text encoder
                self.pipe = None
                lucid.metal.empty_cache()
        t0 = time.time()
        embeds = self.encoder(s.prompt)
        negative = self.encoder(NEGATIVE_PROMPT) if variant.teacher else None
        s.emit("status", message=f"prompt features ready ({time.time() - t0:.1f}s)")
        pipe = self._ensure(s.model, s)
        image = image_from_bytes(s.image) if s.image else read_image(EXAMPLES / s.example)
        latent = pipe.encode_image(image_to_model(image))

        controller = CameraController() if variant.camera else None
        tokens_cache: dict[int, lucid.Tensor] = {}

        applied: list[str] = []

        def camera(start: int, n: int) -> lucid.Tensor:
            if start not in tokens_cache:
                while len(controller.poses) < start + n:
                    # Held keys plus anything tapped since the previous frame started.
                    keys = "".join(sorted(set(s.keys) | s.latched))
                    s.latched.clear()
                    applied.append(keys)
                    controller.advance(CameraAction.from_keys(keys, s.speed, s.turn))
                    p = controller.poses[-1]
                    s.poses.append([p[0][3], p[1][3], p[2][3], p[0][2], p[2][2]])
                tokens_cache.clear()
                tokens_cache[start] = controller.tokens(start, n, variant.block).to(pipe.device).to(pipe.dtype)
            return tokens_cache[start]

        endless = s.max_frames <= 0
        frames = None if endless else s.max_frames
        decode = pipe.decoder()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        s.video = OUTPUTS / f"{s.id}.mp4"
        writer = VideoWriter(s.video, 832, 480)
        index = 0
        s.emit("start", model=s.model, block=variant.block, camera=variant.camera, frames=frames)
        stream = pipe.stream(latent, embeds, frames, NoiseSource(s.seed), camera if controller else None,
                             negative, endless=endless)
        try:
            for block in stream:
                pixels = decode(block.latents)
                rgb = to_uint8_frames(pixels)
                writer.write(rgb)
                jpegs = [base64.b64encode(encode_jpeg(f, 832, 480)).decode() for f in rgb]
                n_lat = block.latents.shape[2]
                s.emit("frames", start=index, images=jpegs, latent=block.start, seconds=round(block.seconds, 2),
                       poses=s.poses[-n_lat:] if controller else [],
                       applied=applied[-n_lat:] if controller and block.start > 0 else [])
                index += len(rgb)
                if s.stop:
                    break
        finally:
            stream.close()
            writer.close()
        s.emit("done", frames=index, video=f"/api/video/{s.id}.mp4")


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    sessions: dict[str, Session] = {}

    def log_message(self, fmt, *args) -> None:  # quiet
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if url.path == "/api/info":
            return self._json(_info())
        if url.path.startswith("/api/example/"):
            name = Path(url.path.removeprefix("/api/example/")).name
            ctype = "image/png" if name.endswith(".png") else "text/plain; charset=utf-8"
            return self._file(EXAMPLES / name, ctype)
        if url.path.startswith("/api/video/"):
            return self._file(OUTPUTS / Path(url.path).name, "video/mp4")
        if url.path == "/api/events":
            sid = parse_qs(url.query).get("session", [""])[0]
            session = self.sessions.get(sid)
            if session is None:
                return self._json({"error": "unknown session"}, 404)
            return self._events(session)
        self.send_error(404)

    def _events(self, session: Session) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        while True:
            try:
                event = session.events.get(timeout=15)
            except queue.Empty:
                event = {"type": "ping"}
            try:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if event["type"] == "end":
                return

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if url.path == "/api/start":
            model = body.get("model", "chunk1-camera")
            if model not in catalog.VARIANTS or not catalog.VARIANTS[model].path.exists():
                return self._json({"error": f"model {model} is not downloaded"}, 400)
            image = base64.b64decode(body["image"].split(",", 1)[-1]) if body.get("image") else None
            session = Session(
                id=uuid.uuid4().hex[:12], model=model, prompt=str(body.get("prompt", "")).strip(), image=image,
                example=body.get("example") or "image.png", seed=int(body.get("seed", 22)),
                max_frames=int(body.get("frames", 0)), speed=float(body.get("speed", 0.16)),
                turn=float(body.get("turn", 3.0)),
            )
            if not session.prompt:
                return self._json({"error": "a prompt is required"}, 400)
            self.sessions[session.id] = session
            self.engine.submit(session)
            return self._json({"session": session.id})
        session = self.sessions.get(body.get("session", ""))
        if session is None:
            return self._json({"error": "unknown session"}, 404)
        if url.path == "/api/keys":
            session.keys = "".join(k for k in str(body.get("keys", "")) if k in "wasdqeijkl")
            session.latched.update(session.keys)
            return self._json({"ok": True})
        if url.path == "/api/stop":
            session.stop = True
            return self._json({"ok": True})
        self.send_error(404)


def _info() -> dict:
    models = [
        {"key": v.key, "camera": v.camera, "teacher": v.teacher, "block": v.block, "local_attn": v.local_attn,
         "latent_frames": v.latent_frames, "pixel_frames": v.pixel_frames, "available": v.path.exists()}
        for v in catalog.VARIANTS.values()
    ]
    examples = [p.name for p in sorted(EXAMPLES.glob("*.png"))]
    return {"models": models, "examples": examples, "device": lucid.metal.get_device_name()}


def _tailscale_ip() -> str | None:
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        return out[0] if out else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", help="default: the Tailscale IPv4 address, else 127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    args = p.parse_args()
    host = args.host or _tailscale_ip() or "127.0.0.1"
    Handler.engine = Engine(args.dtype)
    server = ThreadingHTTPServer((host, args.port), Handler)
    server.daemon_threads = True
    print(f"CMD demo on http://{host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
