"""OpenAI-compatible chat completions over a transformers vision-language model.

One process, one model, one request at a time. The endpoint speaks the
subset of the OpenAI chat API that grpc-vlm-convert sends: a user message
whose content mixes text parts and image_url parts (base64 data URIs; http
URLs only when the operator allows fetching), max_tokens, temperature,
stop. Anything else is ignored with a warning rather than refused, so a
client that sends more keeps working.

Whoever can reach the port gets a GPU, so the defaults are closed:
compose publishes the port on 127.0.0.1 only, remote image URLs are not
fetched (NORTH_ALLOW_REMOTE_IMAGES=1 turns that on), bodies need a
Content-Length under NORTH_MAX_BODY_BYTES, images are capped in bytes and
pixels, and NORTH_API_KEY, when set, is required as a bearer key on /v1/.

The device is chosen at startup: NORTH_DEVICE=auto picks CUDA, then Intel
XPU, then CPU. bfloat16 wherever the device supports it.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import io
import logging
import os
import threading
import time
import uuid
import warnings
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor


def env_flag(name: str) -> bool:
    """1/true or 0/false; anything else stops the server rather than guess."""
    value = os.environ.get(name, "0").strip().lower()
    if value in ("1", "true"):
        return True
    if value in ("", "0", "false"):
        return False
    raise SystemExit(f"{name} must be 1, 0, true or false")


MODEL_ID = os.environ.get("NORTH_MODEL_ID", "CohereLabs/North-Micro-Vision-Instruct")
# A commit hash pins the weights; unset follows the repository's main branch.
MODEL_REVISION = os.environ.get("NORTH_MODEL_REVISION") or None
MAX_NEW_TOKENS_CAP = int(os.environ.get("NORTH_MAX_NEW_TOKENS", "8192"))
# Request bodies past this are refused before a byte of them is read.
MAX_BODY_BYTES = int(os.environ.get("NORTH_MAX_BODY_BYTES", str(64 * 1024 * 1024)))
# One image's encoded bytes, inline or fetched.
MAX_IMAGE_BYTES = int(os.environ.get("NORTH_MAX_IMAGE_BYTES", str(32 * 1024 * 1024)))
# Fetching http(s) image_url parts lets any caller make this server request
# any URL it can reach (cloud metadata, internal services): off by default.
ALLOW_REMOTE_IMAGES = env_flag("NORTH_ALLOW_REMOTE_IMAGES")
# Bearer key for /v1/ routes; empty leaves them open. Never logged.
API_KEY = os.environ.get("NORTH_API_KEY", "")

# A small, highly compressible file can declare a gigapixel image. PIL
# refuses past twice MAX_IMAGE_PIXELS but only warns between one and two
# times it; the warning is made an error so the cap is the cap.
Image.MAX_IMAGE_PIXELS = int(os.environ.get("NORTH_MAX_IMAGE_PIXELS", "40000000"))
warnings.simplefilter("error", Image.DecompressionBombWarning)

log = logging.getLogger("north-vision")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def pick_device() -> torch.device:
    wanted = os.environ.get("NORTH_DEVICE", "auto")
    if wanted == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    device = torch.device(wanted)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("NORTH_DEVICE=cuda but CUDA is not available")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("NORTH_DEVICE=xpu but no Intel XPU is available")
    return device


class Engine:
    def __init__(self) -> None:
        self.device = pick_device()
        # bf16 is the checkpoint precision; every supported accelerator and
        # modern CPUs run it. float32 only where bf16 matmul is missing.
        self.dtype = torch.bfloat16
        if self.device.type == "cpu" and not torch.backends.mkldnn.is_available():
            self.dtype = torch.float32
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=self.dtype,
        )
        self.model.to(self.device).eval()
        self.lock = threading.Lock()
        log.info(
            "loaded %s on %s (%s) in %.1fs", MODEL_ID, self.device, self.dtype,
            time.monotonic() - started,
        )

    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(0)
        if self.device.type == "xpu":
            return torch.xpu.get_device_name(0)
        return "cpu"

    @torch.inference_mode()
    def generate(
        self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float,
        top_p: float, top_k: int, repetition_penalty: float, stop: list[str],
    ) -> tuple[str, int, int, str]:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            kwargs.update(do_sample=False)
        if stop:
            kwargs.update(stop_strings=stop, tokenizer=self.processor.tokenizer)
        with self.lock:
            outputs = self.model.generate(**inputs, **kwargs)
        generated = outputs[0][prompt_tokens:]
        completion_tokens = int(generated.shape[0])
        text = self.processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        for marker in stop:
            if text.endswith(marker):
                text = text[: -len(marker)]
        finish = "length" if completion_tokens >= max_new_tokens else "stop"
        return text, prompt_tokens, completion_tokens, finish


def open_image(raw: bytes) -> Image.Image:
    """Decodes one image within the byte and pixel caps; 4xx otherwise."""
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"image is above the {MAX_IMAGE_BYTES}-byte cap")
    too_large = HTTPException(413, f"image is above the {Image.MAX_IMAGE_PIXELS}-pixel decode cap")
    try:
        # open() reads the header only; the pixels are decoded by convert().
        image = Image.open(io.BytesIO(raw))
        # Checked here as well as by the warning filter above, which a host
        # process (a test runner, say) may have reset.
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise too_large
        return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise too_large from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "image_url does not hold a decodable image") from None


def decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            raise HTTPException(400, "only base64 data URIs are accepted")
        # Four base64 characters per three bytes: refuse before decoding
        # what could not fit the cap anyway.
        if len(payload) > (MAX_IMAGE_BYTES + 2) // 3 * 4 + 4:
            raise HTTPException(413, f"image is above the {MAX_IMAGE_BYTES}-byte cap")
        try:
            raw = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            raise HTTPException(400, "image_url data URI is not valid base64") from None
        return open_image(raw)
    if url.startswith("http://") or url.startswith("https://"):
        if not ALLOW_REMOTE_IMAGES:
            raise HTTPException(
                400,
                "remote image URLs are not fetched; send the image as a base64 data URI "
                "(the operator can allow fetching with NORTH_ALLOW_REMOTE_IMAGES=1)",
            )
        import urllib.request

        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            # One byte past the cap is enough to know it does not fit.
            return open_image(response.read(MAX_IMAGE_BYTES + 1))
    raise HTTPException(400, "image_url must be a data URI or an http(s) URL")


def to_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The OpenAI message shapes, as the processor's chat template wants them."""
    converted = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif kind == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append({"type": "image", "image": decode_image(url)})
                else:
                    log.warning("ignoring content part of type %r", kind)
        converted.append({"role": role, "content": parts})
    return converted


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = 0.0
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    stop: list[str] | str | None = None
    stream: bool = False


class Guard:
    """What must be settled before a request body is read.

    /v1/ routes need the bearer key when NORTH_API_KEY is set (the key is
    compared in constant time and never logged). A request with a body
    needs a Content-Length no larger than NORTH_MAX_BODY_BYTES: uvicorn
    then reads at most that many bytes, so the cap holds before anything
    is buffered; a chunked body, whose size is unknown up front, is
    refused.
    """

    def __init__(self, app: Any, max_body_bytes: int, api_key: str) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.api_key = api_key.encode()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        if self.api_key and scope.get("path", "").startswith("/v1/"):
            supplied = headers.get(b"authorization", b"")
            token = supplied[len(b"Bearer "):] if supplied.startswith(b"Bearer ") else b""
            if not hmac.compare_digest(token, self.api_key):
                await self.refuse(send, 401, "missing or wrong bearer key",
                                  [(b"www-authenticate", b"Bearer")])
                return
        if b"transfer-encoding" in headers:
            await self.refuse(send, 411, "send the body with a Content-Length")
            return
        length = headers.get(b"content-length")
        if length is not None:
            if not length.isdigit():
                await self.refuse(send, 400, "malformed Content-Length")
                return
            if int(length) > self.max_body_bytes:
                await self.refuse(
                    send, 413, f"request body is above the {self.max_body_bytes}-byte cap",
                )
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def refuse(send: Any, status: int, detail: str,
                     extra: list[tuple[bytes, bytes]] | None = None) -> None:
        body = ('{"detail": "' + detail + '"}').encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"connection", b"close"), *(extra or [])],
        })
        await send({"type": "http.response.body", "body": body})


app = FastAPI(title="north-micro-vision", version="0.1.0")
app.add_middleware(Guard, max_body_bytes=MAX_BODY_BYTES, api_key=API_KEY)
engine: Engine | None = None


@app.on_event("startup")
def load() -> None:
    global engine
    engine = Engine()


@app.get("/")
def root() -> dict[str, Any]:
    # A browser landing on the bare host should learn what this is and where
    # the real endpoints are, not get a bare 404.
    return {
        "service": "north-micro-vision",
        "model": MODEL_ID,
        "status": "ok" if engine is not None else "loading",
        "device": str(engine.device) if engine is not None else None,
        "endpoints": ["/health", "/v1/models", "/v1/chat/completions"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    if engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_ID, "device": str(engine.device), "device_name": engine.device_name()}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "CohereLabs"}]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest) -> dict[str, Any]:
    if engine is None:
        raise HTTPException(503, "model is still loading")
    if request.stream:
        raise HTTPException(400, "streaming is not supported")
    if request.model and request.model != MODEL_ID:
        log.warning("request names model %r; serving %s", request.model, MODEL_ID)
    stop = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
    max_new = min(request.max_tokens or 4096, MAX_NEW_TOKENS_CAP)
    started = time.monotonic()
    text, prompt_tokens, completion_tokens, finish = engine.generate(
        to_chat(request.messages), max_new, request.temperature, request.top_p,
        request.top_k, request.repetition_penalty, stop,
    )
    elapsed = time.monotonic() - started
    log.info(
        "completion: %d prompt tokens, %d generated in %.1fs (%.1f tok/s), %s",
        prompt_tokens, completion_tokens, elapsed,
        completion_tokens / elapsed if elapsed > 0 else 0.0, finish,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_seconds": round(elapsed, 3),
        },
    }
