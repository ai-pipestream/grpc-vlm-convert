"""Unit tests for the request-side guards of server.py.

No model and no GPU: torch and transformers are stubbed before the server
module is imported, so only fastapi and pillow (both in requirements.txt)
are needed:

    python -m unittest test_server        (from this directory)
"""

from __future__ import annotations

import asyncio
import base64
import io
import pathlib
import sys
import types
import unittest
from typing import Any
from unittest import mock


def _stub_model_stack() -> None:
    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: (lambda function: function)
    sys.modules.setdefault("torch", torch)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForImageTextToText = object
    transformers.AutoProcessor = object
    sys.modules.setdefault("transformers", transformers)


_stub_model_stack()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import server  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from PIL import Image  # noqa: E402


def png_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


class DecodeImageTest(unittest.TestCase):
    def status_of(self, url: str) -> int:
        with self.assertRaises(HTTPException) as caught:
            server.decode_image(url)
        return caught.exception.status_code

    def test_data_uri_decodes(self) -> None:
        image = server.decode_image(data_uri(png_bytes(4, 3)))
        self.assertEqual(image.size, (4, 3))

    def test_remote_urls_are_not_fetched_by_default(self) -> None:
        with mock.patch("urllib.request.urlopen") as urlopen:
            for url in ("http://169.254.169.254/latest/meta-data/", "https://example.com/a.png"):
                self.assertEqual(self.status_of(url), 400)
            urlopen.assert_not_called()

    def test_remote_fetch_when_allowed_is_capped(self) -> None:
        class Response:
            def __init__(self) -> None:
                self.asked = None

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def read(self, size: int) -> bytes:
                self.asked = size
                return b"x" * size

        response = Response()
        with mock.patch.object(server, "ALLOW_REMOTE_IMAGES", True), \
                mock.patch.object(server, "MAX_IMAGE_BYTES", 64), \
                mock.patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(self.status_of("http://images.test/big.png"), 413)
        self.assertEqual(response.asked, 65, "reads one byte past the cap, never the whole body")

    def test_bad_base64_and_non_images_are_client_errors(self) -> None:
        self.assertEqual(self.status_of("data:image/png;base64,abc"), 400)
        self.assertEqual(self.status_of(data_uri(b"not an image at all")), 400)
        self.assertEqual(self.status_of("data:image/png,rawbytes"), 400)

    def test_byte_cap(self) -> None:
        with mock.patch.object(server, "MAX_IMAGE_BYTES", 32):
            self.assertEqual(self.status_of(data_uri(png_bytes(64, 64))), 413)

    def test_pixel_cap_is_a_hard_limit(self) -> None:
        # PIL only warns between one and two times MAX_IMAGE_PIXELS; the
        # server makes that range an error too.
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            self.assertEqual(self.status_of(data_uri(png_bytes(12, 12))), 413)  # 144 px
            self.assertEqual(self.status_of(data_uri(png_bytes(20, 20))), 413)  # 400 px
            self.assertEqual(server.decode_image(data_uri(png_bytes(8, 8))).size, (8, 8))


class GuardTest(unittest.TestCase):
    def run_guard(self, path: str, headers: dict[str, str], api_key: str = "",
                  max_body_bytes: int = 1000) -> tuple[int | None, bool]:
        reached = []

        async def inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
            reached.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        sent: list[dict[str, Any]] = []

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        guard = server.Guard(inner, max_body_bytes=max_body_bytes, api_key=api_key)
        scope = {
            "type": "http", "method": "POST", "path": path,
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
        asyncio.run(guard(scope, receive, send))
        status = sent[0]["status"] if sent else None
        return status, bool(reached)

    def test_oversized_body_is_refused_before_it_is_read(self) -> None:
        status, reached = self.run_guard("/v1/chat/completions", {"Content-Length": "1001"})
        self.assertEqual((status, reached), (413, False))
        status, reached = self.run_guard("/v1/chat/completions", {"Content-Length": "1000"})
        self.assertEqual((status, reached), (200, True))

    def test_chunked_bodies_are_refused(self) -> None:
        status, reached = self.run_guard("/v1/chat/completions",
                                         {"Transfer-Encoding": "chunked"})
        self.assertEqual((status, reached), (411, False))

    def test_api_key_guards_the_api_but_not_health(self) -> None:
        key = "north-key-9f3"
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": key}):
            status, reached = self.run_guard("/v1/chat/completions", headers, api_key=key)
            self.assertEqual((status, reached), (401, False), headers)
        status, reached = self.run_guard("/v1/chat/completions",
                                         {"Authorization": "Bearer " + key}, api_key=key)
        self.assertEqual((status, reached), (200, True))
        status, reached = self.run_guard("/health", {}, api_key=key)
        self.assertEqual((status, reached), (200, True))

    def test_no_key_configured_leaves_the_api_open(self) -> None:
        status, reached = self.run_guard("/v1/models", {})
        self.assertEqual((status, reached), (200, True))


class ComposeTest(unittest.TestCase):
    def test_ports_are_published_on_loopback_by_default(self) -> None:
        compose = (pathlib.Path(__file__).resolve().parent / "compose.yaml").read_text()
        ports = [line.strip() for line in compose.splitlines() if line.strip().startswith("ports:")]
        self.assertEqual(len(ports), 3)
        for line in ports:
            self.assertIn('"${NORTH_PUBLISH_ADDRESS:-127.0.0.1}:8086:8080"', line)


if __name__ == "__main__":
    unittest.main()
