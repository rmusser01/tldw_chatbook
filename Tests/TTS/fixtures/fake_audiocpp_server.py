"""Controlled stdlib-only audio.cpp HTTP fixture for real-child tests."""

from __future__ import annotations

import json
import os
import signal
import struct
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def write_executable_wrapper(path: Path) -> Path:
    """Write a direct-exec Python wrapper that preserves the exact child argv."""
    fixture_path = Path(__file__).resolve()
    source = (
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        f"fixture = Path({str(fixture_path)!r})\n"
        "namespace = {'__name__': '__main__', '__file__': str(fixture)}\n"
        "exec(compile(fixture.read_bytes(), str(fixture), 'exec'), namespace)\n"
    )
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)
    return path


def _wav() -> bytes:
    data = b"\x00\x00\x00\x00"
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        1,
        24_000,
        48_000,
        2,
        16,
    )
    payload = b"WAVE" + fmt + struct.pack("<4sI", b"data", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


def _write_chunks(stream: Any, chunks: object) -> None:
    if not isinstance(chunks, list):
        return
    for chunk in chunks:
        if isinstance(chunk, str):
            stream.write(chunk)
            stream.flush()


def _spawn_descriptor_holder(behavior: dict[str, Any]) -> None:
    if not behavior.get("inherit_pipes_descendant") or not hasattr(os, "fork"):
        return
    pid_file_value = behavior.get("descendant_pid_file")
    if not isinstance(pid_file_value, str):
        raise ValueError("descendant_pid_file is required")
    hold_seconds = behavior.get("descendant_hold_seconds", 30.0)
    if not isinstance(hold_seconds, (int, float)) or isinstance(hold_seconds, bool):
        raise ValueError("descendant_hold_seconds is invalid")

    descendant = os.fork()
    if descendant != 0:
        return
    try:
        Path(pid_file_value).write_text(str(os.getpid()), encoding="ascii")
        time.sleep(float(hold_seconds))
    finally:
        os._exit(0)


class _Handler(BaseHTTPRequestHandler):
    server_version = "fake-audiocpp"
    protocol_version = "HTTP/1.0"

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    @property
    def behavior(self) -> dict[str, Any]:
        return self.server.behavior  # type: ignore[attr-defined,no-any-return]

    @property
    def models(self) -> list[dict[str, str]]:
        return self.server.models  # type: ignore[attr-defined,no-any-return]

    def _send_json(self, value: object) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "backend": self.server.backend,  # type: ignore[attr-defined]
                    "models": len(self.models),
                }
            )
            return
        if path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": self.models,
                }
            )
            if self.behavior.get("exit_after_models"):
                self.server.exit_requested = True  # type: ignore[attr-defined]
            return
        if path == "/v1/audio/voices":
            self._send_json({"voices": ["fixture-voice"]})
            return
        if path == "/test/state":
            names = self.behavior.get("observe_environment_names", [])
            observed = {
                name: name in os.environ for name in names if isinstance(name, str)
            }
            self._send_json(
                {
                    "pid": os.getpid(),
                    "argv": sys.argv,
                    "cwd": os.getcwd(),
                    "environment_present": observed,
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if urlsplit(self.path).path != "/v1/audio/speech":
            self.send_error(404)
            return
        length_value = self.headers.get("Content-Length", "0")
        try:
            length = min(max(int(length_value), 0), 1_048_576)
        except ValueError:
            length = 0
        if length:
            self.rfile.read(length)
        payload = _wav()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _read_invocation() -> tuple[Path, dict[str, Any]]:
    if len(sys.argv) != 3 or sys.argv[1] != "--config":
        raise SystemExit(2)
    config_path = Path(sys.argv[2])
    if not config_path.is_absolute():
        raise SystemExit(2)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(2)
    return config_path, document


def _catalog_models(config: dict[str, Any]) -> list[dict[str, str]]:
    configured = config.get("models")
    if configured is None:
        return [
            {
                "id": "fixture-model",
                "object": "model",
                "owned_by": "engine",
                "family": "fixture",
                "task": "tts",
                "mode": "native",
            }
        ]
    if not isinstance(configured, list):
        raise ValueError("models must be a list")
    models: list[dict[str, str]] = []
    for configured_model in configured:
        if not isinstance(configured_model, dict):
            raise ValueError("model must be an object")
        public = {
            key: configured_model.get(key) for key in ("id", "family", "task", "mode")
        }
        if not all(isinstance(value, str) for value in public.values()):
            raise ValueError("model catalog fields must be strings")
        models.append(
            {
                "id": public["id"],  # type: ignore[dict-item]
                "object": "model",
                "owned_by": "engine",
                "family": public["family"],  # type: ignore[dict-item]
                "task": public["task"],  # type: ignore[dict-item]
                "mode": public["mode"],  # type: ignore[dict-item]
            }
        )
    return models


def main() -> int:
    _config_path, config = _read_invocation()
    models = _catalog_models(config)
    behavior_value = config.get("test_behavior", {})
    behavior = behavior_value if isinstance(behavior_value, dict) else {}
    _write_chunks(sys.stdout, behavior.get("stdout_chunks"))
    _write_chunks(sys.stderr, behavior.get("stderr_chunks"))
    if behavior.get("early_exit"):
        exit_code = behavior.get("exit_code", 7)
        return int(exit_code) if isinstance(exit_code, int) else 7

    delay = behavior.get("readiness_delay_seconds", 0.0)
    if isinstance(delay, (int, float)) and not isinstance(delay, bool) and delay > 0:
        time.sleep(float(delay))
    _spawn_descriptor_holder(behavior)

    host = config.get("host")
    port = config.get("port")
    if host != "127.0.0.1" or isinstance(port, bool) or not isinstance(port, int):
        return 2

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    if behavior.get("ignore_terminate"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, request_stop)

    server = HTTPServer((host, port), _Handler)
    server.timeout = 0.05
    server.behavior = behavior  # type: ignore[attr-defined]
    server.models = models  # type: ignore[attr-defined]
    backend = config.get("backend", "cpu")
    server.backend = backend if isinstance(backend, str) else "cpu"  # type: ignore[attr-defined]
    server.exit_requested = False  # type: ignore[attr-defined]
    try:
        while not stopping and not server.exit_requested:  # type: ignore[attr-defined]
            server.handle_request()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
