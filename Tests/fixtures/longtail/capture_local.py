#!/usr/bin/env python3
"""Capture full-raw OpenAI-shaped envelopes from locally hosted servers.

Per available server (llama-server, Ollama; vLLM only with a demonstrable
CUDA runtime) this script starts the server, runs THREE capture rounds,

1. plain chat            (tiny prompt, max_tokens 32),
2. tool-call round       (one function tool + tool_choice "auto"),
3. streamed chat         (full ordered SSE ``data`` payloads, ``[DONE]``
   included),

records the FULL RAW response bodies (parsed JSON objects -- never
normalized, nothing dropped), stops the server again, and writes one
fixture per server next to this script: ``<server>.json``.

Fixture schema::

    {"server", "base_url", "chat_response", "tool_call_response",
     "stream_events", "models_response", "captured_at", "capture_cmd"}

``capture_cmd`` is a sanitized replay template; local servers need no
credential, so no key ever appears.  Servers whose binaries or models are
absent are skipped with a printed note (their fixture is simply not
written); ``CAPTURE.md`` records the skips.

Usage (any python3 >= 3.12; stdlib only except the LLAMA_GGUF override,
which is routed through the repo's ``tldw_chatbook.Utils.path_validation``
and refused when that import is unavailable)::

    python3 Tests/fixtures/longtail/capture_local.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent

# Distinct loopback ports so a capture never collides with a developer's
# own running instance of the same server (which we leave untouched).
LLAMA_PORT = 8901
OLLAMA_PORT = 8902
VLLM_PORT = 8903

STARTUP_TIMEOUT_S = 180.0
REQUEST_TIMEOUT_S = 300.0
MAX_STREAM_EVENTS = 5000

PLAIN_PAYLOAD_TAIL: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Say ok."}],
    "max_tokens": 32,
}
TOOL_PAYLOAD_TAIL: dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "What is the weather in Tokyo? Use the get_weather tool."}
    ],
    "max_tokens": 64,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; credentials, when a caller needs them, are
# built into headers at CALL time and never persisted).
# ---------------------------------------------------------------------------


def _post_json(
    url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None
) -> tuple[int, Any]:
    """POST payload, return (status, parsed-or-raw body)."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:  # record error bodies too -- they are evidence
        raw = error.read().decode("utf-8", errors="replace")
        status = error.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def _get_json(url: str, *, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(url, method="GET", headers={**(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        status = error.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def _post_stream_data_payloads(
    url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None
) -> list[str]:
    """Stream a chat completion, return the ordered SSE ``data`` payloads.

    Each payload is stored exactly as received after the ``data: `` prefix,
    including the terminal ``[DONE]`` sentinel.
    """
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json", **(headers or {})}
    )
    payloads: list[str] = []
    data_lines: list[str] = []
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip(" "))
            elif line == "" and data_lines:
                # Blank line terminates one SSE record per the wire spec.
                payloads.append("\n".join(data_lines))
                data_lines = []
                if len(payloads) >= MAX_STREAM_EVENTS:
                    break
                if payloads[-1] == "[DONE]":
                    break
    if data_lines:  # server hung up without a trailing blank line
        payloads.append("\n".join(data_lines))
    return payloads


def _wait_for_http_ok(url: str, *, accept: Any = None, timeout_s: float = STARTUP_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status, body = _get_json(url)
            if 200 <= status < 300:
                if accept is None or accept(body):
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


# ---------------------------------------------------------------------------
# Capture rounds (shared by every server).
# ---------------------------------------------------------------------------


def _capture_rounds(base_url: str, *, model: str | None) -> tuple[Any, Any, list[str]]:
    """Run plain / tools / streamed rounds; return their raw captures."""
    chat_payload = dict(PLAIN_PAYLOAD_TAIL)
    tool_payload = dict(TOOL_PAYLOAD_TAIL)
    if model is not None:
        chat_payload["model"] = model
        tool_payload["model"] = model

    chat_status, chat_body = _post_json(f"{base_url}/v1/chat/completions", chat_payload)
    if chat_status != 200:
        print(f"  ! plain chat round failed (HTTP {chat_status}); body recorded as-is")

    tool_status, tool_body = _post_json(f"{base_url}/v1/chat/completions", tool_payload)
    if tool_status != 200:
        print(f"  ! tool-call round failed (HTTP {tool_status}); body recorded as-is")

    try:
        events = _post_stream_data_payloads(f"{base_url}/v1/chat/completions", chat_payload)
    except (urllib.error.URLError, OSError) as error:
        print(f"  ! streamed round failed: {error}")
        events = []

    return chat_body, tool_body, events


def _sanitized_replay_cmd(base_url: str, *, model: str | None) -> str:
    payload = dict(PLAIN_PAYLOAD_TAIL)
    if model is not None:
        payload["model"] = model
    return (
        f"curl -s {base_url}/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d '{json.dumps(payload, separators=(',', ':'))}'"
    )


def _require_complete_stream(events: list[str], *, server: str) -> None:
    """Refuse to write a degraded fixture: a failed streamed round that
    silently yields no (or unterminated) events would otherwise rot the
    stream evidence for servers whose pinned inventory is empty."""
    if not events:
        raise RuntimeError(
            f"{server}: streamed round produced no SSE events -- refusing to write a degraded fixture"
        )
    if events[-1] != "[DONE]":
        raise RuntimeError(
            f"{server}: streamed round did not terminate in [DONE] -- refusing to write a degraded fixture"
        )


def _write_fixture(
    server: str,
    base_url: str,
    chat_body: Any,
    tool_body: Any,
    events: list[str],
    models_body: Any,
    replay_cmd: str,
) -> Path:
    _require_complete_stream(events, server=server)
    fixture = {
        "server": server,
        "base_url": base_url,
        "chat_response": chat_body,
        "tool_call_response": tool_body,
        "stream_events": events,
        "models_response": models_body,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture_cmd": replay_cmd,
    }
    path = FIXTURE_DIR / f"{server}.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _version_of(binary: str, *args: str) -> str:
    try:
        done = subprocess.run(
            [binary, *args], capture_output=True, text=True, timeout=60, check=False
        )
        first = (done.stdout or done.stderr).strip().splitlines()
        return first[0].strip() if first else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------------------
# llama-server (llama.cpp; also stands in for the LM Studio server family,
# which is GUI-only and unscripted).
# ---------------------------------------------------------------------------


def _gguf_allowed_roots() -> tuple[Path, ...]:
    """Return the default allowed roots for an LLAMA_GGUF override.

    The override is untrusted environment input feeding a server process,
    so it is confined (read-mode) to the user's home model directories --
    anywhere else is refused rather than read.
    """
    home = Path.home()
    return (
        home / ".ollama" / "models",
        home / ".lmstudio" / "models",
        home / ".cache" / "llama.cpp",
        home / "models",
    )


def _llama_gguf_override(override: str) -> Path | None:
    """Resolve the LLAMA_GGUF override inside one allowed root, or refuse.

    Routes the raw environment value through the repo's shared path
    validation (read-mode containment: symlink-resolved, traversal-proof)
    against the allowed model roots; only a returned, validated, existing
    file is ever used.
    """
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from tldw_chatbook.Utils.path_validation import validate_path
    except Exception:
        print(
            "  NOTE: LLAMA_GGUF override refused: the repo's path "
            "validation could not be imported."
        )
        return None
    for root in _gguf_allowed_roots():
        try:
            candidate = validate_path(override, root, allow_hidden=True)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _find_gguf() -> Path | None:
    """Locate a small gguf to serve.

    Explicit override first (validated against the allowed model roots --
    traversal or out-of-root values are refused, never read), then any
    ollama blob (llama.cpp loads ollama's gguf blobs directly), largest
    first (weights, not config shards).
    """
    override = os.environ.get("LLAMA_GGUF")
    if override:
        resolved = _llama_gguf_override(override)
        if resolved is not None:
            return resolved
        print(
            "  NOTE: LLAMA_GGUF override refused: it must be an existing "
            "file inside one of the home model directories "
            f"({', '.join(str(root) for root in _gguf_allowed_roots())})."
        )
    blobs_dir = Path.home() / ".ollama" / "models" / "blobs"
    if blobs_dir.is_dir():
        candidates = sorted(blobs_dir.glob("sha256-*"), key=lambda p: p.stat().st_size, reverse=True)
        for candidate in candidates[:5]:
            if candidate.stat().st_size >= 50_000_000:  # a model, not a config shard
                return candidate
    return None


def capture_llama_server() -> bool:
    print("[llama-server]")
    binary = shutil.which("llama-server")
    if binary is None:
        print("  SKIP: llama-server binary not found on PATH")
        return False
    gguf = _find_gguf()
    if gguf is None:
        print("  SKIP: no gguf model found (set LLAMA_GGUF=<path> or pull an ollama model)")
        return False
    if not _port_free(LLAMA_PORT):
        print(f"  SKIP: port {LLAMA_PORT} already in use")
        return False
    print(f"  version: {_version_of(binary, '--version')}  model: {gguf.name}")
    server = subprocess.Popen(
        [binary, "-m", str(gguf), "--port", str(LLAMA_PORT), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        base_url = f"http://127.0.0.1:{LLAMA_PORT}"
        if not _wait_for_http_ok(f"{base_url}/health", accept=lambda b: b == {"status": "ok"}):
            print("  SKIP: llama-server did not become healthy in time")
            return False
        chat_body, tool_body, events = _capture_rounds(base_url, model=None)
        path = _write_fixture(
            "llama-server",
            base_url,
            chat_body,
            tool_body,
            events,
            None,  # models GET is a cloud-capture round only
            _sanitized_replay_cmd(base_url, model=None),
        )
        print(f"  wrote {path.relative_to(FIXTURE_DIR.parent.parent.parent)}")
        return True
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


# ---------------------------------------------------------------------------
# Ollama.
# ---------------------------------------------------------------------------

OLLAMA_MODEL_PREFERENCES = ("qwen2.5:0.5b", "qwen3:0.6b", "smollm2:360m", "llama3.2:1b")


def _ollama_binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    brew = Path("/opt/homebrew/opt/ollama/bin/ollama")
    return str(brew) if brew.is_file() else None


def capture_ollama() -> bool:
    print("[ollama]")
    binary = _ollama_binary()
    if binary is None:
        print("  SKIP: ollama binary not found")
        return False
    if not _port_free(OLLAMA_PORT):
        print(f"  SKIP: port {OLLAMA_PORT} already in use")
        return False
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}"}
    server = subprocess.Popen(
        [binary, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    try:
        base_url = f"http://127.0.0.1:{OLLAMA_PORT}"
        if not _wait_for_http_ok(f"{base_url}/api/version"):
            print("  SKIP: ollama serve did not become healthy in time")
            return False
        _, version_body = _get_json(f"{base_url}/api/version")
        print(f"  version: {version_body}")

        _, tags = _get_json(f"{base_url}/api/tags")
        present = {m.get("name", "") for m in (tags or {}).get("models", [])}
        model = next((m for m in OLLAMA_MODEL_PREFERENCES if m in present), None)
        if model is None:
            model = OLLAMA_MODEL_PREFERENCES[0]
            print(f"  pulling {model} ...")
            done = subprocess.run(
                [binary, "pull", model], capture_output=True, text=True, env=env, timeout=1800, check=False
            )
            if done.returncode != 0:
                print(f"  SKIP: ollama pull {model} failed: {(done.stderr or done.stdout).strip()[:200]}")
                return False

        chat_body, tool_body, events = _capture_rounds(base_url, model=model)
        path = _write_fixture(
            "ollama",
            base_url,
            chat_body,
            tool_body,
            events,
            None,  # models GET is a cloud-capture round only
            _sanitized_replay_cmd(base_url, model=model),
        )
        print(f"  wrote {path.relative_to(FIXTURE_DIR.parent.parent.parent)}")
        return True
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


# ---------------------------------------------------------------------------
# vLLM -- attempted ONLY when a CUDA runtime demonstrably exists.
# ---------------------------------------------------------------------------


def _cuda_runtime_exists() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    done = subprocess.run([nvidia_smi], capture_output=True, timeout=60, check=False)
    return done.returncode == 0


def capture_vllm() -> bool:
    print("[vllm]")
    if not _cuda_runtime_exists():
        print(
            "  SKIP: no CUDA runtime (nvidia-smi absent/failed); the vllm/vllm-openai "
            "image is CUDA-only, so no capture is attempted on this machine. "
            "vLLM-specific keys (stop_reason) remain UNVERIFIED MEMORY, NOT EVIDENCE "
            "until a Linux capture lands."
        )
        return False
    docker = shutil.which("docker")
    if docker is None:
        print("  SKIP: CUDA present but docker CLI not found")
        return False
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    container = "tldw-fixture-vllm"
    run = subprocess.run(
        [
            docker,
            "run",
            "-d",
            "--rm",
            "--gpus",
            "all",
            "-p",
            f"{VLLM_PORT}:8000",
            "--name",
            container,
            "-e",
            f"VLLM_MODEL={model}",
            "vllm/vllm-openai:latest",
            "--model",
            model,
            "--max-model-len",
            "2048",
        ],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT_S,
        check=False,
    )
    if run.returncode != 0:
        print(f"  SKIP: docker run failed: {(run.stderr or '').strip()[:200]}")
        return False
    try:
        base_url = f"http://127.0.0.1:{VLLM_PORT}"
        if not _wait_for_http_ok(f"{base_url}/health"):
            print("  SKIP: vLLM container did not become healthy in time")
            return False
        chat_body, tool_body, events = _capture_rounds(base_url, model=model)
        path = _write_fixture(
            "vllm",
            base_url,
            chat_body,
            tool_body,
            events,
            None,  # models GET is a cloud-capture round only
            _sanitized_replay_cmd(base_url, model=model),
        )
        print(f"  wrote {path.relative_to(FIXTURE_DIR.parent.parent.parent)}")
        return True
    finally:
        subprocess.run(
            [docker, "rm", "-f", container], capture_output=True, timeout=60, check=False
        )


def main() -> int:
    captured_any = False
    failures: list[str] = []
    captures = (
        ("llama-server", capture_llama_server),
        ("ollama", capture_ollama),
        ("vllm", capture_vllm),
    )
    for name, capture in captures:
        try:
            if capture():
                captured_any = True
        except RuntimeError as error:  # degraded capture refused at write time
            print(f"  FAIL ({name}): {error}")
            failures.append(name)
    if failures:
        print(f"\nDegraded captures refused (no fixture written): {', '.join(failures)}")
    if not captured_any:
        print("\nNo local server captured; no fixtures written.")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
