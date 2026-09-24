#!/usr/bin/env python3
"""Capture full-raw OpenAI-shaped envelopes from inference-cloud providers.

Env-gated per provider; a missing key is a clean skip with a printed note
(no fixture written)::

    TOGETHER_API_KEY    -> https://api.together.xyz/v1
    CEREBRAS_API_KEY    -> https://api.cerebras.ai/v1
    FIREWORKS_API_KEY   -> https://api.fireworks.ai/inference/v1

Per provider with a key this script runs FOUR rounds:

1. ``GET {base}/models``            (the preset's discovery route),
2. plain chat                       (tiny prompt, max_tokens 32),
3. tool-call round                  (one function tool + tool_choice "auto"),
4. streamed chat                    (full ordered SSE ``data`` payloads,
   ``[DONE]`` included),

and writes ``Tests/fixtures/longtail/together.json`` / ``cerebras.json`` /
``fireworks.json`` with FULL RAW bodies (never normalized).

SECURITY: the Authorization header is built at CALL time from the
environment and is never written to any stored field; ``capture_cmd`` is a
sanitized template in which the credential appears only as the literal
placeholder ``<key>``.

Usage (any python3 >= 3.12, stdlib only)::

    TOGETHER_API_KEY=... python3 Tests/fixtures/longtail/capture_cloud.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent

REQUEST_TIMEOUT_S = 120.0
MAX_STREAM_EVENTS = 5000

PROVIDERS: dict[str, dict[str, Any]] = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env": "TOGETHER_API_KEY",
        "model_preferences": (
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        ),
        "fallback_pattern": "Instruct-Turbo",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env": "CEREBRAS_API_KEY",
        "model_preferences": ("llama3.2-1b", "llama-3.2-1b", "llama3.1-8b"),
        "fallback_pattern": "-1b",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env": "FIREWORKS_API_KEY",
        "model_preferences": (
            "accounts/fireworks/models/qwen2-5-7b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
        ),
        "fallback_pattern": "-instruct",
    },
}

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


def _request(
    url: str, *, method: str, api_key: str, body: bytes | None = None
) -> tuple[int, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}  # built at call time only
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        status = error.code
    except urllib.error.URLError as error:
        return -1, str(error)
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def _stream_data_payloads(url: str, payload: dict[str, Any], *, api_key: str) -> list[str]:
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    payloads: list[str] = []
    data_lines: list[str] = []
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip(" "))
            elif line == "" and data_lines:
                payloads.append("\n".join(data_lines))
                data_lines = []
                if payloads[-1] == "[DONE]" or len(payloads) >= MAX_STREAM_EVENTS:
                    break
    if data_lines:
        payloads.append("\n".join(data_lines))
    return payloads


def _choose_model(models_body: Any, spec: dict[str, Any]) -> str | None:
    ids = [m.get("id", "") for m in (models_body or {}).get("data", []) if isinstance(m, dict)]
    for preference in spec["model_preferences"]:
        if preference in ids:
            return preference
    for model_id in ids:
        if spec["fallback_pattern"] in model_id:
            return model_id
    return ids[0] if ids else None


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


def _sanitized_replay_cmd(base_url: str, model: str) -> str:
    payload = {**PLAIN_PAYLOAD_TAIL, "model": model}
    return (
        f"curl -s {base_url}/chat/completions -H 'Authorization: Bearer <key>' "
        f"-H 'Content-Type: application/json' -d '{json.dumps(payload, separators=(',', ':'))}'"
    )


def capture_provider(name: str, spec: dict[str, Any]) -> bool:
    print(f"[{name}]")
    api_key = os.environ.get(spec["env"], "").strip()
    if not api_key:
        print(f"  SKIP: {spec['env']} not set -- no fixture written (clean skip)")
        return False
    base_url: str = spec["base_url"]

    status, models_body = _request(f"{base_url}/models", method="GET", api_key=api_key)
    if status != 200:
        print(f"  SKIP: GET {base_url}/models failed (HTTP {status}); body: {str(models_body)[:200]}")
        return False
    model = _choose_model(models_body, spec)
    if model is None:
        print("  SKIP: no model ids in /models response")
        return False
    print(f"  base: {base_url}  model: {model}")

    chat_payload = {**PLAIN_PAYLOAD_TAIL, "model": model}
    tool_payload = {**TOOL_PAYLOAD_TAIL, "model": model}

    chat_status, chat_body = _request(
        f"{base_url}/chat/completions", method="POST", api_key=api_key, body=json.dumps(chat_payload).encode()
    )
    if chat_status != 200:
        print(f"  ! plain chat round failed (HTTP {chat_status}); body recorded as-is")
    tool_status, tool_body = _request(
        f"{base_url}/chat/completions", method="POST", api_key=api_key, body=json.dumps(tool_payload).encode()
    )
    if tool_status != 200:
        print(f"  ! tool-call round failed (HTTP {tool_status}); body recorded as-is")

    try:
        events = _stream_data_payloads(f"{base_url}/chat/completions", chat_payload, api_key=api_key)
    except (urllib.error.URLError, OSError) as error:
        print(f"  ! streamed round failed: {error}")
        events = []

    _require_complete_stream(events, server=name)
    fixture = {
        "server": name,
        "base_url": base_url,
        "chat_response": chat_body,
        "tool_call_response": tool_body,
        "stream_events": events,
        "models_response": models_body,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture_cmd": _sanitized_replay_cmd(base_url, model),
    }
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(FIXTURE_DIR.parent.parent.parent)}")
    return True


def main() -> int:
    captured_any = False
    failures: list[str] = []
    for name, spec in PROVIDERS.items():
        try:
            if capture_provider(name, spec):
                captured_any = True
        except RuntimeError as error:  # degraded capture refused at write time
            print(f"  FAIL ({name}): {error}")
            failures.append(name)
    if failures:
        print(f"\nDegraded captures refused (no fixture written): {', '.join(failures)}")
    if not captured_any:
        print("\nNo provider key found; no cloud fixtures written.")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
