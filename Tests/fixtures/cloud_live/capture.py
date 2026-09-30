#!/usr/bin/env python3
"""Capture raw responses from every engine preset's real API (TASK-33640).

Registry-driven: every engine-driven record in ``tldw_chatbook.provider_registry``
(except the ``custom-hosted`` execution key) is a candidate, and only those
whose API key is available are captured -- the rest skip cleanly. Per
provider, up to four rounds, each built the way the engine builds it (base
URL + suffix, ``api-key`` vs Bearer auth, ``max_tokens`` spelling, extra body
fields, ``stream_options`` when the record asks, ``tool_choice`` only where
the record allows it):

1. ``GET {base}/models``      (only when the record has a discovery route),
2. plain chat                  (tiny prompt),
3. tool-call round             (one function tool; skipped when the record
                                ships native tools off),
4. streamed chat               (every SSE ``data`` payload, ``[DONE]`` included).

Each capture is written RAW to ``Tests/fixtures/cloud_live/<key>.json``; the
offline replay test (``Tests/LLM_Calls/test_live_capture_replay.py``) then
runs it through the real engine. After each capture this script also prints
the response KEY NAMES the record's allowances do not cover -- never content.

Keys: taken from the environment, or from a user-owned keys file (default
``~/.config/tldw-live/keys.env``; ``NAME=value`` lines, ``export`` and quotes
allowed; the environment wins). A key is only ever placed in the request
header at call time: it is never printed, logged, or written to a fixture.

Per-provider environment overrides (``<KEY>`` is the record key upper-cased):

- ``TLDW_LIVE_<KEY>_MODEL``     the model to use (required for Azure, where
                                 it is your deployment name),
- ``TLDW_LIVE_<KEY>_BASE_URL``  the base URL (required for Azure, Cloudflare
                                 and Databricks, whose URL is per account),
- ``TLDW_LIVE_<KEY>_<SETTING>`` a config-sourced header value (e.g.
                                 ``TLDW_LIVE_WANDB_PROJECT``,
                                 ``TLDW_LIVE_CLOUDFLARE_GATEWAY_ID``).

Usage (repo root)::

    .venv/bin/python Tests/fixtures/cloud_live/capture.py            # every available key
    .venv/bin/python Tests/fixtures/cloud_live/capture.py together nvidia
    .venv/bin/python Tests/fixtures/cloud_live/capture.py --list     # who would run, and why not
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tldw_chatbook.provider_registry import ALL_RECORDS, ProviderRecord  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent
DEFAULT_KEYS_FILE = Path.home() / ".config" / "tldw-live" / "keys.env"
MAX_STREAM_EVENTS = 5000
MAX_LISTED_MODELS = 20
MAX_TOKENS = 200
PER_ACCOUNT = "<per-account>"

# Closed shapes of the engine's strict parser (hosted_chat.py); anything else
# must be in the record's allowances.
KNOWN_TOP = frozenset({"id", "object", "created", "model", "system_fingerprint", "choices", "usage"})
KNOWN_CHOICE = frozenset({"index", "message", "finish_reason"})
KNOWN_STREAM_CHOICE = frozenset({"index", "delta", "finish_reason", "usage"})
KNOWN_MESSAGE = frozenset({"role", "content", "reasoning_content", "tool_calls"})

PLAIN_MESSAGES = [{"role": "user", "content": "Say ok."}]
TOOL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Tokyo? Use the get_weather tool."}
]
TOOL = {
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
# Model ids that are clearly not chat models, skipped when choosing from a listing.
_NON_CHAT = re.compile(
    r"embed|rerank|whisper|tts|speech|audio|image|flux|sdxl|diffusion|moderation|guard|ocr|clip",
    re.IGNORECASE,
)
_CHEAP = re.compile(r"mini|flash|small|lite|nano|haiku|turbo|[^0-9](1|3|7|8)b", re.IGNORECASE)


def engine_presets() -> list[ProviderRecord]:
    """Every engine preset a user can configure, in registry order."""
    return [r for r in ALL_RECORDS if r.engine_driven and r.key != "custom-hosted"]


def read_keys_file(path: Path) -> dict[str, str]:
    """Parse ``NAME=value`` lines; never echoes values.

    Args:
        path: The keys file; a missing file yields no keys.

    Returns:
        Names to values, excluding blanks and comments.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            values[name.strip()] = value
    return values


def _env_name(record: ProviderRecord, suffix: str) -> str:
    return f"TLDW_LIVE_{record.key.upper().replace('-', '_')}_{suffix}"


def _config_seeds(record: ProviderRecord) -> list[str]:
    """The ``[providers]`` seed list shipped in config.py, read as text.

    Importing ``tldw_chatbook.config`` would load the real app configuration,
    so the seed line is parsed from the source instead.
    """
    source = (REPO_ROOT / "tldw_chatbook" / "config.py").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(record.config_key)} = (\[.*\])", source, re.MULTILINE)
    if not match:
        return []
    try:
        seeds = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    return [seed for seed in seeds if isinstance(seed, str)]


class Target:
    """One provider's resolved capture settings (the key is never exposed)."""

    def __init__(self, record: ProviderRecord, env: dict[str, str]) -> None:
        self.record = record
        self._api_key = next(
            (env[name] for name in record.api_key_env_candidates if env.get(name, "").strip()), ""
        )
        override = env.get(_env_name(record, "BASE_URL"), "").strip()
        base = override or (record.default_base_url or "")
        if base and record.base_url_suffix and urllib.parse.urlsplit(base).path in ("", "/"):
            base = base.rstrip("/") + record.base_url_suffix
        self.base_url = base.rstrip("/")
        self.base_url_display = PER_ACCOUNT if override else self.base_url
        self.model_override = env.get(_env_name(record, "MODEL"), "").strip()
        self.headers_extra = {
            header: env[_env_name(record, setting.upper())].strip()
            for header, setting in record.config_headers.items()
            if env.get(_env_name(record, setting.upper()), "").strip()
        }
        timeout = record.settings_defaults.get("timeout", 120)
        self.timeout = float(timeout) if isinstance(timeout, (int, float)) else 120.0

    def skip_reason(self) -> str | None:
        """Why this provider cannot be captured, or None when it can."""
        if not self._api_key:
            return f"no key ({' / '.join(self.record.api_key_env_candidates)})"
        if not self.base_url:
            return f"per-account URL: set {_env_name(self.record, 'BASE_URL')}"
        if self.record.key == "azure" and not self.model_override:
            return f"deployment name: set {_env_name(self.record, 'MODEL')}"
        return None

    def headers(self) -> dict[str, str]:
        """Request headers, credential included, built at call time only."""
        auth = (
            {"api-key": self._api_key}
            if self.record.auth_scheme == "api_key_header"
            else {"Authorization": f"Bearer {self._api_key}"}
        )
        return {**auth, "Content-Type": "application/json", **self.headers_extra}

    def payload(self, model: str, messages: list[dict[str, Any]], *, stream: bool, tools: bool) -> dict[str, Any]:
        """A request body shaped the way the engine shapes it for this record."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            self.record.max_tokens_key or "max_tokens": MAX_TOKENS,
        }
        if stream and self.record.stream_include_usage:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = [TOOL]
            if "tool_choice" in self.record.payload_flags:
                body["tool_choice"] = "auto"
        body.update(self.record.extra_body_fields)
        return body


def _request(target: Target, url: str, body: dict[str, Any] | None) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, method="POST" if body is not None else "GET", headers=target.headers()
    )
    try:
        with urllib.request.urlopen(request, timeout=target.timeout) as response:
            raw, status = response.read().decode("utf-8", errors="replace"), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read().decode("utf-8", errors="replace"), error.code
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return -1, f"{type(error).__name__}"
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw[:2000]


def _stream(target: Target, url: str, body: dict[str, Any]) -> tuple[int, list[str]]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST", headers=target.headers()
    )
    payloads: list[str] = []
    data_lines: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=target.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip(" "))
                elif line == "" and data_lines:
                    payloads.append("\n".join(data_lines))
                    data_lines = []
                    if payloads[-1] == "[DONE]" or len(payloads) >= MAX_STREAM_EVENTS:
                        break
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, [error.read().decode("utf-8", errors="replace")[:2000]]
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return -1, [type(error).__name__]
    if data_lines:
        payloads.append("\n".join(data_lines))
    return status, payloads


def choose_model(target: Target, listing: Any) -> str | None:
    """Override, else a cheap-looking chat model from the listing, else the first seed."""
    if target.model_override:
        return target.model_override
    ids = [
        entry.get("id")
        for entry in (listing.get("data") if isinstance(listing, dict) else None) or []
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]
    chat_ids = [model for model in ids if not _NON_CHAT.search(model)]
    for model in chat_ids:
        if _CHEAP.search(model):
            return model
    if chat_ids:
        return chat_ids[0]
    seeds = _config_seeds(target.record)
    return seeds[0] if seeds else None


def uncovered_keys(record: ProviderRecord, fixture: dict[str, Any]) -> dict[str, list[str]]:
    """Response key NAMES outside the strict shape AND the record's allowances."""
    top: set[str] = set()
    choice: set[str] = set()
    message: set[str] = set()
    for field in ("chat_response", "tool_call_response"):
        body = fixture.get(field)
        if not isinstance(body, dict) or "choices" not in body:
            continue
        top |= set(body) - KNOWN_TOP
        for item in body.get("choices") or []:
            if isinstance(item, dict):
                choice |= set(item) - KNOWN_CHOICE
                if isinstance(item.get("message"), dict):
                    message |= set(item["message"]) - KNOWN_MESSAGE
    for payload in fixture.get("stream_events") or []:
        try:
            event = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        top |= set(event) - KNOWN_TOP
        for item in event.get("choices") or []:
            if isinstance(item, dict):
                choice |= set(item) - KNOWN_STREAM_CHOICE
                if isinstance(item.get("delta"), dict):
                    message |= set(item["delta"]) - KNOWN_MESSAGE
    return {
        "top": sorted(top - record.response_allowances),
        "choice": sorted(choice - record.choice_allowances),
        "message": sorted(message - record.message_allowances),
    }


def capture(target: Target) -> Path | None:
    """Run the rounds for one provider and write its raw fixture."""
    record = target.record
    listing: Any = None
    if record.discovery_route is not None and not target.model_override:
        status, listing = _request(target, f"{target.base_url}/{record.discovery_route}", None)
        if status != 200:
            print(f"  ! GET /{record.discovery_route} -> HTTP {status}")
    model = choose_model(target, listing)
    if not model:
        print(f"  SKIP: no model (set {_env_name(record, 'MODEL')})")
        return None
    print(f"  model: {model}")
    url = f"{target.base_url}/chat/completions"
    chat_status, chat_body = _request(target, url, target.payload(model, PLAIN_MESSAGES, stream=False, tools=False))
    tool_status, tool_body = (None, None)
    if record.native_tools:
        tool_status, tool_body = _request(target, url, target.payload(model, TOOL_MESSAGES, stream=False, tools=True))
    stream_status, events = _stream(target, url, target.payload(model, PLAIN_MESSAGES, stream=True, tools=False))
    print(f"  plain HTTP {chat_status}  tool HTTP {tool_status}  stream HTTP {stream_status}"
          f" ({len(events)} events, ends [DONE]: {bool(events) and events[-1] == '[DONE]'})")
    listed = (listing or {}).get("data") if isinstance(listing, dict) else None
    fixture = {
        "server": record.key,
        "base_url": target.base_url_display,
        "model": model,
        "statuses": {"plain": chat_status, "tool": tool_status, "stream": stream_status},
        "chat_response": chat_body,
        "tool_call_response": tool_body,
        "stream_events": events,
        "models_response": (
            None if listed is None
            else {"count": len(listed), "sample": listed[:MAX_LISTED_MODELS]}
        ),
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = FIXTURE_DIR / f"{record.key}.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gaps = {level: keys for level, keys in uncovered_keys(record, fixture).items() if keys}
    print(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"  uncovered keys: {gaps or 'none'}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("providers", nargs="*", help="record keys to capture (default: every available)")
    parser.add_argument("--keys-file", type=Path, default=DEFAULT_KEYS_FILE)
    parser.add_argument("--list", action="store_true", help="show who would run, and why not")
    args = parser.parse_args(argv)
    env = {**read_keys_file(args.keys_file), **os.environ}
    records = engine_presets()
    unknown = set(args.providers) - {r.key for r in records}
    if unknown:
        parser.error(f"unknown provider(s): {', '.join(sorted(unknown))}")
    targets = [Target(r, env) for r in records if not args.providers or r.key in args.providers]
    captured = 0
    for target in targets:
        reason = target.skip_reason()
        if args.list or reason:
            print(f"[{target.record.key}] {'ready' if reason is None else 'skip: ' + reason}")
            continue
        print(f"[{target.record.key}]")
        captured += capture(target) is not None
    if not args.list:
        print(f"captured {captured} of {len(targets)} provider(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
