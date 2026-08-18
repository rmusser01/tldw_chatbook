"""Pure exchange-capture records for the Console Conversation Inspector.

No I/O here: builders take the gateway's ``chat_api_call`` kwargs and
produce allowlisted, binary-stubbed, blob-serializable records. The
allowlist is the contract: a kwarg key not named below NEVER persists
(spec: Docs/superpowers/specs/2026-08-18-console-conversation-inspector-design.md).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

CAPTURE_REQUEST_ALLOWLIST: frozenset[str] = frozenset({
    "api_endpoint", "api_base_url", "system_message", "messages_payload",
    "tools", "model", "streaming", "temp", "topp", "maxp", "topk", "minp",
    "max_tokens", "seed", "presence_penalty", "frequency_penalty",
    "reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort",
    "thinking_budget_tokens", "prompt_caching", "response_format",
    "api_mode", "request_timeout", "request_retries", "request_retry_delay",
    "provider_continuations",
})
# Deliberately OFF the allowlist: "api_key" (credential) and
# "api_key_resolved" (credential-adjacent marker) — they surface in
# omitted_keys instead.

#: Strings at/above this length are candidates for base64 stubbing.
_STUB_MIN_CHARS = 4096
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<data>.+)$", re.DOTALL)

EXCHANGE_BLOB_MAX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ExchangeCapture:
    """One provider call's captured request/response pair."""

    run_tag: str
    seq: int
    created_at: str
    provider: str
    model: str
    endpoint: str | None
    request: dict
    response: dict
    status: str  # "complete" | "stopped" | "error" | "truncated"
    usage_json: str | None  # THIS call's normalized ProviderUsage.to_json()
    omitted_keys: tuple[str, ...]


def _stub_for(data: str, mime: str) -> str:
    digest = hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()[:16]
    approx = (len(data) * 3) // 4
    if approx >= 1024 * 1024:
        size = f"{approx / (1024 * 1024):.1f} MB"
    else:
        size = f"{approx / 1024:.1f} KB"
    return f"[{mime}, {size}, sha256:{digest}]"


def _maybe_stub_string(value: str, mime_hint: str | None = None) -> str:
    if len(value) < _STUB_MIN_CHARS:
        return value
    match = _DATA_URI_RE.match(value)
    if match:
        return _stub_for(match.group("data"), match.group("mime"))
    if _BASE64_RE.match(value):
        try:
            base64.b64decode(value[:4096], validate=True)
        except Exception:
            return value
        return _stub_for(value, mime_hint or "application/octet-stream")
    return value


def stub_binary_strings(obj: Any) -> Any:
    """Recursively replace base64/data-URI payloads with honest stubs.

    Deterministic: identical input bytes always produce the identical stub
    (size + sha256 prefix), so a viewer can verify attachment identity
    across calls without the bytes themselves.
    """
    if isinstance(obj, str):
        return _maybe_stub_string(obj)
    if isinstance(obj, Mapping):
        mime_hint = obj.get("media_type") or obj.get("mime_type")
        out = {}
        for key, value in obj.items():
            if key in {"data", "b64_json"} and isinstance(value, str):
                out[key] = _maybe_stub_string(value, mime_hint if isinstance(mime_hint, str) else None)
            else:
                out[key] = stub_binary_strings(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [stub_binary_strings(item) for item in obj]
    return obj


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return json.loads(json.dumps(obj, default=str))


def build_request_capture(kwargs: Mapping[str, Any]) -> tuple[dict, tuple[str, ...]]:
    """Return (allowlisted+stubbed request dict, names of dropped keys)."""
    request: dict = {}
    omitted: list[str] = []
    for key, value in kwargs.items():
        if key in CAPTURE_REQUEST_ALLOWLIST:
            request[key] = stub_binary_strings(_jsonable(value))
        else:
            omitted.append(str(key))
    return request, tuple(sorted(omitted))


def capture_to_blob(capture: ExchangeCapture) -> bytes:
    """zlib-compressed JSON; oversize captures truncate, never fail."""
    blob = zlib.compress(json.dumps(asdict(capture), default=str).encode("utf-8"))
    if len(blob) <= EXCHANGE_BLOB_MAX_BYTES:
        return blob
    truncated = replace(
        capture,
        status="truncated",
        request={"truncated": f"capture exceeded {EXCHANGE_BLOB_MAX_BYTES} bytes compressed"},
        response={"truncated": True},
    )
    return zlib.compress(json.dumps(asdict(truncated), default=str).encode("utf-8"))


def capture_from_blob(blob: bytes) -> ExchangeCapture:
    data = json.loads(zlib.decompress(blob))
    data["omitted_keys"] = tuple(data.get("omitted_keys") or ())
    return ExchangeCapture(**data)
