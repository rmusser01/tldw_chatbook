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
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

#: Value ``EPHEMERAL_ORIGIN_KEY`` carries on an automatically-injected
#: project-instruction row (``Agents/project_instruction_runtime.py``'s
#: ``_source_row``/``_outcome_row``/``_warning_row``,
#: ``Agents/agent_service.py``'s ``build_project_instruction_row``). Kept as
#: a local literal rather than importing ``PROJECT_INSTRUCTION_ORIGIN`` from
#: either of those modules -- both define the same literal independently and
#: neither is this module's natural dependency.
_PROJECT_INSTRUCTION_ORIGIN = "project_instructions"

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
    status: str  # "complete" | "stopped" | "error" -- the real outcome,
    # even when capture_to_blob's own oversize truncation fires (M13):
    # truncation is a separate `truncated: True` marker inside request/
    # response, never a fourth status value overwriting this one.
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
    # Review finding M1: this gate used to measure the RAW value's length
    # while the hash/size below already measure the CANONICAL
    # (whitespace-stripped) bytes -- so line-wrapping alone could push
    # otherwise-identical content across the threshold in one direction
    # but not the other (canonical len=4088 vs. wrapped len=4141 straddling
    # a 4096 gate), stubbing one variant of the same bytes and not the
    # other. Gate on the same canonical length the hash/size actually use.
    if len("".join(value.split())) < _STUB_MIN_CHARS:
        return value
    match = _DATA_URI_RE.match(value)
    if match:
        # Qodo PR #1883 finding: hash/size the whitespace-stripped payload,
        # not the raw one -- otherwise the same bytes line-wrapped at a
        # different column produce a different sha256/size and the
        # "deterministic stub" promise (identical bytes -> identical stub)
        # breaks for any line-wrapped data URI.
        canonical_data = "".join(match.group("data").split())
        return _stub_for(canonical_data, match.group("mime"))
    if _BASE64_RE.match(value):
        # Review finding M12: `_BASE64_RE` permits embedded whitespace
        # (line-wrapped base64), but `b64decode(..., validate=True)`
        # rejects it outright -- without stripping first, line-wrapped
        # base64 always fails validation and is never stubbed, landing in
        # the blob verbatim (a size/redaction-completeness gap, not a
        # safety one: it is still allowlist-filtered content).
        candidate = "".join(value.split())
        try:
            base64.b64decode(candidate[:4096], validate=True)
        except Exception:
            return value
        # Qodo PR #1883 finding: stub the STRIPPED candidate, not the raw
        # `value` -- otherwise the same underlying bytes wrapped at
        # different line lengths (e.g. 76 cols vs. unwrapped) hash and
        # size differently, breaking `stub_binary_strings`'s documented
        # determinism guarantee and misreporting size in the inspector.
        return _stub_for(candidate, mime_hint or "application/octet-stream")
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


def _redact_project_instruction_rows(messages_payload: Any) -> tuple[Any, tuple[str, ...]]:
    """Replace any project-instruction row's body with a content-free marker.

    C1 (privacy): ``messages_payload`` is on ``CAPTURE_REQUEST_ALLOWLIST``
    verbatim, but a row the ephemeral-injection path tagged with
    ``EPHEMERAL_ORIGIN_KEY == "project_instructions"`` (the automatically
    loaded AGENTS.md/CLAUDE.md instruction rider -- ``Agents/
    project_instruction_runtime.py``'s ``_source_row``/``_outcome_row``/
    ``_warning_row``, ``Agents/agent_service.py``'s
    ``build_project_instruction_row``) is never something the user typed,
    and the shipped user guide (``Docs/User_Guide/console/context-and-
    rag.md``) promises its exact body appears ONLY in the Next Send tab's
    disposable preview -- never in an export, a display, or at rest.

    This is the ONE seam that redaction has to sit at to keep that promise:
    ``build_request_capture``'s output is what gets displayed on the
    Exchange tab, exported via Copy JSON/Save to File, AND persisted to
    ``message_exchanges`` -- filtering only the two export methods would
    still leave the body sitting in the DB and on screen.

    The row's ``role`` and its origin tag both survive unchanged -- only
    ``content`` is replaced -- so the Inspector can still show that such a
    row was sent, and roughly how large it was, rather than a permanently
    empty-looking gap.
    """
    if not isinstance(messages_payload, list):
        return messages_payload, ()
    redacted_paths: list[str] = []
    rows: list[Any] = []
    for index, row in enumerate(messages_payload):
        if not (
            isinstance(row, Mapping)
            and row.get(EPHEMERAL_ORIGIN_KEY) == _PROJECT_INSTRUCTION_ORIGIN
        ):
            rows.append(row)
            continue
        content = row.get("content")
        char_count = len(content) if isinstance(content, str) else 0
        new_row = dict(row)
        new_row["content"] = (
            f"[project instruction body omitted by capture policy -- "
            f"{char_count} chars]"
        )
        rows.append(new_row)
        redacted_paths.append(f"messages_payload[{index}].content")
    return rows, tuple(redacted_paths)


def build_request_capture(kwargs: Mapping[str, Any]) -> tuple[dict, tuple[str, ...]]:
    """Return (allowlisted+stubbed request dict, names of dropped keys).

    ``omitted_keys`` doubles as the redaction-visibility signal (C1): when
    ``messages_payload`` contains a project-instruction row, its
    ``messages_payload[<index>].content`` path is folded into this same
    tuple alongside genuinely dropped top-level keys (e.g. ``api_key``) --
    the Inspector already renders this tuple verbatim as an "Omitted by
    capture policy" line, so a viewer sees the withholding without any new
    UI surface.
    """
    request: dict = {}
    omitted: list[str] = []
    for key, value in kwargs.items():
        if key in CAPTURE_REQUEST_ALLOWLIST:
            if key == "messages_payload":
                value, redacted_paths = _redact_project_instruction_rows(value)
                omitted.extend(redacted_paths)
            request[key] = stub_binary_strings(_jsonable(value))
        else:
            omitted.append(str(key))
    return request, tuple(sorted(omitted))


def capture_to_blob(capture: ExchangeCapture) -> bytes:
    """zlib-compressed JSON; oversize captures truncate, never fail.

    Review finding M13: the oversize branch used to overwrite ``status``
    with ``"truncated"``, discarding whether the call had actually
    completed/stopped/errored. The real outcome is preserved; truncation
    is marked separately via a ``truncated: True`` key in the (now
    stubbed) request/response dicts.
    """
    blob = zlib.compress(json.dumps(asdict(capture), default=str).encode("utf-8"))
    if len(blob) <= EXCHANGE_BLOB_MAX_BYTES:
        return blob
    truncated = replace(
        capture,
        request={
            "truncated": True,
            "reason": f"capture exceeded {EXCHANGE_BLOB_MAX_BYTES} bytes compressed",
        },
        response={"truncated": True},
    )
    return zlib.compress(json.dumps(asdict(truncated), default=str).encode("utf-8"))


def capture_from_blob(blob: bytes) -> ExchangeCapture:
    """Inverse of :func:`capture_to_blob`.

    Review finding M11: filters to ``ExchangeCapture``'s own known field
    names before construction -- a future blob written by a newer version
    with an extra field would otherwise raise ``TypeError`` here today,
    on every OLDER build reading it back.
    """
    data = json.loads(zlib.decompress(blob))
    data["omitted_keys"] = tuple(data.get("omitted_keys") or ())
    known_fields = {f.name for f in fields(ExchangeCapture)}
    filtered = {key: value for key, value in data.items() if key in known_fields}
    return ExchangeCapture(**filtered)
