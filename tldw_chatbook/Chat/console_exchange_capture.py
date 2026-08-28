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
from enum import Enum
from typing import Any, Mapping, Sequence

from tldw_chatbook.Chat.console_project_instructions import (
    EPHEMERAL_ORIGIN_KEY,
    canonical_provider_endpoint_identity,
)

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
#
# Privacy stance for what IS on the list (task-23026): capture exists to
# answer "what did we actually hand the provider adapter on this call", so
# the conversation content is the subject, not an accident — it is retained
# by design, under the user-controllable Safe/Full detail policy rather
# than the allowlist. Under Safe (the default) retention is BOUNDED: the
# newest ``CAPTURE_SAFE_HISTORY_TAIL_ROWS`` payload rows (this turn's delta
# plus immediate context) persist verbatim, and every older history row is
# replaced in place by a content-free fingerprint (role, origin tag, char
# count, sha256 prefix) — the bodies of those rows are already durably
# stored once in the ``messages`` table and once in the earlier capture
# whose tail they were new in, so a per-turn re-copy stored nothing new
# and grew O(n²) (measured 21.33 MB for one 200-turn conversation).
# Full — explicit, consent-gated, purgeable (ADR-092) — retains the whole
# payload verbatim. Credentials never persist at any detail level, and
# every withholding (dropped key, redacted instruction body, elided
# history range) is named in ``omitted_keys`` so the Inspector shows it.

#: Strings at/above this length are candidates for base64 stubbing.
_STUB_MIN_CHARS = 4096
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<data>.+)$", re.DOTALL)
_CREDENTIAL_KEYS = frozenset({
    "api_key", "authorization", "password", "token", "secret",
    "access_token", "client_secret",
})
_SEMANTIC_JSON_STRING_KEYS = frozenset({"arguments", "input", "result", "output"})

EXCHANGE_BLOB_MAX_BYTES = 16 * 1024 * 1024
CAPTURE_JSON_MAX_BYTES = 64 * 1024 * 1024

#: How many trailing ``messages_payload`` rows a Safe capture keeps verbatim.
#: The tail always covers the turn's NEW rows (one user row on the direct
#: path; assistant tool_calls + tool results on an agent-loop call) plus a
#: few rows of immediate context; everything older is fingerprinted in
#: place (task-23026).
CAPTURE_SAFE_HISTORY_TAIL_ROWS = 8

#: Structural marker on a fingerprint row produced by
#: :func:`elide_safe_history_rows`. Keyed structurally (never by content
#: prefix) so re-running the builder over a stored request — the export
#: projection path — passes already-elided rows through untouched.
CAPTURE_ELIDED_ROW_KEY = "capture_elided"


class CaptureDetail(str, Enum):
    """The permitted local capture detail levels."""

    SAFE = "safe"
    FULL = "full"


class CapturePolicySource(str, Enum):
    """The source whose valid capture-detail value won precedence."""

    DISABLED = "disabled"
    NEXT_SEND = "next_send"
    CONVERSATION = "conversation"
    GLOBAL = "global"
    APPLICATION = "application"


@dataclass(frozen=True)
class CapturePolicyResolution:
    enabled: bool
    detail: CaptureDetail
    source: CapturePolicySource
    invalid_sources: tuple[str, ...]


@dataclass
class CaptureBudget:
    """One bounded uncompressed capture budget shared by request and response."""

    limit_bytes: int = CAPTURE_JSON_MAX_BYTES
    used_bytes: int = 0

    def retain(self, value: Any) -> bool:
        size = 0
        for chunk in json.JSONEncoder(default=str, ensure_ascii=False).iterencode(value):
            size += len(chunk.encode("utf-8"))
            if self.used_bytes + size > self.limit_bytes:
                return False
        self.used_bytes += size
        return True


class CaptureUnavailableError(ValueError):
    """Capture bytes cannot be safely decoded or retained."""


class CaptureCorruptError(CaptureUnavailableError):
    """Persisted capture provenance is malformed or inconsistent."""


def _capture_detail(value: object) -> CaptureDetail | None:
    if isinstance(value, CaptureDetail):
        return value
    if isinstance(value, str):
        try:
            return CaptureDetail(value)
        except ValueError:
            return None
    return None


def resolve_capture_policy(
    *,
    enabled: bool,
    next_send: object = None,
    conversation: object = None,
    global_default: object = None,
    allow_next_send: bool = True,
) -> CapturePolicyResolution:
    """Resolve capture detail without treating an invalid value as Full.

    Args:
        enabled: Whether exchange capture is enabled for future calls.
        next_send: Optional one-shot detail override.
        conversation: Optional persisted conversation detail override.
        global_default: Optional global detail default.
        allow_next_send: Whether this admission may consume the one-shot scope.

    Returns:
        The first valid scoped detail, its source, and any invalid sources
        skipped while resolving. Safe is returned when none are valid.
    """
    candidates = (
        ("next_send", CapturePolicySource.NEXT_SEND, next_send, allow_next_send),
        ("conversation", CapturePolicySource.CONVERSATION, conversation, True),
        ("global", CapturePolicySource.GLOBAL, global_default, True),
    )
    invalid: list[str] = []
    for name, source, value, allowed in candidates:
        if not allowed or value is None:
            continue
        detail = _capture_detail(value)
        if detail is None:
            invalid.append(name)
            continue
        return CapturePolicyResolution(
            enabled=enabled,
            detail=detail,
            source=source,
            invalid_sources=tuple(invalid),
        )
    return CapturePolicyResolution(
        enabled=enabled,
        detail=CaptureDetail.SAFE,
        source=CapturePolicySource.APPLICATION,
        invalid_sources=tuple(invalid),
    )


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
    capture_detail: CaptureDetail = CaptureDetail.SAFE


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
    match = _DATA_URI_RE.match(value)
    if match:
        # Qodo PR #1883 finding: hash/size the whitespace-stripped payload,
        # not the raw one -- otherwise the same bytes line-wrapped at a
        # different column produce a different sha256/size and the
        # "deterministic stub" promise (identical bytes -> identical stub)
        # breaks for any line-wrapped data URI.
        canonical_data = "".join(match.group("data").split())
        return _stub_for(canonical_data, match.group("mime"))
    if len("".join(value.split())) < _STUB_MIN_CHARS and mime_hint is None:
        return value
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
                out[key] = _maybe_stub_string(
                    value,
                    mime_hint
                    if isinstance(mime_hint, str)
                    else "application/octet-stream",
                )
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


def _redact_project_instruction_rows(
    messages_payload: Any, capture_detail: CaptureDetail
) -> tuple[Any, tuple[str, ...]]:
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
        if capture_detail is CaptureDetail.FULL or not (
            isinstance(row, Mapping)
            and row.get(EPHEMERAL_ORIGIN_KEY) == _PROJECT_INSTRUCTION_ORIGIN
            # task-23026: a fingerprint row from `elide_safe_history_rows`
            # is already content-free — re-marking it would replace the
            # fingerprint with a marker measuring the MARKER's length, so
            # the export path's re-run of this builder over a stored
            # request must pass it through untouched.
            and not row.get(CAPTURE_ELIDED_ROW_KEY)
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


def _remove_nested_credentials(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_semantic_json_string(nested)
            if str(key).lower() in _SEMANTIC_JSON_STRING_KEYS and isinstance(nested, str)
            else _remove_nested_credentials(nested)
            for key, nested in value.items()
            if str(key).lower() not in _CREDENTIAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_remove_nested_credentials(item) for item in value]
    return value


def _sanitize_semantic_json_string(value: str) -> str:
    """Sanitize structured provider tool payloads without changing plain text."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(parsed, (Mapping, list)):
        return value
    return json.dumps(
        sanitize_capture_value(parsed),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sanitize_capture_value(value: Any) -> Any:
    """Remove structured credentials and stub binary from an arbitrary value.

    Args:
        value: Provider-shaped semantic value to make capture-safe.

    Returns:
        A JSON-compatible value with credential fields removed and binary or
        base64 strings replaced by bounded stubs.
    """
    return stub_binary_strings(_remove_nested_credentials(_jsonable(value)))


def _retain_with_budget(
    value: Any, budget: CaptureBudget, path: str, inventory: list[str]
) -> Any:
    if budget.retain(value):
        return value
    inventory.append(path)
    return {"truncated": True}


def _canonical_row_json(row: Any) -> str:
    return json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _fingerprint_row(row: Any) -> dict[str, Any]:
    """Replace one history row with a content-free, verifiable fingerprint.

    The sha256 prefix is computed over the row's canonical sanitized JSON,
    which is exactly the form an earlier capture retained when this row was
    in its verbatim tail — so a viewer can confirm identity across turns
    (and against the transcript) without the body, the same contract the
    binary stubs already make.
    """
    digest = hashlib.sha256(
        _canonical_row_json(row).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    role = row.get("role") if isinstance(row, Mapping) else None
    content = row.get("content") if isinstance(row, Mapping) else row
    if isinstance(content, str):
        chars = len(content)
    elif content is None:
        chars = 0
    else:
        chars = len(_canonical_row_json(content))
    out: dict[str, Any] = {
        "role": str(role) if role is not None else "?",
        CAPTURE_ELIDED_ROW_KEY: True,
        "content": (
            f"[conversation history elided by capture policy -- "
            f"{chars} chars, sha256:{digest}]"
        ),
    }
    if isinstance(row, Mapping) and EPHEMERAL_ORIGIN_KEY in row:
        out[EPHEMERAL_ORIGIN_KEY] = row[EPHEMERAL_ORIGIN_KEY]
    return out


def elide_safe_history_rows(
    rows: Any,
    capture_detail: CaptureDetail,
    *,
    path: str = "messages_payload",
) -> tuple[Any, tuple[str, ...]]:
    """Bound a Safe capture's per-turn conversation-history copy (task-23026).

    Every send's payload carries the whole conversation so far, so
    persisting it verbatim per turn re-stored the entire history O(n²)
    (21.33 MB measured for one 200-turn conversation). Under Safe — the
    default — only the newest ``CAPTURE_SAFE_HISTORY_TAIL_ROWS`` rows (the
    turn's new rows plus immediate context) stay verbatim; each older row
    is replaced IN PLACE by a fingerprint row (role + origin tag + char
    count + sha256 prefix), so the payload's exact shape, order, and
    per-row identity remain inspectable and the count the Inspector shows
    stays truthful. Idempotent: rows already carrying
    ``CAPTURE_ELIDED_ROW_KEY`` pass through unchanged, so the export
    projection's re-run over a stored request is a fixed point.

    Args:
        rows: The sanitized ``messages_payload`` (or wire ``messages``)
            list; any non-list shape passes through untouched.
        capture_detail: Full skips elision entirely — Full is the explicit,
            consent-gated, purgeable verbatim mode (ADR-092).
        path: Inventory path prefix for the returned omission entry.

    Returns:
        The (possibly new) row list and a 0- or 1-entry inventory tuple
        naming the elided range, rendered by the Inspector's existing
        "Omitted by capture policy" line.
    """
    if capture_detail is CaptureDetail.FULL or not isinstance(rows, list):
        return rows, ()
    if len(rows) <= CAPTURE_SAFE_HISTORY_TAIL_ROWS:
        return rows, ()
    cut = len(rows) - CAPTURE_SAFE_HISTORY_TAIL_ROWS
    out: list[Any] = []
    elided_any = False
    for index, row in enumerate(rows):
        if index >= cut or (
            isinstance(row, Mapping) and row.get(CAPTURE_ELIDED_ROW_KEY)
        ):
            out.append(row)
            continue
        out.append(_fingerprint_row(row))
        elided_any = True
    if not elided_any:
        return rows, ()
    return out, (
        f"{path}[0..{cut - 1}].content (conversation history elided)",
    )


def build_request_capture(
    kwargs: Mapping[str, Any],
    *,
    capture_detail: CaptureDetail = CaptureDetail.SAFE,
    budget: CaptureBudget | None = None,
) -> tuple[dict, tuple[str, ...]]:
    """Return (allowlisted+stubbed request dict, names of dropped keys).

    ``omitted_keys`` doubles as the redaction-visibility signal (C1): when
    ``messages_payload`` contains a project-instruction row, its
    ``messages_payload[<index>].content`` path is folded into this same
    tuple alongside genuinely dropped top-level keys (e.g. ``api_key``) --
    the Inspector already renders this tuple verbatim as an "Omitted by
    capture policy" line, so a viewer sees the withholding without any new
    UI surface.

    Args:
        kwargs: Provider request values at the semantic adapter boundary.
        capture_detail: Safe or Full instruction-retention policy.
        budget: Optional shared uncompressed capture budget.

    Returns:
        The allowlisted, sanitized request and sorted omission inventory.
    """
    active_budget = budget or CaptureBudget()
    request: dict = {}
    omitted: list[str] = []
    truncation_inventory: list[str] = []
    for key, value in kwargs.items():
        if key in CAPTURE_REQUEST_ALLOWLIST:
            if key == "messages_payload":
                value, redacted_paths = _redact_project_instruction_rows(value, capture_detail)
                omitted.extend(redacted_paths)
            if key in {"api_endpoint", "api_base_url"} and isinstance(value, str):
                try:
                    value = canonical_provider_endpoint_identity(value)
                except ValueError:
                    value = "[invalid endpoint]"
            value = sanitize_capture_value(value)
            if key == "messages_payload":
                # task-23026: after sanitization (so fingerprints hash the
                # exact form an earlier capture's tail retained), bound the
                # per-turn history copy under Safe.
                value, elided_paths = elide_safe_history_rows(
                    value, capture_detail
                )
                omitted.extend(elided_paths)
            request[key] = _retain_with_budget(
                value, active_budget, key, truncation_inventory
            )
        else:
            omitted.append(str(key))
    request["truncation_inventory"] = tuple(truncation_inventory)
    return request, tuple(sorted(omitted))


def build_response_capture(
    *,
    content: str,
    tool_calls: Sequence[Mapping[str, Any]],
    synthetic_fallback: bool = False,
    budget: CaptureBudget | None = None,
) -> dict[str, Any]:
    """Build a binary-stubbed response under the same capture budget.

    Args:
        content: Accumulated provider response text.
        tool_calls: Accumulated provider tool-call structures.
        synthetic_fallback: Whether the response came from a fallback path.
        budget: Optional shared uncompressed capture budget.

    Returns:
        A sanitized response mapping with a truncation inventory.
    """
    active_budget = budget or CaptureBudget()
    inventory: list[str] = []
    response = {
        "content": _retain_with_budget(
            sanitize_capture_value(content), active_budget, "content", inventory
        ),
        "tool_calls": _retain_with_budget(
            sanitize_capture_value(tool_calls),
            active_budget,
            "tool_calls",
            inventory,
        ),
        "synthetic_fallback": bool(synthetic_fallback),
    }
    response["truncation_inventory"] = tuple(inventory)
    return response


def capture_to_blob(capture: ExchangeCapture) -> bytes:
    """zlib-compressed JSON; oversize captures truncate, never fail.

    Review finding M13: the oversize branch used to overwrite ``status``
    with ``"truncated"``, discarding whether the call had actually
    completed/stopped/errored. The real outcome is preserved; truncation
    is marked separately via a ``truncated: True`` key in the (now
    stubbed) request/response dicts.
    """
    payload = asdict(capture)
    payload["capture_detail"] = capture.capture_detail.value
    try:
        raw = _encode_capture_json(payload)
    except CaptureUnavailableError:
        raw = _encode_capture_json(_truncated_capture_payload(capture))
    blob = zlib.compress(raw)
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
    return zlib.compress(_encode_capture_json(_capture_payload(truncated)))


def _capture_payload(capture: ExchangeCapture) -> dict[str, Any]:
    payload = asdict(capture)
    payload["capture_detail"] = capture.capture_detail.value
    return payload


def _truncated_capture_payload(capture: ExchangeCapture) -> dict[str, Any]:
    return _capture_payload(replace(
        capture,
        request={"truncated": True, "reason": "capture exceeds safe encode limit"},
        response={"truncated": True},
    ))


def _encode_capture_json(value: Any) -> bytes:
    total = 0
    chunks: list[bytes] = []
    for chunk in json.JSONEncoder(default=str, ensure_ascii=False).iterencode(value):
        encoded = chunk.encode("utf-8")
        total += len(encoded)
        if total > CAPTURE_JSON_MAX_BYTES:
            raise CaptureUnavailableError("capture exceeds safe encode limit")
        chunks.append(encoded)
    return b"".join(chunks)


def capture_from_blob(blob: bytes) -> ExchangeCapture:
    """Inverse of :func:`capture_to_blob`.

    Review finding M11: filters to ``ExchangeCapture``'s own known field
    names before construction -- a future blob written by a newer version
    with an extra field would otherwise raise ``TypeError`` here today,
    on every OLDER build reading it back.
    """
    if len(blob) > EXCHANGE_BLOB_MAX_BYTES:
        raise CaptureUnavailableError("capture exceeds safe decode limit")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(blob, CAPTURE_JSON_MAX_BYTES + 1)
        if len(raw) > CAPTURE_JSON_MAX_BYTES or decompressor.unconsumed_tail:
            raise CaptureUnavailableError("capture exceeds safe decode limit")
        raw += decompressor.flush()
        if len(raw) > CAPTURE_JSON_MAX_BYTES:
            raise CaptureUnavailableError("capture exceeds safe decode limit")
        data = json.loads(raw)
    except CaptureUnavailableError:
        raise
    except (ValueError, zlib.error, json.JSONDecodeError) as exc:
        raise CaptureCorruptError("capture is corrupt") from exc
    if not isinstance(data, dict):
        raise CaptureCorruptError("capture is corrupt")
    detail = _capture_detail(data.get("capture_detail", CaptureDetail.SAFE))
    if detail is None:
        raise CaptureCorruptError("capture detail is corrupt")
    data["capture_detail"] = detail
    data["omitted_keys"] = tuple(data.get("omitted_keys") or ())
    known_fields = {f.name for f in fields(ExchangeCapture)}
    filtered = {key: value for key, value in data.items() if key in known_fields}
    return ExchangeCapture(**filtered)


def capture_from_storage(blob: bytes, declared_detail: object) -> ExchangeCapture:
    """Decode local capture bytes only when the row and blob agree."""
    capture = capture_from_blob(blob)
    detail = _capture_detail(declared_detail)
    if detail is None or capture.capture_detail is not detail:
        raise CaptureCorruptError("capture provenance mismatch")
    return capture


def trim_safe_capture_blob(blob: bytes) -> bytes | None:
    """Apply Safe history elision to one STORED capture blob (task-23026).

    Pure helper for the ChaChaNotes v52→v53 migration: captures persisted
    before elision existed carry the whole conversation per turn. This
    decodes the blob, bounds ``request.messages_payload`` (and the
    llama.cpp branch's ``request.wire_payload.messages``) exactly the way
    :func:`build_request_capture` now does at capture time, folds the
    elided range into ``omitted_keys``, and re-encodes. Everything else —
    response, usage, status, provenance, the retained tail — is preserved
    value-identical.

    Args:
        blob: One ``message_exchanges.capture_blob`` value.

    Returns:
        The trimmed replacement blob, or ``None`` when nothing changed
        (already trimmed, short payload, or a Full capture — Full is the
        deliberate verbatim mode and is never rewritten here).

    Raises:
        CaptureUnavailableError: If the blob cannot be decoded (corrupt or
            over the safety limits) — the caller leaves such a row as-is.
    """
    capture = capture_from_blob(blob)
    if capture.capture_detail is not CaptureDetail.SAFE:
        return None
    if not isinstance(capture.request, dict):
        return None
    new_request = dict(capture.request)
    new_paths: list[str] = []
    rows = new_request.get("messages_payload")
    trimmed_rows, elided = elide_safe_history_rows(rows, CaptureDetail.SAFE)
    if elided:
        new_request["messages_payload"] = trimmed_rows
        new_paths.extend(elided)
    wire = new_request.get("wire_payload")
    if isinstance(wire, Mapping):
        trimmed_wire_rows, wire_elided = elide_safe_history_rows(
            wire.get("messages"),
            CaptureDetail.SAFE,
            path="wire_payload.messages",
        )
        if wire_elided:
            new_wire = dict(wire)
            new_wire["messages"] = trimmed_wire_rows
            new_request["wire_payload"] = new_wire
            new_paths.extend(wire_elided)
    if not new_paths:
        return None
    merged_omitted = tuple(sorted(set(capture.omitted_keys).union(new_paths)))
    return capture_to_blob(
        replace(capture, request=new_request, omitted_keys=merged_omitted)
    )
