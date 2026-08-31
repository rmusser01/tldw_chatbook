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
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from tldw_chatbook.Chat.console_project_instructions import (
    EPHEMERAL_ORIGIN_KEY,
    canonical_provider_endpoint_identity,
)
from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer

#: Value ``EPHEMERAL_ORIGIN_KEY`` carries on an automatically-injected
#: project-instruction row (``Agents/project_instruction_runtime.py``'s
#: ``_source_row``/``_outcome_row``/``_warning_row``,
#: ``Agents/agent_service.py``'s ``build_project_instruction_row``). Kept as
#: a local literal rather than importing ``PROJECT_INSTRUCTION_ORIGIN`` from
#: either of those modules -- both define the same literal independently and
#: neither is this module's natural dependency.
_PROJECT_INSTRUCTION_ORIGIN = "project_instructions"

CAPTURE_REQUEST_ALLOWLIST: frozenset[str] = frozenset(
    {
        "api_endpoint",
        "api_base_url",
        "system_message",
        "messages_payload",
        "tools",
        "model",
        "streaming",
        "temp",
        "topp",
        "maxp",
        "topk",
        "minp",
        "max_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
        "prompt_caching",
        "response_format",
        "api_mode",
        "request_timeout",
        "request_retries",
        "request_retry_delay",
        "provider_continuations",
    }
)
# Deliberately OFF the allowlist: "api_key" (credential) and
# "api_key_resolved" (credential-adjacent marker) — they surface in
# omitted_keys instead.
#
# Privacy stance for what IS on the list (task-23026, ADR-096): capture
# exists to answer "what did we actually hand the provider adapter on this
# call", so the conversation content is the subject, not an accident — it
# is retained by design, under the user-controllable Safe/Full detail
# policy rather than the allowlist. Under Safe (the default) retention is
# BOUNDED per ADR-096's contract: the first system row, the latest user
# row, and the final eight physical payload rows persist verbatim; every
# other row is represented by ONE content-free aggregate marker (counts
# and retained positions only — no content, snippets, per-row lengths,
# hashes, IDs, or timestamps; a digest of omitted text is explicitly
# forbidden because it would let anyone with DB access confirm guesses
# about private content). Storing the whole history per turn grew O(n²) —
# measured 21.33 MB for one 200-turn conversation. Full — explicit,
# consent-gated, purgeable (ADR-092) — retains the whole payload verbatim.
# Credentials never persist at any detail level, and every withholding
# (dropped key, redacted instruction body, elided history) is named in
# ``omitted_keys`` so the Inspector shows it.

#: Strings at/above this length are candidates for base64 stubbing.
_STUB_MIN_CHARS = 4096
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<data>.+)$", re.DOTALL
)
_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "token",
        "secret",
        "access_token",
        "client_secret",
    }
)
_SEMANTIC_JSON_STRING_KEYS = frozenset({"arguments", "input", "result", "output"})
_CREDENTIAL_OMISSION = {"omitted": True}
_UNKNOWN_PARAMETER_OMISSION = "unknown_parameter"
_BOUNDED_DROPPED_PARAMETER_LABELS = _CREDENTIAL_KEYS | {"api_key_resolved"}

EXCHANGE_BLOB_MAX_BYTES = 16 * 1024 * 1024
CAPTURE_JSON_MAX_BYTES = 64 * 1024 * 1024

#: How many trailing physical payload rows a Safe capture keeps verbatim
#: (ADR-096). The tail always covers the turn's NEW rows (one user row on
#: the direct path; assistant tool_calls + tool results on an agent-loop
#: call) plus a few rows of immediate context.
CAPTURE_SAFE_HISTORY_TAIL_ROWS = 8

#: Versioned kind discriminator of the ONE aggregate history-elision
#: marker :func:`compact_safe_history_rows` inserts (ADR-096). Recognition
#: is structural and strict (`_is_valid_history_elision_marker`); a
#: malformed lookalike is an ordinary row.
CAPTURE_HISTORY_ELISION_KIND = "tldw.exchange_capture.safe_history_elision"
CAPTURE_HISTORY_ELISION_VERSION = 1

#: The marker's EXACT key set. ADR-096 forbids the marker carrying
#: anything beyond these — in particular content, snippets, per-row
#: lengths, hashes/digests, IDs, or timestamps. The shape-guard test pins
#: this frozen set so a digest cannot be reintroduced silently.
CAPTURE_HISTORY_MARKER_KEYS = frozenset(
    {
        "kind",
        "version",
        "original_rows",
        "omitted_rows",
        "omitted_roles",
        "retained_positions",
    }
)

#: Normalized role buckets the marker counts omitted rows into. Unknown,
#: missing, or non-string roles count only toward ``other`` — their raw
#: values are never retained in the marker.
CAPTURE_HISTORY_MARKER_ROLES = ("system", "user", "assistant", "tool", "other")


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
        for chunk in json.JSONEncoder(default=str, ensure_ascii=False).iterencode(
            value
        ):
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
            if str(key).lower() in _SEMANTIC_JSON_STRING_KEYS
            and isinstance(nested, str)
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


def sanitize_capture_value_with_omission(
    value: Any,
    *,
    known_credentials: tuple[str, ...] = (),
) -> tuple[Any, bool]:
    result = CredentialSanitizer(known_credentials=known_credentials).sanitize(value)
    if not result.available:
        return dict(_CREDENTIAL_OMISSION), True
    return (
        stub_binary_strings(_remove_nested_credentials(result.value)),
        result.redacted,
    )


def sanitize_capture_value(value: Any) -> Any:
    """Remove structured credentials and stub binary from an arbitrary value.

    Args:
        value: Provider-shaped semantic value to make capture-safe.

    Returns:
        A JSON-compatible value with credential fields removed and binary or
        base64 strings replaced by bounded stubs.
    """
    sanitized, _omitted = sanitize_capture_value_with_omission(value)
    return sanitized


def _retain_with_budget(
    value: Any, budget: CaptureBudget, path: str, inventory: list[str]
) -> Any:
    if budget.retain(value):
        return value
    inventory.append(path)
    return {"truncated": True}


def _is_valid_history_elision_marker(row: Any) -> bool:
    """Strict structural recognition of the capture's own marker (ADR-096).

    Recognition is by exact versioned shape, never by content prefix: the
    key set must equal ``CAPTURE_HISTORY_MARKER_KEYS`` exactly, the kind
    and version must match, counts must be real non-negative ints (bools
    rejected), ``omitted_roles`` must carry exactly the five normalized
    buckets with int values, and ``retained_positions`` must be a list of
    ints. A malformed lookalike is an ordinary row and remains subject to
    the normal bounded selection.
    """
    if not isinstance(row, Mapping):
        return False
    if set(row.keys()) != CAPTURE_HISTORY_MARKER_KEYS:
        return False
    if row.get("kind") != CAPTURE_HISTORY_ELISION_KIND:
        return False
    version = row.get("version")
    if type(version) is not int or version != CAPTURE_HISTORY_ELISION_VERSION:
        return False
    for key in ("original_rows", "omitted_rows"):
        value = row.get(key)
        if type(value) is not int or value < 0:
            return False
    roles = row.get("omitted_roles")
    if not isinstance(roles, Mapping):
        return False
    if set(roles.keys()) != set(CAPTURE_HISTORY_MARKER_ROLES):
        return False
    if any(type(value) is not int or value < 0 for value in roles.values()):
        return False
    positions = row.get("retained_positions")
    if not isinstance(positions, list):
        return False
    return all(type(position) is int for position in positions)


def history_elision_marker(rows: Any) -> Mapping[str, Any] | None:
    """Return the list's valid history-elision marker, if any.

    Public reader for the Inspector: lets the Messages section title state
    the original row count honestly when the stored list is compacted.
    """
    if not isinstance(rows, (list, tuple)):
        return None
    for row in rows:
        if _is_valid_history_elision_marker(row):
            return row
    return None


def _normalized_omitted_role(row: Any) -> str:
    if isinstance(row, Mapping):
        role = row.get("role")
        if isinstance(role, str) and role in ("system", "user", "assistant", "tool"):
            return role
    return "other"


def compact_safe_history_rows(
    rows: Any,
    capture_detail: CaptureDetail,
    *,
    path: str = "messages_payload",
) -> tuple[Any, tuple[str, ...]]:
    """Bound a Safe capture's per-turn conversation-history copy (ADR-096).

    Every send's payload carries the whole conversation so far, so
    persisting it verbatim per turn re-stored the entire history O(n²)
    (21.33 MB measured for one 200-turn conversation). Under Safe — the
    default — the retained set is the union of: the first mapping row
    whose ``role`` is exactly ``system``, the last mapping row whose
    ``role`` is exactly ``user``, and the final
    ``CAPTURE_SAFE_HISTORY_TAIL_ROWS`` physical rows — deduplicated, in
    original relative order, values untouched. Non-mapping rows are
    eligible only through the tail. Every other row is represented by ONE
    content-free aggregate marker inserted at the position of the first
    omitted row (counts and retained positions only — never content,
    snippets, per-row lengths, digests, IDs, or timestamps: a digest would
    let anyone with database access confirm guesses about omitted private
    text).

    Idempotent by marker recognition: a recognized valid marker is
    transparent to selection and preserved when nothing else is omitted,
    so re-projecting a stored Safe request (the export path re-runs this
    builder) is a fixed point. An input marker never disables compaction
    of surrounding rows — when new rows must be omitted, stale markers are
    dropped and exactly one fresh marker describes this pass.

    Args:
        rows: The sanitized ``messages_payload`` (or wire ``messages``)
            list; any non-list shape passes through untouched.
        capture_detail: Full skips compaction entirely — Full is the
            explicit, consent-gated, purgeable verbatim mode (ADR-092).
        path: Stable omission-inventory prefix; the entry is
            ``f"{path}.history"`` so repeated projection cannot create
            duplicate or ever-changing strings.

    Returns:
        The (possibly new) row list and a 0- or 1-entry inventory tuple,
        rendered by the Inspector's existing "Omitted by capture policy"
        line.
    """
    if capture_detail is CaptureDetail.FULL or not isinstance(rows, list):
        return rows, ()
    real = [
        (position, row)
        for position, row in enumerate(rows)
        if not _is_valid_history_elision_marker(row)
    ]
    if not real:
        return rows, ()
    retained: set[int] = set()
    for position, row in real:
        if isinstance(row, Mapping) and row.get("role") == "system":
            retained.add(position)
            break
    for position, row in reversed(real):
        if isinstance(row, Mapping) and row.get("role") == "user":
            retained.add(position)
            break
    tail_start = max(0, len(rows) - CAPTURE_SAFE_HISTORY_TAIL_ROWS)
    retained.update(position for position, _row in real if position >= tail_start)
    omitted = [(position, row) for position, row in real if position not in retained]
    if not omitted:
        # Fixed point: nothing new to omit — the list (including any
        # already-present marker) is returned unchanged.
        return rows, ()
    role_counts = {role: 0 for role in CAPTURE_HISTORY_MARKER_ROLES}
    for _position, row in omitted:
        role_counts[_normalized_omitted_role(row)] += 1
    marker: dict[str, Any] = {
        "kind": CAPTURE_HISTORY_ELISION_KIND,
        "version": CAPTURE_HISTORY_ELISION_VERSION,
        "original_rows": len(real),
        "omitted_rows": len(omitted),
        "omitted_roles": role_counts,
        "retained_positions": sorted(retained),
    }
    first_omitted = omitted[0][0]
    out: list[Any] = []
    for position, row in enumerate(rows):
        if position == first_omitted:
            out.append(marker)
        if _is_valid_history_elision_marker(row):
            # Stale metadata from a previous compaction of a DIFFERENT row
            # set — exactly one fresh marker describes this pass.
            continue
        if position in retained:
            out.append(row)
    return out, (f"{path}.history",)


def build_request_capture(
    kwargs: Mapping[str, Any],
    *,
    capture_detail: CaptureDetail = CaptureDetail.SAFE,
    budget: CaptureBudget | None = None,
    known_credentials: tuple[str, ...] = (),
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
                value, redacted_paths = _redact_project_instruction_rows(
                    value, capture_detail
                )
                omitted.extend(redacted_paths)
            if key in {"api_endpoint", "api_base_url"} and isinstance(value, str):
                try:
                    value = canonical_provider_endpoint_identity(value)
                except ValueError:
                    value = "[invalid endpoint]"
            value, credential_omitted = sanitize_capture_value_with_omission(
                value,
                known_credentials=known_credentials,
            )
            if credential_omitted:
                omitted.append(key)
            if key == "messages_payload":
                # ADR-096 ordering: compaction runs AFTER redaction and
                # sanitization (so the marker can never describe raw
                # secret/binary values) and BEFORE the shared budget.
                value, elided_paths = compact_safe_history_rows(value, capture_detail)
                omitted.extend(elided_paths)
            request[key] = _retain_with_budget(
                value, active_budget, key, truncation_inventory
            )
        else:
            omitted.append(
                key
                if key in _BOUNDED_DROPPED_PARAMETER_LABELS
                else _UNKNOWN_PARAMETER_OMISSION
            )
    request["truncation_inventory"] = tuple(truncation_inventory)
    return request, tuple(sorted(set(omitted)))


def build_response_capture(
    *,
    content: str,
    tool_calls: Sequence[Mapping[str, Any]],
    synthetic_fallback: bool = False,
    budget: CaptureBudget | None = None,
    known_credentials: tuple[str, ...] = (),
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
    content_value, content_omitted = sanitize_capture_value_with_omission(
        content,
        known_credentials=known_credentials,
    )
    tools_value, tools_omitted = sanitize_capture_value_with_omission(
        tool_calls,
        known_credentials=known_credentials,
    )
    response = {
        "content": _retain_with_budget(
            content_value, active_budget, "content", inventory
        ),
        "tool_calls": _retain_with_budget(
            tools_value,
            active_budget,
            "tool_calls",
            inventory,
        ),
        "synthetic_fallback": bool(synthetic_fallback),
    }
    response["truncation_inventory"] = tuple(inventory)
    response["credential_omission_inventory"] = tuple(
        name
        for name, omitted in (
            ("content", content_omitted),
            ("tool_calls", tools_omitted),
        )
        if omitted
    )
    return response


def capture_to_blob(capture: ExchangeCapture) -> bytes:
    """zlib-compressed JSON; oversize captures truncate, never fail.

    Review finding M13: the oversize branch used to overwrite ``status``
    with ``"truncated"``, discarding whether the call had actually
    completed/stopped/errored. The real outcome is preserved; truncation
    is marked separately via a ``truncated: True`` key in the (now
    stubbed) request/response dicts.
    """
    payload = _capture_payload(capture)
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
    """Apply the mandatory final credential gate before serialization."""
    payload = {
        "run_tag": capture.run_tag,
        "seq": capture.seq,
        "created_at": capture.created_at,
        "provider": capture.provider,
        "model": capture.model,
        "endpoint": capture.endpoint,
        "request": capture.request,
        "response": capture.response,
        "status": capture.status,
        "usage_json": capture.usage_json,
        "omitted_keys": capture.omitted_keys,
        "capture_detail": capture.capture_detail.value,
    }
    result = CredentialSanitizer().sanitize(payload)
    if result.available and isinstance(result.value, Mapping):
        sanitized = dict(result.value)
        if result.redacted:
            omitted = sanitized.get("omitted_keys")
            omitted_names = (
                {item for item in omitted if type(item) is str}
                if isinstance(omitted, (list, tuple))
                else set()
            )
            omitted_names.add("capture.credential_redacted")
            sanitized["omitted_keys"] = sorted(omitted_names)
        return sanitized
    return {
        "run_tag": "",
        "seq": capture.seq if type(capture.seq) is int else 0,
        "created_at": "",
        "provider": "",
        "model": "",
        "endpoint": None,
        "request": dict(_CREDENTIAL_OMISSION),
        "response": dict(_CREDENTIAL_OMISSION),
        "status": (
            capture.status
            if capture.status in {"complete", "stopped", "error"}
            else "error"
        ),
        "usage_json": None,
        "omitted_keys": ["capture"],
        "capture_detail": capture.capture_detail.value,
    }


def _truncated_capture_payload(capture: ExchangeCapture) -> dict[str, Any]:
    return _capture_payload(
        replace(
            capture,
            request={"truncated": True, "reason": "capture exceeds safe encode limit"},
            response={"truncated": True},
        )
    )


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
    """Apply ADR-096 Safe history compaction to one STORED capture blob.

    Pure helper for the ChaChaNotes v52→v53 migration: captures persisted
    before compaction existed carry the whole conversation per turn. This
    decodes the blob, compacts ``request.messages_payload`` (and the
    llama.cpp branch's ``request.wire_payload.messages``) exactly the way
    :func:`build_request_capture` now does at capture time, merges the
    stable history-elision paths into ``omitted_keys``, and re-encodes.
    Everything else — response, usage, status, provenance, the retained
    rows — is preserved value-identical.

    Args:
        blob: One ``message_exchanges.capture_blob`` value.

    Returns:
        The compacted replacement blob, or ``None`` when nothing changed
        (already compacted, small payload, or a Full capture — Full is the
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
    compacted_rows, elided = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    if elided:
        new_request["messages_payload"] = compacted_rows
        new_paths.extend(elided)
    wire = new_request.get("wire_payload")
    if isinstance(wire, Mapping):
        compacted_wire_rows, wire_elided = compact_safe_history_rows(
            wire.get("messages"),
            CaptureDetail.SAFE,
            path="wire_payload.messages",
        )
        if wire_elided:
            new_wire = dict(wire)
            new_wire["messages"] = compacted_wire_rows
            new_request["wire_payload"] = new_wire
            new_paths.extend(wire_elided)
    if not new_paths:
        return None
    merged_omitted = tuple(sorted(set(capture.omitted_keys).union(new_paths)))
    return capture_to_blob(
        replace(capture, request=new_request, omitted_keys=merged_omitted)
    )
