"""Bounded, device-local review records for assistant Library operations."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionWriter,
)
from tldw_chatbook.Chat.trajectory import contains_local_path, redact_local_paths
from tldw_chatbook.Utils.log_sanitizer import redact_log_line

LIBRARY_ACTIVITY_EVENT_KIND = "library_activity"
LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS = 160
LIBRARY_ACTIVITY_TITLE_MAX_CHARS = 160
LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS = 240
LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS = 200
LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT = 8
LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES = 8192

_OPAQUE_ID_MAX_CHARS = 200
_RESULT_COUNT_MAX = (1 << 31) - 1
_VIEW_MAX_ACTIONS = 256
_STABLE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EVENT_KEYS = frozenset(
    {
        "version",
        "event_id",
        "attempt_id",
        "run_id",
        "actor",
        "library_provider",
        "operation",
        "status",
        "result_count",
        "query_preview",
        "source_refs",
        "error_code",
        "error_summary",
    }
)
_ACTOR_KEYS = frozenset({"kind", "run_id", "parent_run_id"})
_SOURCE_REF_KEYS = frozenset({"type", "id", "title"})
_ROW_KEYS = ("results", "items", "messages", "members")
_ID_KEYS = (
    "id",
    "source_id",
    "result_id",
    "note_id",
    "media_id",
    "conversation_id",
    "prompt_id",
    "skill_id",
    "collection_id",
)


@dataclass(frozen=True, slots=True)
class LibraryActivitySourceRef:
    """One bounded source identity safe for local activity review."""

    source_type: str
    source_id: str
    title: str

    def to_payload(self) -> dict[str, str]:
        """Return the stable v1 source-reference mapping."""
        return {
            "type": self.source_type,
            "id": self.source_id,
            "title": self.title,
        }


@dataclass(frozen=True, slots=True)
class LibraryActivityEvent:
    """One minimized built-in Library operation."""

    version: Literal[1]
    event_id: str
    attempt_id: str
    run_id: str
    actor_kind: Literal["primary", "subagent"]
    parent_run_id: str | None
    library_provider: Literal["direct", "rag"]
    operation: str
    status: Literal["succeeded", "empty", "blocked", "failed"]
    result_count: int
    query_preview: str | None
    source_refs: tuple[LibraryActivitySourceRef, ...]
    error_code: str | None
    error_summary: str | None

    def to_payload(self) -> dict[str, Any]:
        """Return the exact bounded v1 persistence/export payload."""
        return {
            "version": self.version,
            "event_id": self.event_id,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "actor": {
                "kind": self.actor_kind,
                "run_id": self.run_id,
                "parent_run_id": self.parent_run_id,
            },
            "library_provider": self.library_provider,
            "operation": self.operation,
            "status": self.status,
            "result_count": self.result_count,
            "query_preview": self.query_preview,
            "source_refs": [ref.to_payload() for ref in self.source_refs],
            "error_code": self.error_code,
            "error_summary": self.error_summary,
        }


@dataclass(frozen=True, slots=True)
class LibraryActivityContributionItem:
    """One buffered event and its native durable-turn opener key."""

    owner_message_key: str
    event: LibraryActivityEvent
    captured_at: float


@dataclass(frozen=True, slots=True)
class LibraryActivityContribution:
    """Persist an ordered batch through the existing Console transaction seam."""

    items: tuple[LibraryActivityContributionItem, ...]

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Insert every event under its durable USER turn opener."""
        if not self.items:
            raise ValueError("Library activity contribution cannot be empty.")
        prepared: list[tuple[str, str, float]] = []
        for item in self.items:
            owner_id = message_ids.get(item.owner_message_key)
            if not isinstance(owner_id, str) or not owner_id:
                raise ValueError("Library activity requires a durable turn opener.")
            if not isinstance(item.captured_at, (int, float)) or isinstance(
                item.captured_at, bool
            ) or not math.isfinite(float(item.captured_at)):
                raise ValueError("Library activity capture time is invalid.")
            prepared.append(
                (owner_id, encode_library_activity_event(item.event), item.captured_at)
            )

        rows = []
        for owner_id, payload_json, captured_at in prepared:
            rows.append(
                (
                    owner_id,
                    conversation_id,
                    owner_id,
                    writer.next_trajectory_sequence(),
                    LIBRARY_ACTIVITY_EVENT_KIND,
                    captured_at,
                    payload_json,
                )
            )
        writer.executemany(
            "INSERT INTO message_trajectory_metadata("
            "message_id, conversation_id, turn_id, seq, event_kind, "
            "step_started_at, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


@dataclass(frozen=True, slots=True)
class LibraryActivityRecord:
    """One validated action projected under a selected durable turn."""

    turn_id: str
    sequence: int
    occurred_at: float | None
    event: LibraryActivityEvent


@dataclass(frozen=True, slots=True)
class LibraryActivityView:
    """Pure selected-turn activity projection for the Console Inspector."""

    selected_turn_id: str | None
    actions: tuple[LibraryActivityRecord, ...]
    corrupt_row_count: int = 0

    @property
    def status(self) -> Literal["empty", "ready", "corrupt"]:
        """Return the bounded presentation state for this projection."""
        if self.corrupt_row_count:
            return "corrupt"
        return "ready" if self.actions else "empty"


@dataclass(frozen=True, slots=True)
class LibraryActivityCandidate:
    """Trusted provider-boundary inputs awaiting minimization."""

    attempt_id: str
    actor_kind: Literal["primary", "subagent"]
    run_id: str
    parent_run_id: str | None
    library_provider: Literal["direct", "rag"]
    operation: str
    arguments: Mapping[str, object]
    structured_result: object
    failure_code: str | None


LibraryActivitySink = Callable[[LibraryActivityEvent], None]


def _bounded_identifier(value: object, *, required: bool = False) -> str | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("required activity identifier is empty")
        return None
    if contains_local_path(text) or redact_log_line(text, max_length=0) != text:
        if required:
            raise ValueError("activity identifier contains private data")
        return None
    return text[:_OPAQUE_ID_MAX_CHARS]


def _bounded_text(value: object, limit: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = redact_local_paths(redact_log_line(text, max_length=0))
    collapsed = " ".join(text.split())
    return collapsed[:limit] or None


def _value(result: object, name: str, default: object = None) -> object:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _rows(result: object) -> tuple[object, ...]:
    for key in _ROW_KEYS:
        raw = _value(result, key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return tuple(raw)
    return ()


def _row_value(row: object, *names: str) -> object:
    for name in names:
        value = row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)
        if value not in (None, ""):
            return value
    return None


def _source_refs(result: object) -> tuple[LibraryActivitySourceRef, ...]:
    refs: list[LibraryActivitySourceRef] = []
    for row in _rows(result):
        source_id = _bounded_identifier(_row_value(row, *_ID_KEYS))
        if source_id is None:
            continue
        source_type = _bounded_text(
            _row_value(row, "type", "source_type", "kind"), 40
        ) or "library"
        title = _bounded_text(
            _row_value(row, "title", "name", "label"),
            LIBRARY_ACTIVITY_TITLE_MAX_CHARS,
        ) or "Untitled"
        refs.append(
            LibraryActivitySourceRef(
                source_type=source_type,
                source_id=source_id[:LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS],
                title=title,
            )
        )
        if len(refs) == LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT:
            break
    return tuple(refs)


def _error(result: object, explicit_code: str | None) -> tuple[str | None, str | None]:
    raw = _value(result, "error")
    if isinstance(raw, Mapping):
        code = raw.get("code") or explicit_code
        summary = raw.get("message")
    else:
        code = explicit_code
        summary = None
    stable_code = str(code or "").strip().lower()
    if not _STABLE_CODE_RE.fullmatch(stable_code):
        stable_code = "activity_failed" if code else ""
    return (
        stable_code or None,
        _bounded_text(summary, LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS),
    )


def _result_count(result: object, rows: tuple[object, ...]) -> int:
    for key in ("total", "returned", "result_count", "returned_message_count"):
        raw = _value(result, key)
        if isinstance(raw, int) and not isinstance(raw, bool):
            return max(0, raw)
    return len(rows)


def minimize_library_activity(candidate: LibraryActivityCandidate) -> LibraryActivityEvent:
    """Convert one trusted provider result into the bounded v1 review event."""
    if candidate.actor_kind not in ("primary", "subagent"):
        raise ValueError("unsupported activity actor kind")
    if candidate.library_provider not in ("direct", "rag"):
        raise ValueError("unsupported Library provider kind")
    attempt_id = _bounded_identifier(candidate.attempt_id, required=True)
    run_id = _bounded_identifier(candidate.run_id, required=True)
    parent_run_id = _bounded_identifier(candidate.parent_run_id)
    operation = str(candidate.operation or "").strip()
    if not _STABLE_CODE_RE.fullmatch(operation):
        raise ValueError("invalid Library activity operation")

    rows = _rows(candidate.structured_result)
    count = _result_count(candidate.structured_result, rows)
    error_code, error_summary = _error(
        candidate.structured_result, candidate.failure_code
    )
    raw_status = str(_value(candidate.structured_result, "status", "") or "").lower()
    if raw_status == "blocked" or error_code == "blocked":
        status: Literal["succeeded", "empty", "blocked", "failed"] = "blocked"
    elif error_code is not None or raw_status == "failed":
        status = "failed"
    elif count == 0:
        status = "empty"
    else:
        status = "succeeded"

    query_preview = _bounded_text(
        candidate.arguments.get("query"), LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS
    )
    event = LibraryActivityEvent(
        version=1,
        event_id=uuid4().hex,
        attempt_id=attempt_id or "",
        run_id=run_id or "",
        actor_kind=candidate.actor_kind,
        parent_run_id=parent_run_id,
        library_provider=candidate.library_provider,
        operation=operation,
        status=status,
        result_count=count,
        query_preview=query_preview,
        source_refs=_source_refs(candidate.structured_result),
        error_code=error_code,
        error_summary=error_summary,
    )
    size = len(
        json.dumps(event.to_payload(), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if size > LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES:
        raise ValueError("bounded Library activity payload exceeds byte ceiling")
    _validate_event(event)
    return event


def encode_library_activity_event(event: LibraryActivityEvent) -> str:
    """Encode one exact bounded v1 activity payload."""
    _validate_event(event)
    encoded = json.dumps(
        event.to_payload(), ensure_ascii=False, separators=(",", ":")
    )
    if len(encoded.encode("utf-8")) > LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES:
        raise ValueError("bounded Library activity payload exceeds byte ceiling")
    return encoded


def decode_library_activity_event(value: object) -> LibraryActivityEvent:
    """Decode an exact v1 payload while rejecting additive or unsafe fields."""
    if type(value) is not str:
        raise ValueError("Invalid Library activity payload.")
    try:
        if len(value.encode("utf-8")) > LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES:
            raise ValueError("Invalid Library activity payload.")
    except UnicodeEncodeError as exc:
        raise ValueError("Invalid Library activity payload.") from exc

    def reject_constant(_value: str) -> None:
        raise ValueError("Invalid Library activity payload.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, item in pairs:
            if key in decoded:
                raise ValueError("Invalid Library activity payload.")
            decoded[key] = item
        return decoded

    try:
        data = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid Library activity payload.") from exc
    if type(data) is not dict or set(data) != _EVENT_KEYS:
        raise ValueError("Invalid Library activity payload.")
    actor = data["actor"]
    refs = data["source_refs"]
    if type(actor) is not dict or set(actor) != _ACTOR_KEYS or type(refs) is not list:
        raise ValueError("Invalid Library activity payload.")
    decoded_refs: list[LibraryActivitySourceRef] = []
    for ref in refs:
        if type(ref) is not dict or set(ref) != _SOURCE_REF_KEYS:
            raise ValueError("Invalid Library activity payload.")
        decoded_refs.append(
            LibraryActivitySourceRef(
                source_type=ref["type"],
                source_id=ref["id"],
                title=ref["title"],
            )
        )
    event = LibraryActivityEvent(
        version=data["version"],
        event_id=data["event_id"],
        attempt_id=data["attempt_id"],
        run_id=data["run_id"],
        actor_kind=actor["kind"],
        parent_run_id=actor["parent_run_id"],
        library_provider=data["library_provider"],
        operation=data["operation"],
        status=data["status"],
        result_count=data["result_count"],
        query_preview=data["query_preview"],
        source_refs=tuple(decoded_refs),
        error_code=data["error_code"],
        error_summary=data["error_summary"],
    )
    if event.run_id != actor["run_id"]:
        raise ValueError("Invalid Library activity payload.")
    _validate_event(event)
    return event


def redacted_library_activity_payload(event: LibraryActivityEvent) -> str:
    """Return the default-export outcome summary without query/source identity."""
    _validate_event(event)
    source_types = list(
        dict.fromkeys(ref.source_type for ref in event.source_refs)
    )
    return json.dumps(
        {
            "version": event.version,
            "operation": event.operation,
            "status": event.status,
            "result_count": event.result_count,
            "source_types": source_types,
            "error_code": event.error_code,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def project_library_activity(
    rows: Sequence[Any],
    active_turn_ids: Sequence[str],
    selected_turn_id: str | None,
) -> LibraryActivityView:
    """Project valid activity for one selected turn on the active lineage."""
    active = frozenset(active_turn_ids)
    if selected_turn_id is None or selected_turn_id not in active:
        return LibraryActivityView(selected_turn_id=selected_turn_id, actions=())

    candidates: list[LibraryActivityRecord] = []
    corrupt = 0
    seen_event_ids: set[str] = set()
    for row in rows:
        if str(_field(row, "event_kind") or "") != LIBRARY_ACTIVITY_EVENT_KIND:
            continue
        turn_id = _field(row, "turn_id")
        if turn_id != selected_turn_id:
            continue
        message_id = _field(row, "message_id")
        sequence = _field(row, "seq")
        if message_id != turn_id or type(sequence) is not int or sequence < 1:
            corrupt += 1
            continue
        try:
            event = decode_library_activity_event(_field(row, "payload_json"))
        except ValueError:
            corrupt += 1
            continue
        if event.event_id in seen_event_ids:
            corrupt += 1
            continue
        seen_event_ids.add(event.event_id)
        occurred_at = _finite_timestamp(_field(row, "step_started_at"))
        candidates.append(
            LibraryActivityRecord(
                turn_id=turn_id,
                sequence=sequence,
                occurred_at=occurred_at,
                event=event,
            )
        )
    candidates.sort(key=lambda action: (action.sequence, action.event.event_id))
    return LibraryActivityView(
        selected_turn_id=selected_turn_id,
        actions=tuple(candidates[:_VIEW_MAX_ACTIONS]),
        corrupt_row_count=min(corrupt, _VIEW_MAX_ACTIONS),
    )


def _validate_event(event: LibraryActivityEvent) -> None:
    if type(event.version) is not int or event.version != 1:
        raise ValueError("Invalid Library activity event.")
    for value in (event.event_id, event.attempt_id, event.run_id):
        if _bounded_identifier(value, required=True) != value:
            raise ValueError("Invalid Library activity event.")
    if event.parent_run_id is not None and (
        _bounded_identifier(event.parent_run_id) != event.parent_run_id
    ):
        raise ValueError("Invalid Library activity event.")
    if event.actor_kind not in ("primary", "subagent"):
        raise ValueError("Invalid Library activity event.")
    if event.library_provider not in ("direct", "rag"):
        raise ValueError("Invalid Library activity event.")
    if not _STABLE_CODE_RE.fullmatch(event.operation):
        raise ValueError("Invalid Library activity event.")
    if event.status not in ("succeeded", "empty", "blocked", "failed"):
        raise ValueError("Invalid Library activity event.")
    if (
        type(event.result_count) is not int
        or event.result_count < 0
        or event.result_count > _RESULT_COUNT_MAX
    ):
        raise ValueError("Invalid Library activity event.")
    _validate_optional_text(
        event.query_preview, LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS
    )
    if type(event.source_refs) is not tuple or len(event.source_refs) > (
        LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT
    ):
        raise ValueError("Invalid Library activity event.")
    for ref in event.source_refs:
        if not isinstance(ref, LibraryActivitySourceRef):
            raise ValueError("Invalid Library activity event.")
        _validate_required_text(ref.source_type, 40)
        _validate_required_text(ref.source_id, LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS)
        _validate_required_text(ref.title, LIBRARY_ACTIVITY_TITLE_MAX_CHARS)
    if event.error_code is not None and (
        type(event.error_code) is not str
        or _STABLE_CODE_RE.fullmatch(event.error_code) is None
    ):
        raise ValueError("Invalid Library activity event.")
    _validate_optional_text(
        event.error_summary, LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS
    )


def _validate_required_text(value: object, limit: int) -> None:
    if type(value) is not str or not value or len(value) > limit:
        raise ValueError("Invalid Library activity event.")
    if _bounded_text(value, limit) != value:
        raise ValueError("Invalid Library activity event.")


def _validate_optional_text(value: object, limit: int) -> None:
    if value is None:
        return
    _validate_required_text(value, limit)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _finite_timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return None


__all__ = [
    "LIBRARY_ACTIVITY_EVENT_KIND",
    "LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS",
    "LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES",
    "LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS",
    "LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS",
    "LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT",
    "LIBRARY_ACTIVITY_TITLE_MAX_CHARS",
    "LibraryActivityCandidate",
    "LibraryActivityContribution",
    "LibraryActivityContributionItem",
    "LibraryActivityEvent",
    "LibraryActivityRecord",
    "LibraryActivitySink",
    "LibraryActivitySourceRef",
    "LibraryActivityView",
    "decode_library_activity_event",
    "encode_library_activity_event",
    "minimize_library_activity",
    "project_library_activity",
    "redacted_library_activity_payload",
]
