"""Bounded, device-local review records for assistant Library operations."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from tldw_chatbook.Chat.trajectory import contains_local_path, redact_local_paths
from tldw_chatbook.Utils.log_sanitizer import redact_log_line

LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS = 160
LIBRARY_ACTIVITY_TITLE_MAX_CHARS = 160
LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS = 240
LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS = 200
LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT = 8
LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES = 8192

_OPAQUE_ID_MAX_CHARS = 200
_STABLE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
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
    return event


__all__ = [
    "LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS",
    "LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES",
    "LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS",
    "LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS",
    "LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT",
    "LIBRARY_ACTIVITY_TITLE_MAX_CHARS",
    "LibraryActivityCandidate",
    "LibraryActivityEvent",
    "LibraryActivitySink",
    "LibraryActivitySourceRef",
    "minimize_library_activity",
]
