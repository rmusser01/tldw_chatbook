"""Pure display-state contracts for the Library Browse ▸ Conversations canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

LIBRARY_CONVERSATIONS_EMPTY_COPY = (
    "No saved conversations yet. Save a Console chat and it appears here."
)

_ID_KEYS = ("id", "conversation_id", "uuid")
_UPDATED_KEYS = ("updated_at", "last_updated", "last_modified", "updated")
_MESSAGE_COUNT_KEYS = (
    "message_count",
    "messages_count",
    "messageCount",
    "message_total",
    "messages_total",
)


@dataclass(frozen=True)
class LibraryConversationRow:
    """One selectable row in the Library Browse ▸ Conversations canvas."""

    conversation_id: str
    title: str
    secondary: str
    selected: bool = False


@dataclass(frozen=True)
class LibraryConversationsCanvasState:
    """Pure display state for the Library Browse ▸ Conversations canvas."""

    rows: tuple[LibraryConversationRow, ...]
    status_copy: str
    empty_copy: str
    selected_id: str
    preview_lines: tuple[str, ...]
    query: str


@dataclass(frozen=True)
class _ConversationEntry:
    """Internal per-record fields used before rendering a display row."""

    conversation_id: str
    title: str
    updated_raw: str
    message_count: int | None
    sort_timestamp: datetime | None


def _first_present_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _record_title(record: Mapping[str, Any]) -> str:
    value = record.get("title")
    if value is None:
        return "Untitled conversation"
    text = str(value).strip()
    return text or "Untitled conversation"


def _record_message_count(record: Mapping[str, Any]) -> int | None:
    # Mirrors `LibraryScreen._conversation_message_count_label` key handling.
    for key in _MESSAGE_COUNT_KEYS:
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _parse_timestamp(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _secondary_text(message_count: int | None, age: str) -> str:
    if message_count is None:
        return "conversation"
    if age:
        return f"{message_count} messages - {age}"
    return f"{message_count} messages"


def _sort_key(entry: _ConversationEntry) -> tuple[int, float]:
    if entry.sort_timestamp is None:
        return (1, 0.0)
    return (0, -entry.sort_timestamp.timestamp())


def build_library_conversations_state(
    records: Sequence[Mapping[str, Any]],
    *,
    query: str = "",
    selected_id: str = "",
    now: datetime | None = None,
    limit: int = 75,
) -> LibraryConversationsCanvasState:
    """Build the Library Browse ▸ Conversations canvas display state.

    Args:
        records: Conversation records from the screen's conversation service.
            Tolerated to have missing/None fields.
        query: Case-insensitive substring filter applied to titles.
        selected_id: Requested selected conversation id; falls back to the
            first displayed row when absent from the filtered/limited rows.
        now: Reference time for relative age labels; defaults to current UTC time.
        limit: Maximum number of rows to display after sorting and filtering.

    Returns:
        Immutable canvas state: rows, status/empty copy, selection, and
        preview lines for the selected row.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)
    normalized_query = str(query or "").strip()

    entries: list[_ConversationEntry] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        conversation_id = _first_present_text(record, _ID_KEYS)
        if not conversation_id:
            continue
        updated_raw = _first_present_text(record, _UPDATED_KEYS)
        entries.append(
            _ConversationEntry(
                conversation_id=conversation_id,
                title=_record_title(record),
                updated_raw=updated_raw,
                message_count=_record_message_count(record),
                sort_timestamp=_parse_timestamp(updated_raw),
            )
        )

    if normalized_query:
        lowered_query = normalized_query.lower()
        entries = [entry for entry in entries if lowered_query in entry.title.lower()]

    entries.sort(key=_sort_key)

    limited_entries = entries[: max(0, limit)]

    resolved_selected_id = str(selected_id or "")
    displayed_ids = {entry.conversation_id for entry in limited_entries}
    if resolved_selected_id not in displayed_ids:
        resolved_selected_id = limited_entries[0].conversation_id if limited_entries else ""

    rows = tuple(
        LibraryConversationRow(
            conversation_id=entry.conversation_id,
            title=entry.title,
            secondary=_secondary_text(
                entry.message_count,
                format_console_relative_age(entry.updated_raw, now=reference_now),
            ),
            selected=entry.conversation_id == resolved_selected_id,
        )
        for entry in limited_entries
    )

    if normalized_query:
        # match_count is from the filtered-but-unlimited entries list, not the limited rows
        match_count = len(entries)
        suffix = "match" if match_count == 1 else "matches"
        status_copy = f"{match_count} {suffix} for '{normalized_query}'"
    else:
        status_copy = ""

    if rows:
        empty_copy = ""
    elif normalized_query:
        empty_copy = f"No conversations match '{normalized_query}'."
    else:
        empty_copy = LIBRARY_CONVERSATIONS_EMPTY_COPY

    selected_entry = next(
        (entry for entry in limited_entries if entry.conversation_id == resolved_selected_id),
        None,
    )
    if selected_entry is None:
        preview_lines: tuple[str, ...] = ()
    else:
        age = format_console_relative_age(selected_entry.updated_raw, now=reference_now)
        count_text = (
            str(selected_entry.message_count)
            if selected_entry.message_count is not None
            else "unknown"
        )
        preview_lines = (
            selected_entry.title,
            f"Messages: {count_text}",
            f"Updated: {age or 'unknown'}",
        )

    return LibraryConversationsCanvasState(
        rows=rows,
        status_copy=status_copy,
        empty_copy=empty_copy,
        selected_id=resolved_selected_id,
        preview_lines=preview_lines,
        query=normalized_query,
    )
