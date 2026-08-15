"""Pure display-state contracts for the Library Browse ▸ Conversations canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Library.library_pager_state import (
    LibraryPagerDisplay,
    PageFreshness,
    build_library_pager_display,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)

LIBRARY_CONVERSATIONS_EMPTY_COPY = (
    "No conversations yet. Chat in Console and it appears here."
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
_SQLITE_SIGNED_INT_MAX = 2**63 - 1


@dataclass(frozen=True)
class LibraryConversationRow:
    """One selectable row in the Library Browse ▸ Conversations canvas."""

    conversation_id: str
    title: str
    secondary: str
    selected: bool = False
    checked: bool = False


@dataclass(frozen=True)
class LibraryConversationsCanvasState:
    """Pure display state for the Library Browse ▸ Conversations canvas."""

    rows: tuple[LibraryConversationRow, ...]
    status_copy: str
    empty_copy: str
    selected_id: str
    preview_lines: tuple[str, ...]
    query: str
    select_mode: bool = False
    selected_count: int = 0
    range_copy: str = ""
    page_copy: str = ""
    previous_disabled: bool = True
    next_disabled: bool = True
    loading: bool = False
    error_copy: str = ""
    pager: LibraryPagerDisplay | None = None
    selection_notice: str = ""
    actions_disabled: bool = False


@dataclass(frozen=True)
class ValidatedLibraryConversationPage:
    """Validated ordinary Conversation service page."""

    items: tuple[Mapping[str, Any], ...]
    limit: int
    offset: int
    total: int
    has_more: bool


@dataclass(frozen=True)
class _ConversationEntry:
    """Internal per-record fields used before rendering a display row."""

    conversation_id: str
    title: str
    updated_raw: str
    message_count: int | None


def _stable_conversation_identity(record: Mapping[str, Any]) -> str:
    for key in _ID_KEYS:
        if key not in record:
            continue
        value = record[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("item must have a stable conversation identity")
        return value.strip()
    raise ValueError("item must have a stable conversation identity")


def _validate_conversation_items(
    items: Sequence[object],
) -> tuple[Mapping[str, Any], ...]:
    validated: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("item must be a mapping with a stable conversation identity")
        identity = _stable_conversation_identity(item)
        if identity in identities:
            raise ValueError("page contains a duplicate stable conversation identity")
        identities.add(identity)
        validated.append(item)
    return tuple(validated)


def _validated_page_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > _SQLITE_SIGNED_INT_MAX:
        raise ValueError(f"{name} is outside SQLite signed-integer bounds")
    return value


def validate_library_conversation_page(
    response: Mapping[str, Any],
    *,
    requested_limit: int,
    requested_offset: int,
) -> ValidatedLibraryConversationPage:
    """Validate one ordinary Conversation service page before rendering."""

    expected_limit = _validated_page_integer(
        requested_limit,
        "requested_limit",
        minimum=1,
    )
    expected_offset = _validated_page_integer(
        requested_offset,
        "requested_offset",
        minimum=0,
    )
    if not isinstance(response, Mapping):
        raise ValueError("conversation page response must be a mapping")

    raw_items = response.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("conversation page items must be a list")
    items = _validate_conversation_items(raw_items)

    pagination = response.get("pagination")
    if not isinstance(pagination, Mapping):
        raise ValueError("conversation page pagination must be a mapping")
    limit = _validated_page_integer(pagination.get("limit"), "limit", minimum=1)
    offset = _validated_page_integer(pagination.get("offset"), "offset", minimum=0)
    total = _validated_page_integer(pagination.get("total"), "total", minimum=0)
    has_more = pagination.get("has_more")
    if not isinstance(has_more, bool):
        raise ValueError("has_more must be a boolean")

    if limit != expected_limit or offset != expected_offset:
        raise ValueError("conversation page coordinate echo does not match the request")
    if offset > 0 and offset >= total:
        raise ValueError("conversation page offset is out of range for its total")

    expected_count = min(limit, max(total - offset, 0))
    if len(items) != expected_count:
        raise ValueError("conversation page cardinality does not match its coordinates")
    if has_more != (offset + len(items) < total):
        raise ValueError("has_more disagrees with conversation page coordinates")

    return ValidatedLibraryConversationPage(
        items=items,
        limit=limit,
        offset=offset,
        total=total,
        has_more=has_more,
    )


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


def _secondary_text(message_count: int | None, age: str) -> str:
    if message_count is None:
        return "conversation"
    if age:
        return f"{message_count} messages - {age}"
    return f"{message_count} messages"


def build_library_conversations_state(
    records: Sequence[Mapping[str, Any]],
    *,
    query: str = "",
    selected_id: str = "",
    now: datetime | None = None,
    page: int = 1,
    requested_page: int | None = None,
    page_size: int = 20,
    total_count: int | None = None,
    total_known: bool = True,
    has_more: bool = False,
    freshness: PageFreshness | None = None,
    stale_copy: str = "",
    loading: bool = False,
    error_copy: str = "",
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
    selection_notice: str = "",
) -> LibraryConversationsCanvasState:
    """Build the Library Browse ▸ Conversations canvas display state.

    Args:
        records: Conversation records from the screen's conversation service.
            This is one already-filtered, already-paged result. Every record
            needs a unique stable identity; other missing fields use display
            fallbacks.
        query: Submitted query used for display copy.
        selected_id: Requested selected conversation id; falls back to the
            first displayed row when absent from the supplied page.
        now: Reference time for relative age labels; defaults to current UTC time.
        page: One-based page number.
        requested_page: One-based page targeted by the current or last request.
        page_size: Maximum service page size.
        total_count: Total matching records when known.
        total_known: Whether total_count is authoritative.
        has_more: Legacy service flag retained for caller compatibility; exact
            boundaries come from the shared pager display and total.
        freshness: Whether page metadata is absent, authoritative, or stale.
        stale_copy: Source-owned reason for retained stale rows.
        loading: Whether a page request is in flight.
        error_copy: Recoverable page-load error copy.
        selection_notice: Source-owned notice for cleared page selection.

    Returns:
        Immutable canvas state: rows, status/empty copy, selection, and
        preview lines for the selected row.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)
    normalized_query = str(query or "").strip()
    requested_page = page if requested_page is None else requested_page

    entries: list[_ConversationEntry] = []
    for record in _validate_conversation_items(records):
        conversation_id = _stable_conversation_identity(record)
        updated_raw = _first_present_text(record, _UPDATED_KEYS)
        entries.append(
            _ConversationEntry(
                conversation_id=conversation_id,
                title=_record_title(record),
                updated_raw=updated_raw,
                message_count=_record_message_count(record),
            )
        )

    resolved_freshness: PageFreshness
    if freshness is not None:
        resolved_freshness = freshness
    elif total_known and total_count is not None:
        resolved_freshness = "fresh"
    elif entries:
        resolved_freshness = "stale"
    else:
        resolved_freshness = "uninitialized"
    resolved_stale_copy = stale_copy
    if resolved_freshness == "stale" and not resolved_stale_copy:
        resolved_stale_copy = "List may be out of date"

    pager = build_library_pager_display(
        applied_page=None if resolved_freshness == "uninitialized" else page,
        requested_page=requested_page,
        page_size=page_size,
        row_count=len(entries),
        total=total_count if resolved_freshness == "fresh" else None,
        freshness=resolved_freshness,
        loading=loading,
        error_copy=error_copy,
        stale_copy=resolved_stale_copy,
    )

    resolved_selected_id = str(selected_id or "")
    displayed_ids = {entry.conversation_id for entry in entries}
    if resolved_selected_id not in displayed_ids:
        resolved_selected_id = entries[0].conversation_id if entries else ""

    rows = tuple(
        LibraryConversationRow(
            conversation_id=entry.conversation_id,
            title=entry.title,
            secondary=_secondary_text(
                entry.message_count,
                format_console_relative_age(entry.updated_raw, now=reference_now),
            ),
            selected=entry.conversation_id == resolved_selected_id,
            checked=entry.conversation_id in selected_ids,
        )
        for entry in entries
    )
    selected_count = sum(1 for r in rows if r.checked)

    if pager.status_copy:
        status_copy = pager.status_copy
    elif normalized_query and resolved_freshness == "fresh":
        match_count = pager.title_count
        suffix = "match" if match_count == 1 else "matches"
        status_copy = f"{match_count} {suffix} for '{normalized_query}'"
    else:
        status_copy = ""

    if pager.status_copy or loading or rows or resolved_freshness != "fresh":
        empty_copy = ""
    elif normalized_query:
        empty_copy = f"No conversations match '{normalized_query}'."
    else:
        empty_copy = LIBRARY_CONVERSATIONS_EMPTY_COPY

    selected_entry = next(
        (
            entry
            for entry in entries
            if entry.conversation_id == resolved_selected_id
        ),
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
        range_copy=pager.range_copy,
        page_copy=pager.page_copy,
        previous_disabled=pager.previous_disabled,
        next_disabled=pager.next_disabled,
        select_mode=select_mode,
        selected_count=selected_count,
        loading=loading,
        error_copy=error_copy,
        pager=pager,
        selection_notice=selection_notice,
        actions_disabled=resolved_freshness == "stale",
    )
