"""Pure display-state contracts for the Library Browse ▸ Media canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

LIBRARY_MEDIA_EMPTY_COPY = "No media in your Library yet. Ingest something to see it here."

_ID_KEYS = ("id", "media_id", "uuid")
_TYPE_KEYS = ("type", "media_type")
_UPDATED_KEYS = ("last_modified", "ingestion_date", "date", "updated_at")


@dataclass(frozen=True)
class LibraryMediaRow:
    """One selectable row in the Library Browse ▸ Media canvas."""

    media_id: str
    title: str
    media_type: str
    secondary: str
    selected: bool = False
    checked: bool = False


@dataclass(frozen=True)
class LibraryMediaCanvasState:
    """Pure display state for the Library Browse ▸ Media canvas."""

    rows: tuple[LibraryMediaRow, ...]
    type_options: tuple[str, ...]
    active_type: str
    status_copy: str
    empty_copy: str
    selected_id: str
    preview_lines: tuple[str, ...]
    count: int
    select_mode: bool = False
    selected_count: int = 0


@dataclass(frozen=True)
class _MediaEntry:
    """Internal per-record fields used before rendering a display row."""

    media_id: str
    title: str
    media_type: str
    updated_raw: str
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
        return "Untitled media"
    text = str(value).strip()
    return text or "Untitled media"


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


def _secondary_text(media_type: str, age: str) -> str:
    """Return secondary display text: '{type} · {age}' or fallback.

    Rules:
    - If type and age both present: 'type · age'
    - If only type (no age): 'type'
    - If no type: 'media' (regardless of age)
    """
    has_type = bool(media_type)
    has_age = bool(age)

    if has_type and has_age:
        return f"{media_type} · {age}"
    elif has_type:
        return media_type
    else:
        # When no type, return 'media' regardless of age
        return "media"


def _sort_key(entry: _MediaEntry) -> tuple[int, float]:
    if entry.sort_timestamp is None:
        return (1, 0.0)
    return (0, -entry.sort_timestamp.timestamp())


def build_library_media_state(
    records: Sequence[Mapping[str, Any]],
    *,
    active_type: str = "All",
    selected_id: str = "",
    now: datetime | None = None,
    limit: int = 75,
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
) -> LibraryMediaCanvasState:
    """Build the Library Browse ▸ Media canvas display state.

    Args:
        records: Media records from the screen's media service.
            Tolerated to have missing/None fields.
        active_type: Filter rows to this media type, or "All" for no filter.
        selected_id: Requested selected media id; falls back to the
            first displayed row when absent from the filtered/limited rows.
        now: Reference time for relative age labels; defaults to current UTC time.
        limit: Maximum number of rows to display after sorting and filtering.

    Returns:
        Immutable canvas state: rows, type options, active type, status/empty copy,
        selection, preview lines, and total count.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)

    entries: list[_MediaEntry] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        media_id = _first_present_text(record, _ID_KEYS)
        if not media_id:
            continue
        updated_raw = _first_present_text(record, _UPDATED_KEYS)
        media_type = _first_present_text(record, _TYPE_KEYS)
        entries.append(
            _MediaEntry(
                media_id=media_id,
                title=_record_title(record),
                media_type=media_type,
                updated_raw=updated_raw,
                sort_timestamp=_parse_timestamp(updated_raw),
            )
        )

    # Calculate total count before filtering
    total_count = len(entries)

    # Filter by active_type if not "All"
    if active_type != "All":
        filtered_entries = [e for e in entries if e.media_type == active_type]
    else:
        filtered_entries = entries

    # Sort by updated timestamp desc (missing last)
    filtered_entries.sort(key=_sort_key)

    # Apply limit
    limited_entries = filtered_entries[: max(0, limit)]

    # Resolve selected_id
    resolved_selected_id = str(selected_id or "")
    displayed_ids = {entry.media_id for entry in limited_entries}
    if resolved_selected_id not in displayed_ids:
        resolved_selected_id = limited_entries[0].media_id if limited_entries else ""

    # Build rows
    rows = tuple(
        LibraryMediaRow(
            media_id=entry.media_id,
            title=entry.title,
            media_type=entry.media_type,
            secondary=_secondary_text(
                entry.media_type,
                format_console_relative_age(entry.updated_raw, now=reference_now),
            ),
            selected=entry.media_id == resolved_selected_id,
            checked=entry.media_id in selected_ids,
        )
        for entry in limited_entries
    )
    selected_count = sum(1 for r in rows if r.checked)

    # Build type_options: ("All",) + sorted distinct non-empty types
    distinct_types = {entry.media_type for entry in entries if entry.media_type}
    if active_type != "All":
        distinct_types.add(active_type)
    type_options = ("All",) + tuple(sorted(distinct_types))

    # Build status_copy and empty_copy
    if active_type != "All":
        # When filtering by type, report the count of all matches (pre-limit).
        status_copy = f"{len(filtered_entries)} of {total_count} · type: {active_type}"
        if not rows:
            empty_copy = f"No media of type '{active_type}'."
        else:
            empty_copy = ""
    else:
        # When showing all, no status copy
        status_copy = ""
        if not rows:
            empty_copy = LIBRARY_MEDIA_EMPTY_COPY
        else:
            empty_copy = ""

    # Build preview_lines for selected row
    selected_entry = next(
        (entry for entry in limited_entries if entry.media_id == resolved_selected_id),
        None,
    )
    if selected_entry is None:
        preview_lines: tuple[str, ...] = ()
    else:
        age = format_console_relative_age(selected_entry.updated_raw, now=reference_now)
        type_text = selected_entry.media_type or "unknown"
        age_text = age or "unknown"
        preview_lines = (
            selected_entry.title,
            f"Type: {type_text}",
            f"Updated: {age_text}",
        )

    return LibraryMediaCanvasState(
        rows=rows,
        type_options=type_options,
        active_type=active_type,
        status_copy=status_copy,
        empty_copy=empty_copy,
        selected_id=resolved_selected_id,
        preview_lines=preview_lines,
        count=total_count,
        select_mode=select_mode,
        selected_count=selected_count,
    )
