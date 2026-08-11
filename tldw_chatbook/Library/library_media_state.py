"""Pure display-state contracts for the Library Browse ▸ Media canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)

LIBRARY_MEDIA_EMPTY_COPY = (
    "No media in your Library yet. Import something to see it here."
)

# task-4025: the Trash view's honest empty state -- present tense, no
# promise of anything beyond what the surface does (items land here on
# delete and leave on restore).
LIBRARY_MEDIA_TRASH_EMPTY_COPY = (
    "Trash is empty. Items you delete from Media land here."
)

# task-4025 (F-018): every disabled Library action says why -- the Trash
# view's "Restore" action has two honest disabled reasons (still loading
# vs. genuinely nothing there) plus its enabled description, mirroring the
# Export/Delete-selected tooltip pairs in library_shell_state.py.
LIBRARY_MEDIA_TRASH_RESTORE_TOOLTIP = "Restore the selected item to your Library."
LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_EMPTY_TOOLTIP = "Nothing in Trash to restore."
LIBRARY_MEDIA_TRASH_RESTORE_DISABLED_LOADING_TOOLTIP = "Trash is still loading."

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
    # task-2853 AC3: True while the bulk-delete confirmation ("Delete N
    # selected items? ..." + Delete/Cancel) should render in place of the
    # normal select-mode toolbar row.
    confirming_bulk_delete: bool = False
    # task-4022 AC2: count of items from the most recently completed bulk
    # delete, shown as a "✓ deleted · N items" receipt with Undo/Dismiss
    # until acted on or replaced by a newer bulk-delete action. 0 means no
    # receipt to show (the normal state).
    delete_receipt_count: int = 0


@dataclass(frozen=True)
class LibraryMediaTrashRow:
    """One selectable row in the Library media Trash view (task-4025)."""

    media_id: str
    title: str
    media_type: str
    secondary: str
    selected: bool = False


@dataclass(frozen=True)
class LibraryMediaTrashState:
    """Pure display state for the Library media Trash view (task-4025).

    Attributes:
        rows: Trashed items in the seam's own trash_date-DESC order.
        count: Total trashed items reported by the seam (may exceed
            ``len(rows)`` when the fetch page was smaller than the trash).
        status_copy: Honest truncation line ("showing X of N") when the
            fetched rows undercount the total, else "".
        empty_copy: The empty-Trash copy -- only when the fetch has landed
            (``loading`` False), succeeded (``error`` empty), and found
            nothing; "" otherwise so a loading/error state never claims
            "Trash is empty".
        selected_id: Resolved selected row id ("" when there are no rows).
        loading: True while the trash fetch has not landed yet.
        error: Fetch-failure copy, "" on success.
        notice: Restore feedback line (e.g. "Restored 'Title'."), "" when
            nothing to report. Feedback only -- never a receipt: ADR-055's
            receipts accompany destruction, and restore is recovery.
    """

    rows: tuple[LibraryMediaTrashRow, ...]
    count: int
    status_copy: str
    empty_copy: str
    selected_id: str
    loading: bool = False
    error: str = ""
    notice: str = ""


def build_library_media_trash_state(
    records: Sequence[Any] | None,
    *,
    total: int = 0,
    selected_id: str = "",
    now: datetime | None = None,
    loading: bool | None = None,
    error: str = "",
    notice: str = "",
) -> LibraryMediaTrashState:
    """Build the Library media Trash view display state (task-4025).

    Args:
        records: Trash records from ``list_media_trash`` (id/title/type,
            plus ``trash_date`` where the seam selected it). ``None`` means
            the fetch has not landed yet (loading). Row order is preserved
            as given -- the seam already orders by ``trash_date DESC``, and
            re-sorting here would silently disagree with it for rows whose
            ``trash_date`` did not survive the projection.
        total: The seam's total trashed-item count (may exceed the fetched
            rows).
        selected_id: Requested selected media id; falls back to the first
            row when absent.
        now: Reference time for the "trashed <age>" secondary; defaults to
            current UTC time.
        loading: Explicit loading override; defaults to ``records is None``.
        error: Fetch-failure copy to surface instead of rows.
        notice: Restore feedback line to pass through.

    Returns:
        Immutable Trash view state.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)
    resolved_loading = (records is None) if loading is None else bool(loading)

    entries: list[tuple[str, str, str, str]] = []
    for record in records or ():
        if not isinstance(record, Mapping):
            continue
        media_id = _first_present_text(record, _ID_KEYS)
        if not media_id:
            continue
        entries.append(
            (
                media_id,
                _record_title(record),
                _first_present_text(record, _TYPE_KEYS),
                _first_present_text(record, ("trash_date",)),
            )
        )

    resolved_selected_id = str(selected_id or "")
    entry_ids = {media_id for media_id, _, _, _ in entries}
    if resolved_selected_id not in entry_ids:
        resolved_selected_id = entries[0][0] if entries else ""

    rows = []
    for media_id, title, media_type, trash_date in entries:
        age = format_console_relative_age(trash_date, now=reference_now)
        # The list's own secondary vocabulary ("{type} · {age}" / "{type}"
        # / "media"), with the age labelled for what it is here: when the
        # item was trashed, not when it was updated.
        trashed_age = f"trashed {age}" if age else ""
        rows.append(
            LibraryMediaTrashRow(
                media_id=media_id,
                title=title,
                media_type=media_type,
                secondary=_secondary_text(media_type, trashed_age),
                selected=media_id == resolved_selected_id,
            )
        )

    resolved_total = max(int(total or 0), len(rows))
    status_copy = (
        f"showing {len(rows)} of {resolved_total}"
        if rows and resolved_total > len(rows)
        else ""
    )
    empty_copy = (
        LIBRARY_MEDIA_TRASH_EMPTY_COPY
        if not rows and not resolved_loading and not error
        else ""
    )

    return LibraryMediaTrashState(
        rows=tuple(rows),
        count=resolved_total,
        status_copy=status_copy,
        empty_copy=empty_copy,
        selected_id=resolved_selected_id,
        loading=resolved_loading,
        error=str(error or ""),
        notice=str(notice or ""),
    )


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
    confirming_bulk_delete: bool = False,
    delete_receipt_count: int = 0,
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
        delete_receipt_count: Count of items from the most recently
            completed bulk delete, rendered as a "✓ deleted · N items"
            receipt with Undo/Dismiss until acted on or replaced by a
            newer bulk-delete action. 0 (the default) means no receipt to
            show.

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
        confirming_bulk_delete=confirming_bulk_delete,
        delete_receipt_count=max(0, delete_receipt_count),
    )
