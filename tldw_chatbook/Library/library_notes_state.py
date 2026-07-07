"""Pure display-state contract for the Library notes canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

NOTES_SORT_MODES = ("newest", "oldest", "title")
_UPDATED_KEYS = ("last_modified", "updated_at", "created_at")
_EMPTY_NOTES_COPY = "No notes yet. Create one to see it here."


@dataclass(frozen=True)
class LibraryNotesListRow:
    note_id: str
    title: str
    age_label: str


@dataclass(frozen=True)
class LibraryNotesListState:
    rows: tuple[LibraryNotesListRow, ...]
    header_copy: str
    status_copy: str
    empty_copy: str


@dataclass(frozen=True)
class LibraryNoteEditorState:
    note_id: str
    title: str
    content: str
    keywords_text: str
    version: int | None
    meta_line: str
    has_note: bool


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _updated_raw(record: Mapping[str, Any]) -> str:
    for key in _UPDATED_KEYS:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _row(record: Mapping[str, Any], *, now: datetime) -> LibraryNotesListRow:
    raw = _updated_raw(record)
    return LibraryNotesListRow(
        note_id=_text(record.get("id")),
        title=_text(record.get("title")) or "Untitled",
        age_label=format_console_relative_age(raw, now=now) if raw else "",
    )


def build_library_notes_list_state(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    filter_note: str = "",
    now: datetime | None = None,
) -> LibraryNotesListState:
    reference_now = now if now is not None else datetime.now(timezone.utc)
    rows = tuple(
        _row(record, now=reference_now)
        for record in (records or ())
        if isinstance(record, Mapping) and _text(record.get("id"))
    )
    status_copy = ""
    if filter_note:
        noun = "result" if len(rows) == 1 else "results"
        status_copy = f"filter: {filter_note} · {len(rows)} {noun}"
    return LibraryNotesListState(
        rows=rows,
        header_copy=f"Notes ({len(rows)})",
        status_copy=status_copy,
        empty_copy="" if rows else _EMPTY_NOTES_COPY,
    )


def next_notes_sort_mode(mode: str) -> str:
    try:
        index = NOTES_SORT_MODES.index(mode)
    except ValueError:
        return NOTES_SORT_MODES[0]
    return NOTES_SORT_MODES[(index + 1) % len(NOTES_SORT_MODES)]


def sort_notes_records(
    records: Sequence[Mapping[str, Any]], mode: str
) -> list[Mapping[str, Any]]:
    items = [r for r in records if isinstance(r, Mapping)]
    if mode == "title":
        return sorted(items, key=lambda r: _text(r.get("title")).lower())
    reverse = mode != "oldest"
    return sorted(items, key=_updated_raw, reverse=reverse)


def _keywords_text(detail: Mapping[str, Any]) -> str:
    keywords = detail.get("keywords")
    if isinstance(keywords, str):
        return keywords.strip()
    if isinstance(keywords, Sequence):
        items = []
        for item in keywords:
            if isinstance(item, Mapping):
                item = item.get("keyword") or item.get("text") or item.get("label")
            text = _text(item)
            if text:
                items.append(text)
        return ", ".join(items)
    return ""


def build_library_note_editor_state(
    detail: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> LibraryNoteEditorState:
    if not isinstance(detail, Mapping) or not _text(detail.get("id")):
        return LibraryNoteEditorState(
            note_id="", title="", content="", keywords_text="",
            version=None, meta_line="", has_note=False,
        )
    reference_now = now if now is not None else datetime.now(timezone.utc)
    version_raw = detail.get("version")
    try:
        version: int | None = int(version_raw) if version_raw is not None else None
    except (TypeError, ValueError):
        version = None
    parts: list[str] = []
    created = _text(detail.get("created_at"))
    if created:
        parts.append(f"Created {format_console_relative_age(created, now=reference_now)}")
    modified = _updated_raw(detail)
    if modified:
        parts.append(f"Modified {format_console_relative_age(modified, now=reference_now)}")
    if version is not None:
        parts.append(f"v{version}")
    return LibraryNoteEditorState(
        note_id=_text(detail.get("id")),
        title=_text(detail.get("title")),
        content=str(detail.get("content") or ""),
        keywords_text=_keywords_text(detail),
        version=version,
        meta_line=" · ".join(parts),
        has_note=True,
    )


def notes_autosave_status_text(state: str, *, word_count: int) -> str:
    base = f"{word_count} words" if word_count != 1 else "1 word"
    suffix = {
        "saving": " · saving…",
        "saved": " · saved",
        "conflict": " · changed elsewhere",
        "error": " · save failed",
    }.get(state, "")
    return f"{base}{suffix}"
