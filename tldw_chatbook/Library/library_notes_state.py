"""Pure display-state contract for the Library notes canvas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

NOTES_SORT_MODES = ("newest", "oldest", "title")
_UPDATED_KEYS = ("last_modified", "updated_at", "created_at")
_EMPTY_NOTES_COPY = "No notes yet. Create one to see it here."

# The "blank" template duplicates the dedicated Blank note action (with a
# confusingly different default title), so the create view's template list
# excludes it -- the button is the one canonical empty path.
_BLANK_TEMPLATE_KEY = "blank"


@dataclass(frozen=True)
class LibraryNotesListRow:
    """One row in the Library notes canvas's list view.

    Attributes:
        note_id: The note's id.
        title: Display title (``"Untitled"`` when blank).
        age_label: Relative-age label (e.g. ``"3m"``, ``"1d"``) derived
            from the note's most recent modified/created timestamp, or
            ``""`` when no timestamp is available.
    """

    note_id: str
    title: str
    age_label: str


@dataclass(frozen=True)
class LibraryNotesListState:
    """Display state for the Library notes canvas's list view.

    Attributes:
        rows: The notes to render, already sorted/filtered by the caller.
        header_copy: The list header text (``"Notes (N)"``).
        status_copy: Filter-result status text (e.g. ``"filter: x · N
            results"``), or ``""`` when no filter is active.
        empty_copy: Empty-state copy shown when ``rows`` is empty, or
            ``""`` when there are rows to render.
    """

    rows: tuple[LibraryNotesListRow, ...]
    header_copy: str
    status_copy: str
    empty_copy: str


@dataclass(frozen=True)
class LibraryNoteEditorState:
    """Display state for the Library notes canvas's in-canvas editor.

    Attributes:
        note_id: The open note's id, or ``""`` when there is no note.
        title: The note's title.
        content: The note's body text.
        keywords_text: The note's keywords as a single comma-separated
            string (the editor's keywords ``Input`` value).
        version: The note's optimistic-lock version, or ``None`` when
            unknown/not yet saved.
        meta_line: The rendered Created/Modified/version (and, while
            saving, autosave-status) line.
        has_note: ``False`` for the placeholder "no note open" state;
            ``True`` once a real note has been loaded.
    """

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
    """Build the Library notes canvas's list-view display state.

    Records missing a mapping shape or an ``id`` are silently dropped
    rather than raising, matching the rest of this module's
    degrade-don't-crash behavior for malformed source records.

    Args:
        records: The notes to render (already sorted/filtered by the
            caller), or ``None``.
        filter_note: The active filter text, used only to render the
            result-count status copy; ``""`` when no filter is active.
        now: Reference time for relative-age labels; defaults to the
            current UTC time.

    Returns:
        The list view's display state.
    """
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
    """Cycle to the next notes sort mode in ``NOTES_SORT_MODES`` order.

    An unknown ``mode`` wraps around to the first mode rather than
    raising, so a stale/corrupt persisted sort preference degrades
    gracefully instead of crashing the sort button.

    Args:
        mode: The current sort mode.

    Returns:
        The next mode in ``NOTES_SORT_MODES`` (wrapping past the end).
    """
    try:
        index = NOTES_SORT_MODES.index(mode)
    except ValueError:
        return NOTES_SORT_MODES[0]
    return NOTES_SORT_MODES[(index + 1) % len(NOTES_SORT_MODES)]


def sort_notes_records(
    records: Sequence[Mapping[str, Any]], mode: str
) -> list[Mapping[str, Any]]:
    """Sort note records for the list view per ``mode``.

    Non-mapping records are dropped rather than raising. ``"title"`` sorts
    case-insensitively ascending; any other mode (``"newest"``/``"oldest"``)
    sorts by the record's most recent updated/created timestamp, newest
    first unless ``mode == "oldest"``.

    Args:
        records: The note records to sort.
        mode: One of ``NOTES_SORT_MODES``.

    Returns:
        A new, sorted list of the mapping records.
    """
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
    """Build the note editor's display state from a note detail mapping.

    Args:
        detail: A note detail mapping (the raw notes row, optionally
            enriched with a ``keywords`` list of keyword dicts/strings), or
            None/non-mapping when no note is loaded. Tolerated to have
            missing/None fields.
        now: Reference time for the Created/Modified relative ages;
            defaults to the current UTC time.

    Returns:
        Immutable editor state: field values, optimistic-locking version,
        the muted meta line, and ``has_note`` (False for empty input, which
        yields an all-blank state).
    """
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


def build_note_export_content(
    title: str,
    content: str,
    keywords_text: str,
    note_id: Any,
    export_format: str,
    *,
    now: datetime | None = None,
) -> str:
    """Render a note's export text, mirroring ``notes_screen._build_export_content``.

    Args:
        title: The note's current (possibly unsaved) title. Blank/whitespace
            falls back to ``"Untitled Note"``.
        content: The note's current (possibly unsaved) body text.
        keywords_text: The note's keywords as a single comma-separated
            string (the editor's keywords ``Input`` value).
        note_id: The note's id, interpolated as-is (``str()``-coerced by
            the format strings below).
        export_format: ``"markdown"`` for the frontmatter + ``# title``
            shape; any other value renders the plain-text shape.
        now: Timestamp to stamp the export with. Defaults to
            ``datetime.now()`` (naive, local time) -- matching the
            original's un-timezoned stamp -- so callers can pin it in tests.

    Returns:
        The rendered export text.
    """
    current_title = (title or "").strip() or "Untitled Note"
    timestamp = (now if now is not None else datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    if export_format == "markdown":
        return (
            f"---\n"
            f"title: {current_title}\n"
            f"date: {timestamp}\n"
            f"keywords: {keywords_text}\n"
            f"note_id: {note_id}\n"
            f"---\n\n"
            f"# {current_title}\n\n"
            f"{content}"
        )
    return (
        f"Title: {current_title}\n"
        f"Date: {timestamp}\n"
        f"Keywords: {keywords_text}\n"
        f"Note ID: {note_id}\n\n"
        f"{'=' * 50}\n\n"
        f"{content}"
    )


def notes_autosave_status_text(state: str, *, word_count: int) -> str:
    """Render the note editor meta line's word-count + autosave-status suffix.

    Args:
        state: The autosave state (``"idle"``/``"saving"``/``"saved"``/
            ``"conflict"``/``"error"``). Unrecognized values render no
            suffix.
        word_count: The note body's current word count.

    Returns:
        Text like ``"12 words · saved"`` (``"1 word"`` singular).
    """
    base = f"{word_count} words" if word_count != 1 else "1 word"
    suffix = {
        "saving": " · saving…",
        "saved": " · saved",
        "conflict": " · changed elsewhere",
        "error": " · save failed",
    }.get(state, "")
    return f"{base}{suffix}"


def resolve_note_template_placeholders(text: str, *, now: datetime | None = None) -> str:
    """Resolve ``{date}``/``{time}``/``{datetime}`` placeholders in template text.

    Mirrors the standalone Notes screen's substitution (same placeholder
    names, same ``strftime`` formats). Resolution is per-key (a plain
    ``str.replace`` for each of the three known placeholders), so a
    template that also contains an unknown ``{placeholder}`` or a stray
    brace still gets every *known* placeholder substituted -- only the
    unrecognized text is left literal, rather than the whole template
    degrading to raw, unsubstituted text.

    Args:
        text: Template title or content text.
        now: Reference time; defaults to the current local time (matching
            the standalone screen's naive-local timestamps).

    Returns:
        The text with every known placeholder substituted; any unknown
        ``{placeholder}`` or stray brace is left unchanged.
    """
    reference_now = now if now is not None else datetime.now()
    values = {
        "date": reference_now.strftime("%Y-%m-%d"),
        "time": reference_now.strftime("%H:%M"),
        "datetime": reference_now.strftime("%Y-%m-%d %H:%M"),
    }
    resolved = text
    for key, value in values.items():
        resolved = resolved.replace(f"{{{key}}}", value)
    return resolved


def note_template_keywords(template: Any) -> tuple[str, ...]:
    """Parse a note template's ``keywords`` field into a clean tuple.

    The bundled templates carry comma-separated strings ("meeting, notes");
    a list/tuple value is tolerated too. Anything else yields no keywords.

    Args:
        template: The raw ``NOTE_TEMPLATES[key]`` value.

    Returns:
        Stripped, non-empty keyword strings in template order.
    """
    if not isinstance(template, Mapping):
        return ()
    raw = template.get("keywords")
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, Sequence):
        parts = [str(item) for item in raw]
    else:
        return ()
    return tuple(part.strip() for part in parts if part and str(part).strip())


def _note_template_label(key: str, template: Any) -> str:
    """Human label for a template row (pure mirror of the workbench helper).

    Strips the redundant "template" wording from descriptions ("Template
    for meeting notes" -> "Meeting notes") exactly like
    ``notes_workbench_panes.template_display_label`` -- replicated here so
    the pure module stays Textual-free (the workbench module imports
    Textual and is slated for deletion with the standalone screen).
    """
    raw = ""
    if isinstance(template, Mapping):
        raw = str(template.get("description") or template.get("title") or "")
    raw = raw or str(key).replace("_", " ")
    label = re.sub(r"^\s*templates?\s+for\s+", "", raw, flags=re.IGNORECASE)
    label = re.sub(r"\s*\btemplates?\b\s*$", "", label, flags=re.IGNORECASE)
    label = label.strip(" -–:") or str(key).replace("_", " ")
    return label[:1].upper() + label[1:]


@dataclass(frozen=True)
class LibraryNoteTemplateRow:
    """One template row in the create view's "From a template" list.

    Attributes:
        template_key: The ``NOTE_TEMPLATES`` key the row creates from.
        label: Cleaned human label ("Meeting notes").
        resolved_title: The title the created note will actually get, with
            date/time placeholders already substituted -- shown as the
            row's muted secondary line so the outcome is visible before
            pressing. Empty when it would just repeat the label.
    """

    template_key: str
    label: str
    resolved_title: str


def build_library_note_template_rows(
    templates: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> tuple[LibraryNoteTemplateRow, ...]:
    """Build the create view's template rows from ``NOTE_TEMPLATES``.

    Excludes the ``blank`` template (it duplicates the dedicated Blank
    note action). Rows are sorted by key for a stable order. Malformed
    (non-mapping) template values degrade to a key-derived label with no
    secondary line rather than being dropped -- the screen-side field
    resolver already degrades their creation to a blank note.

    Args:
        templates: The ``NOTE_TEMPLATES`` mapping (or None).
        now: Reference time for resolving title placeholders.

    Returns:
        Immutable, ready-to-render template rows.
    """
    rows: list[LibraryNoteTemplateRow] = []
    for key, template in sorted((templates or {}).items()):
        if str(key) == _BLANK_TEMPLATE_KEY:
            continue
        label = _note_template_label(str(key), template)
        resolved_title = ""
        if isinstance(template, Mapping):
            raw_title = template.get("title")
            if isinstance(raw_title, str) and raw_title.strip():
                resolved_title = resolve_note_template_placeholders(
                    raw_title, now=now
                ).strip()
        if resolved_title.lower() == label.lower():
            resolved_title = ""
        rows.append(
            LibraryNoteTemplateRow(
                template_key=str(key),
                label=label,
                resolved_title=resolved_title,
            )
        )
    return tuple(rows)
