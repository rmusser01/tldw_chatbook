"""Pure display-state contract for the Library notes canvas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from rich.cells import get_character_cell_size

from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)

NOTES_SORT_MODES = ("newest", "oldest", "title")
_UPDATED_KEYS = ("last_modified", "updated_at", "created_at")
_EMPTY_NOTES_COPY = "No notes yet. Create your first note."

LibraryNotesStage = Literal["rail", "notes"]
LibraryNotesRegion = Literal[
    "", "navigator", "editor", "preview", "context", "create", "sync"
]

DATABASE_NOTE_TITLE_MAX_CHARS = 300
DATABASE_NOTE_BODY_MAX_CHARS = 2_000_000
DATABASE_NOTE_KEYWORD_MAX_CHARS = 100

# The "blank" template duplicates the dedicated Blank note action (with a
# confusingly different default title), so the create view's template list
# excludes it -- the button is the one canonical empty path.
_BLANK_TEMPLATE_KEY = "blank"


@dataclass(frozen=True)
class NormalizedDatabaseNote:
    """One coherent persisted Database Note detail returned by a session port.

    Attributes:
        note_id: Stable Database Note identity.
        title: Exact persisted title.
        body: Exact persisted body.
        keywords: Semantic keyword tokens in persisted order.
        version: Current optimistic-lock version.
        created_at: Persisted creation timestamp text.
        modified_at: Persisted modification timestamp text.
    """

    note_id: str
    title: str
    body: str
    keywords: tuple[str, ...]
    version: int
    created_at: str
    modified_at: str


@dataclass(frozen=True)
class DatabaseNoteDraft:
    """Canonical raw in-memory Database Note draft.

    Attributes:
        note_id: Stable Database Note identity.
        title: Raw, untransformed title text.
        body: Raw, untransformed body text.
        keywords_text: Raw comma-delimited keyword input.
        revision: Monotonic draft revision.
    """

    note_id: str
    title: str
    body: str
    keywords_text: str
    revision: int


@dataclass(frozen=True)
class DatabaseNoteSavePayload:
    """Losslessly validated values for one versioned save attempt."""

    title: str
    body: str
    keywords: tuple[str, ...]
    revision: int


@dataclass(frozen=True)
class NoteValidationVeto:
    """Typed actionable reason that the current raw draft cannot be saved."""

    field: Literal["title", "body", "keywords"]
    message: str
    revision: int


@dataclass(frozen=True)
class LibraryNoteSessionSnapshot:
    """Portable immutable view of one active Database Note session."""

    baseline: NormalizedDatabaseNote
    draft: DatabaseNoteDraft
    session_generation: int
    saved_revision: int
    dirty: bool
    saving: bool
    in_conflict: bool
    conflict_generation: int
    status_message: str

    @property
    def note_id(self) -> str:
        """Return the active note identity."""
        return self.draft.note_id

    @property
    def title(self) -> str:
        """Return the canonical raw draft title."""
        return self.draft.title

    @property
    def body(self) -> str:
        """Return the canonical raw draft body."""
        return self.draft.body

    @property
    def keywords_text(self) -> str:
        """Return the canonical raw keyword input."""
        return self.draft.keywords_text

    @property
    def draft_revision(self) -> int:
        """Return the current canonical draft revision."""
        return self.draft.revision

    @property
    def version(self) -> int:
        """Return the last confirmed optimistic-lock version."""
        return self.baseline.version


@dataclass(frozen=True)
class LibraryNotesFocusIdentity:
    """Textual-free semantic focus, selection, and scroll restoration tuple."""

    stage: LibraryNotesStage
    region: LibraryNotesRegion
    note_id: str | None
    semantic_role: str
    body_selection_start: tuple[int, int] | None = None
    body_selection_end: tuple[int, int] | None = None
    scroll_offset: tuple[int, int] | None = None


def _validation_veto(
    draft: DatabaseNoteDraft,
    field: Literal["title", "body", "keywords"],
    message: str,
) -> NoteValidationVeto:
    return NoteValidationVeto(field=field, message=message, revision=draft.revision)


def validate_database_note_draft(
    draft: DatabaseNoteDraft,
) -> DatabaseNoteSavePayload | NoteValidationVeto:
    """Build an exact save payload or veto any transforming persistence path.

    Delimiter-adjacent keyword whitespace and empty comma segments are syntax;
    every remaining token preserves its spelling and order. Title, body, and
    keyword content are never truncated, sanitized into replacement text, or
    silently deduplicated.

    Args:
        draft: Canonical raw Database Note draft.

    Returns:
        An exact persistence payload, or a typed actionable validation veto.
    """
    title = draft.title
    if not isinstance(title, str):
        return _validation_veto(
            draft,
            "title",
            "Title must be text — fix the template or input to save.",
        )
    if len(title) > DATABASE_NOTE_TITLE_MAX_CHARS:
        return _validation_veto(
            draft,
            "title",
            f"Title is {len(title)}/{DATABASE_NOTE_TITLE_MAX_CHARS} characters — shorten it to save.",
        )
    if title != title.strip():
        return _validation_veto(
            draft,
            "title",
            "Title begins or ends with whitespace — remove it to save.",
        )
    if sanitize_string(title, max_length=DATABASE_NOTE_TITLE_MAX_CHARS) != title:
        return _validation_veto(
            draft,
            "title",
            "Title contains unsupported control characters — remove them to save.",
        )
    if not validate_text_input(
        title, max_length=DATABASE_NOTE_TITLE_MAX_CHARS, allow_html=False
    ):
        return _validation_veto(
            draft,
            "title",
            "Title contains unsafe markup — revise it to save.",
        )

    body = draft.body
    if not isinstance(body, str):
        return _validation_veto(
            draft,
            "body",
            "Body must be text — fix the template or input to save.",
        )
    if len(body) > DATABASE_NOTE_BODY_MAX_CHARS:
        return _validation_veto(
            draft,
            "body",
            f"Body is {len(body)}/{DATABASE_NOTE_BODY_MAX_CHARS} characters — shorten it to save.",
        )
    if sanitize_string(body, max_length=DATABASE_NOTE_BODY_MAX_CHARS) != body:
        return _validation_veto(
            draft,
            "body",
            "Body contains unsupported control characters — remove them to save.",
        )

    keywords: list[str] = []
    seen: set[str] = set()
    keywords_text = draft.keywords_text
    if not isinstance(keywords_text, str):
        return _validation_veto(
            draft,
            "keywords",
            "Keywords must be text — fix the template or input to save.",
        )
    for raw_token in keywords_text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if len(token) > DATABASE_NOTE_KEYWORD_MAX_CHARS:
            return _validation_veto(
                draft,
                "keywords",
                f"A keyword is {len(token)}/{DATABASE_NOTE_KEYWORD_MAX_CHARS} characters — shorten it to save.",
            )
        if sanitize_string(token, max_length=DATABASE_NOTE_KEYWORD_MAX_CHARS) != token:
            return _validation_veto(
                draft,
                "keywords",
                "A keyword contains unsupported control characters — revise it to save.",
            )
        if not validate_text_input(
            token, max_length=DATABASE_NOTE_KEYWORD_MAX_CHARS, allow_html=False
        ):
            return _validation_veto(
                draft,
                "keywords",
                "A keyword contains unsafe markup — revise it to save.",
            )
        identity = token.casefold()
        if identity in seen:
            return _validation_veto(
                draft,
                "keywords",
                "Keywords contain a case-insensitive duplicate — remove one to save.",
            )
        seen.add(identity)
        keywords.append(token)

    return DatabaseNoteSavePayload(
        title=title,
        body=body,
        keywords=tuple(keywords),
        revision=draft.revision,
    )


def ellipsize_note_title_cells(title: str, max_cells: int) -> str:
    """Return a plain one-row title no wider than ``max_cells`` terminal cells.

    Args:
        title: Raw title to format for display only.
        max_cells: Maximum terminal-cell width, including an ellipsis.

    Returns:
        The original title when it fits, otherwise a cell-safe ellipsized copy.
        The raw input is never modified.
    """
    if max_cells <= 0:
        return ""

    one_row_title = re.sub(r"[\r\n\t\x85\u2028\u2029]", " ", title)
    width = sum(get_character_cell_size(character) for character in one_row_title)
    if width <= max_cells:
        return one_row_title

    ellipsis = "…"
    remaining = max_cells - get_character_cell_size(ellipsis)
    if remaining <= 0:
        return ellipsis if max_cells >= 1 else ""

    visible: list[str] = []
    used = 0
    for character in one_row_title:
        character_width = get_character_cell_size(character)
        if used + character_width > remaining:
            break
        visible.append(character)
        used += character_width
    return "".join(visible) + ellipsis


@dataclass(frozen=True)
class LibraryNotesListRow:
    """One row in the Library notes canvas's list view.

    Attributes:
        note_id: The note's id.
        title: Display title (``"Untitled"`` when blank).
        age_label: Relative-age label (e.g. ``"3m"``, ``"1d"``) derived
            from the note's most recent modified/created timestamp, or
            ``""`` when no timestamp is available.
        checked: Whether this row is checked in multi-select mode.
    """

    note_id: str
    title: str
    age_label: str
    checked: bool = False


_NOTE_OPERATION_LABELS = {
    "import": "Import",
    "export": "Export",
    "copy": "Copy",
    "console": "Use in Console",
}


@dataclass(frozen=True)
class LibraryNotesOperationState:
    """One token-gated transfer status owned by its initiating region.

    Attributes:
        kind: External action whose status is being reported.
        token: Monotonic operation identity used to reject stale completion.
        phase: Current operation lifecycle phase.
        region: Notes surface that owns and displays this status.
        completion_next_action: Optional recovery step after committed success.
        failure_next_action: Recovery instruction rendered after failure.
    """

    kind: Literal["import", "export", "copy", "console"]
    token: int
    phase: Literal["running", "complete", "failed"]
    region: Literal["navigator", "editor", "context"]
    completion_next_action: str = ""
    failure_next_action: str = "try again"

    @property
    def running(self) -> bool:
        """Return whether the external side effect is still in flight."""
        return self.phase == "running"

    @property
    def status_line(self) -> str:
        """Render the compact active-region status contract."""
        action = _NOTE_OPERATION_LABELS[self.kind]
        if self.phase == "running":
            return f"{action}…"
        if self.phase == "complete":
            if self.completion_next_action:
                next_action = self.completion_next_action.rstrip(". ")
                return f"{action} complete — {next_action}."
            return f"{action} complete."
        next_action = self.failure_next_action.rstrip(". ")
        return f"{action} failed — {next_action}."


@dataclass(frozen=True)
class LibraryNoteCreateOutcome:
    """Typed boundary between persistence and opening the created note.

    Attributes:
        kind: Whether persistence failed, committed without an editor, or
            committed and opened successfully.
        note_id: Persisted identity when ``kind`` is not ``"failed"``.
    """

    kind: Literal["failed", "created_not_opened", "opened"]
    note_id: str = ""


@dataclass(frozen=True)
class LibraryNoteDeleteReceipt:
    """Recovery identity retained after one successful note deletion."""

    note_id: str
    title: str
    expected_version: int


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
        select_mode: Whether multi-select mode is active.
        selected_count: Number of rendered rows currently checked.
        total_count: Total notes in the unfiltered source.
        result_count: Number of rows rendered after filtering.
        empty_kind: Typed distinction between populated, source-empty, and
            filter-empty states.
        sort_choices_visible: Whether direct sort choices are expanded.
        operation_status: Active Navigator transfer status, if any.
        operation_running: Whether Navigator actions must be gated.
        delete_receipt: Most recently deleted note available to Undo.
    """

    rows: tuple[LibraryNotesListRow, ...]
    header_copy: str
    status_copy: str
    empty_copy: str
    select_mode: bool = False
    selected_count: int = 0
    total_count: int = 0
    result_count: int = 0
    empty_kind: Literal["populated", "source-empty", "filter-empty"] = "populated"
    sort_choices_visible: bool = False
    operation_status: str = ""
    operation_running: bool = False
    delete_receipt: LibraryNoteDeleteReceipt | None = None


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


def _row(
    record: Mapping[str, Any], *, now: datetime, selected_ids: frozenset[str]
) -> LibraryNotesListRow:
    raw = _updated_raw(record)
    note_id = _text(record.get("id"))
    return LibraryNotesListRow(
        note_id=note_id,
        title=_text(record.get("title")) or "Untitled",
        age_label=format_console_relative_age(raw, now=now) if raw else "",
        checked=note_id in selected_ids,
    )


def build_library_notes_list_state(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    filter_note: str = "",
    total_count: int | None = None,
    now: datetime | None = None,
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
    sort_choices_visible: bool = False,
    operation_status: str = "",
    operation_running: bool = False,
    delete_receipt: LibraryNoteDeleteReceipt | None = None,
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
        total_count: Total notes in the source before filtering. Defaults to
            the rendered row count when the caller has no separate total.
        now: Reference time for relative-age labels; defaults to the
            current UTC time.
        select_mode: Whether multi-select mode is active.
        selected_ids: The currently checked note ids.
        delete_receipt: Optional recovery identity to render above the rows.

    Returns:
        The list view's display state.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)
    rows = tuple(
        _row(record, now=reference_now, selected_ids=selected_ids)
        for record in (records or ())
        if isinstance(record, Mapping) and _text(record.get("id"))
    )
    status_copy = ""
    if filter_note:
        noun = "result" if len(rows) == 1 else "results"
        status_copy = ellipsize_note_title_cells(
            f"filter: {filter_note} · {len(rows)} {noun}", 52
        )
    if operation_status:
        status_copy = ellipsize_note_title_cells(operation_status, 52)
    selected_count = sum(1 for r in rows if r.checked)
    source_total = len(rows) if total_count is None else max(total_count, 0)
    empty_copy = ""
    empty_kind: Literal["populated", "source-empty", "filter-empty"] = "populated"
    if not rows:
        if source_total == 0:
            empty_copy = _EMPTY_NOTES_COPY
            empty_kind = "source-empty"
        elif filter_note:
            filter_copy = ellipsize_note_title_cells(filter_note, 32)
            empty_copy = f"No notes match “{filter_copy}”. Clear the filter."
            empty_kind = "filter-empty"
    return LibraryNotesListState(
        rows=rows,
        header_copy=f"Notes ({source_total})",
        status_copy=status_copy,
        empty_copy=empty_copy,
        select_mode=select_mode,
        selected_count=selected_count,
        total_count=source_total,
        result_count=len(rows),
        empty_kind=empty_kind,
        sort_choices_visible=sort_choices_visible,
        operation_status=operation_status,
        operation_running=operation_running,
        delete_receipt=delete_receipt,
    )


def patch_note_records_after_save(
    records: Sequence[Mapping[str, Any]] | None,
    note_id: str,
    *,
    title: str,
    modified_at: str,
) -> tuple[Mapping[str, Any], ...]:
    """Return ``records`` with the just-saved note's list fields refreshed.

    A successful in-canvas note save persists to the DB but the notes LIST
    is rendered from the screen's cached source-record snapshot -- without
    this patch the list keeps showing the pre-save title, stale relative
    age, and stale Newest ordering until the next full snapshot refetch.

    Args:
        records: The cached note records (any Mapping shape), or ``None``.
        note_id: The saved note's id.
        title: The saved title to reflect in the list row.
        modified_at: ISO-8601 timestamp of the save, written to the
            record's ``last_modified`` (the first key ``_updated_raw``
            consults for both the age label and Newest/Oldest sorting).

    Returns:
        A new tuple with the matching record replaced by a patched copy;
        non-matching (and non-mapping) entries pass through unchanged.
    """
    target_id = _text(note_id)
    patched: list[Mapping[str, Any]] = []
    for record in records or ():
        if isinstance(record, Mapping) and _text(record.get("id")) == target_id:
            patched.append({**record, "title": title, "last_modified": modified_at})
        else:
            patched.append(record)
    return tuple(patched)


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
            note_id="",
            title="",
            content="",
            keywords_text="",
            version=None,
            meta_line="",
            has_note=False,
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
        parts.append(
            f"Created {format_console_relative_age(created, now=reference_now)}"
        )
    modified = _updated_raw(detail)
    if modified:
        parts.append(
            f"Modified {format_console_relative_age(modified, now=reference_now)}"
        )
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
    """Render a note's export text, mirroring the retired standalone Notes screen's export builder.

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
    timestamp = (now if now is not None else datetime.now()).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
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


def resolve_note_template_placeholders(
    text: str, *, now: datetime | None = None
) -> str:
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
    """Human label for a template row.

    Strips the redundant "template" wording from descriptions ("Template
    for meeting notes" -> "Meeting notes") exactly like the retired
    standalone Notes screen's workbench helper did -- replicated here so
    the pure module stays Textual-free (that workbench module imported
    Textual and was deleted with the standalone screen).
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
    secondary line rather than being dropped. Activating such a row lets the
    screen's lossless validation boundary surface a typed, visible veto.

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
