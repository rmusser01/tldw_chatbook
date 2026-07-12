"""Pure display-state contracts for the Library prompts canvas.

Consumes record mappings shaped like ``PromptsDatabase.fetch_prompt_details``
/ ``list_prompts`` rows (keys: ``id``, ``name``, ``author``, ``details``,
``system_prompt``, ``user_prompt``, ``keywords``, ``last_modified`` /
``created_at``, ``version``). No Textual imports; the only DB import is the
``ConflictError`` exception type used to classify save outcomes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.DB.Prompts_DB import ConflictError
from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

_TIMESTAMP_KEYS = ("last_modified", "created_at")


@dataclass(frozen=True)
class PromptListRow:
    """One row in the Library prompts canvas's list view.

    Attributes:
        prompt_id: The prompt's id.
        name: Display name, raw (the canvas escapes markup at render time).
        secondary: ``"<details> · <age>"`` -- the prompt's purpose, not
            ``author``/``keywords`` (Task 8b D2/U1; see ``_matches_query``'s
            comment for why keywords are never shown here) -- with either
            part (no details, or no timestamp) omitted, along with its
            separator.
    """

    prompt_id: int
    name: str
    secondary: str


@dataclass(frozen=True)
class PromptsListState:
    """Display state for the Library prompts canvas's list view.

    Attributes:
        rows: The prompts to render, already filtered/sorted.
        count: ``len(rows)``.
        sort: The sort mode used to build ``rows`` (``"newest"`` or
            ``"name"``), echoed back for the caller's toggle label.
    """

    rows: tuple[PromptListRow, ...]
    count: int
    sort: str


@dataclass(frozen=True)
class PromptEditorState:
    """Display state for the Library prompts canvas's in-canvas editor.

    Attributes:
        prompt_id: The open prompt's id, or ``None`` when unknown/not yet
            saved.
        name: The prompt's name.
        author: The prompt's author.
        details: The prompt's description/details text.
        system_prompt: The prompt's system-prompt text.
        user_prompt: The prompt's user-prompt text.
        keywords_csv: The prompt's keywords as a single comma-separated
            string.
        version: The prompt's optimistic-lock version, or ``None`` when
            unknown.
        created: Raw ``created_at`` timestamp text, or ``""`` when absent.
        modified: Raw ``last_modified``/``created_at`` timestamp text, or
            ``""`` when absent.
    """

    prompt_id: int | None
    name: str
    author: str
    details: str
    system_prompt: str
    user_prompt: str
    keywords_csv: str
    version: int | None
    created: str
    modified: str


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw_text(value: Any) -> str:
    """Like ``_text`` but preserves body text verbatim (no stripping)."""
    return "" if value is None else str(value)


def _timestamp_raw(record: Mapping[str, Any]) -> str:
    for key in _TIMESTAMP_KEYS:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _csv_from_keywords(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        items = []
        for item in value:
            if isinstance(item, Mapping):
                item = item.get("keyword") or item.get("text") or item.get("label")
            text = _text(item)
            if text:
                items.append(text)
        return ", ".join(items)
    return ""


def _matches_query(record: Mapping[str, Any], query_lower: str) -> bool:
    if not query_lower:
        return True
    if query_lower in _text(record.get("name")).lower():
        return True
    # Task 8b D2: matches `details`, NOT `keywords` -- the raw local
    # `list_prompts` DB query has no per-page-keyword-join seam (only a
    # per-single-id `fetch_keywords_for_prompt`, an N+1 shape for a whole
    # page), so real list-page records never carry `keywords` at all (see
    # `_prompts_page_records_or_empty`'s docstring). Matching on it here
    # would silently promise a capability the filter could never actually
    # deliver. `details` IS present on list rows (the DB query now selects
    # it too), so this is the honest, still-cheap (no extra query) fix.
    # Keyword-in-list filtering awaits a batched per-page keyword-join DB
    # seam (backlog) if that capability is ever wanted.
    return query_lower in _text(record.get("details")).lower()


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _row(record: Mapping[str, Any], *, now: datetime) -> PromptListRow | None:
    prompt_id = _to_int(record.get("id"))
    if prompt_id is None:
        return None
    # Task 8b D2/U1: surfaces the prompt's PURPOSE (details) instead of
    # `author · age` -- author (and keywords, never present on list rows
    # anyway -- see `_matches_query`'s comment) are dropped from the
    # secondary line entirely.
    details = _text(record.get("details"))
    raw_timestamp = _timestamp_raw(record)
    age = format_console_relative_age(raw_timestamp, now=now) if raw_timestamp else ""
    secondary = " · ".join(part for part in (details, age) if part)
    return PromptListRow(prompt_id=prompt_id, name=_text(record.get("name")), secondary=secondary)


def build_prompts_list_state(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    query: str,
    sort: str,
    now: datetime,
) -> PromptsListState:
    """Build the Library prompts canvas's list-view display state.

    Records missing a mapping shape or a convertible ``id`` are silently
    dropped rather than raising, matching the Library notes state module's
    degrade-don't-crash behavior for malformed source records.

    Args:
        records: The prompts to render.
        query: Filter text, matched case-insensitively against name and
            details (Task 8b D2 -- not keywords, a field list-page records
            never actually carry; see ``_matches_query``); ``""`` disables
            filtering.
        sort: ``"name"`` sorts alphabetically case-insensitively; any other
            value (including ``"newest"``) sorts by most-recent
            modified/created timestamp, newest first.
        now: Reference time for the secondary line's relative-age part.

    Returns:
        The list view's display state.
    """
    query_lower = _text(query).lower()
    items = [
        record
        for record in (records or ())
        if isinstance(record, Mapping) and _matches_query(record, query_lower)
    ]
    if sort == "name":
        items.sort(key=lambda record: _text(record.get("name")).lower())
    else:
        items.sort(key=_timestamp_raw, reverse=True)
    rows = tuple(row for row in (_row(record, now=now) for record in items) if row is not None)
    return PromptsListState(rows=rows, count=len(rows), sort=sort)


def build_prompt_editor_state(detail: Mapping[str, Any]) -> PromptEditorState:
    """Build the prompt editor's display state from a prompt detail mapping.

    Args:
        detail: A prompt detail mapping (the raw ``fetch_prompt_details``
            row, ``keywords`` as a list of strings), or a malformed/empty
            mapping. Tolerated to have missing/None fields.

    Returns:
        Immutable editor state, with keywords joined into a single
        comma-separated string.
    """
    if not isinstance(detail, Mapping):
        detail = {}
    return PromptEditorState(
        prompt_id=_to_int(detail.get("id")),
        name=_text(detail.get("name")),
        author=_text(detail.get("author")),
        details=_raw_text(detail.get("details")),
        system_prompt=_raw_text(detail.get("system_prompt")),
        user_prompt=_raw_text(detail.get("user_prompt")),
        keywords_csv=_csv_from_keywords(detail.get("keywords")),
        version=_to_int(detail.get("version")),
        created=_text(detail.get("created_at")),
        modified=_timestamp_raw(detail),
    )


def prompt_editor_meta_line(editor_state: PromptEditorState, *, now: datetime | None = None) -> str:
    """Render the prompt editor's muted meta line.

    Unlike the notes editor's ``meta_line`` (precomputed as part of
    ``LibraryNoteEditorState`` by ``build_library_note_editor_state``),
    ``PromptEditorState`` carries only raw ``modified``/``version`` fields
    -- shared here (rather than duplicated) so both the editor canvas's
    initial render and the screen's post-save targeted Static update agree
    on the exact same text. The Prompts table has no ``created_at`` column
    at all, so this renders only ``Modified <age>`` (never a "Created"
    part, and never a fake one) plus ``vN``.

    Args:
        editor_state: The prompt editor's current display state.
        now: Reference time for the relative-age part; defaults to the
            current UTC time.

    Returns:
        ``"New prompt"`` when ``editor_state.prompt_id`` is ``None`` (the
        Task 8b D1 create-flow sentinel: a blank, not-yet-saved record --
        see ``library_screen.py``'s ``_enter_library_prompt_create_editor``
        and the Duplicate action). Otherwise ``"Modified <age> · vN"``,
        with either part omitted (and its separator) when unknown.
    """
    if editor_state.prompt_id is None:
        return "New prompt"
    reference_now = now if now is not None else datetime.now(timezone.utc)
    parts: list[str] = []
    if editor_state.modified:
        age = format_console_relative_age(editor_state.modified, now=reference_now)
        parts.append(f"Modified {age}")
    if editor_state.version is not None:
        parts.append(f"v{editor_state.version}")
    return " · ".join(parts)


def _is_name_conflict(exc: Exception | None, message_lower: str) -> bool:
    if isinstance(exc, sqlite3.IntegrityError) and "unique" in str(exc).lower():
        return True
    return "unique" in message_lower or "already exists" in message_lower


def classify_prompt_save_error(result_id: Any, message: str, exc: Exception | None) -> str:
    """Classify the outcome of a prompt save (add/update) call.

    Args:
        result_id: The id the save call returned, or ``None`` when it did
            not produce a fresh saved row.
        message: Any accompanying human-readable message from the save
            call (e.g. the ``add_prompt`` tuple's message slot).
        exc: The exception raised by the save call, if any.

    Returns:
        One of ``"soft-deleted-name"``, ``"conflict"``, ``"name-in-use"``,
        ``"ok"``, or ``"error"``.
    """
    message_lower = _text(message).lower()
    if result_id is None and "soft-deleted" in message_lower:
        return "soft-deleted-name"
    if isinstance(exc, ConflictError):
        return "conflict"
    if _is_name_conflict(exc, message_lower):
        return "name-in-use"
    if exc is None and result_id is not None:
        return "ok"
    return "error"
