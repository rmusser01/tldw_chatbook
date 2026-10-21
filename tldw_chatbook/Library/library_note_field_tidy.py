"""Tidy a note field when the user leaves it (TASK-34000.10, review N-07).

ADR-027 forbids the save from stripping a title or deduplicating keywords:
persistence never rewrites what the user typed. So a title ending in a
space, or keywords with a case-insensitive duplicate ("ok, x, X"), veto
every autosave until the user notices -- and the review's first-timer never
did. The repair that keeps ADR-027 is a VISIBLE draft edit made at the
moment the user leaves the field: the Input shows the tidied text, the
session receives it as an ordinary draft mutation through the field's own
``Changed`` handler, and the save payload stays untouched.

The two pure functions are the inverse of exactly the two vetoes they
target (``validate_database_note_draft``'s whitespace and duplicate arms)
and nothing more: no truncation, no markup change, no rewrite of a value
the save would already accept. ``Tests/Library/test_library_note_field_tidy.py``
pins that with Hypothesis.

``tidy_library_note_field_on_blur`` is the one UI-aware seam: it takes the
screen and a Textual ``Input.Blurred`` event by duck type, so this module
imports nothing from Textual and stays a pure Library module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .library_shell_state import LIBRARY_ROW_BROWSE_NOTES, LIBRARY_ROW_CREATE_NOTE

if TYPE_CHECKING:
    from ..UI.Screens.library_screen import LibraryScreen

#: The editor Inputs the tidy applies to, by widget id, with the tidy each
#: one gets and the screen handler its ``Changed`` is routed through. Both
#: keyword boxes edit the one keywords draft (only one is displayed at a
#: time), so both get the keywords tidy.
_TIDY_BY_FIELD_ID: dict[str, tuple[str, str]] = {
    "library-note-title": ("title", "handle_library_note_title_changed"),
    "library-note-keywords": ("keywords", "handle_library_note_keywords_changed"),
    "library-note-context-keywords": (
        "keywords",
        "handle_library_note_context_keywords_changed",
    ),
}

#: The rail rows on which the note editor is the visible surface (the
#: ``resume_library_note_autosave`` precedent). A blur caused by a switch to
#: Media or another reader row must not write into a note that is parked.
_NOTES_EDITOR_ROWS = frozenset({LIBRARY_ROW_BROWSE_NOTES, LIBRARY_ROW_CREATE_NOTE})


def tidy_note_title(value: str) -> str:
    """Return the title without leading or trailing whitespace.

    The inverse of the "Title begins or ends with whitespace" veto. Inner
    whitespace is content and is kept exactly.

    Args:
        value: The title as typed.

    Returns:
        The title with its edge whitespace removed.
    """
    return value.strip()


def tidy_note_keywords(value: str) -> str:
    """Return the keywords with case-insensitive duplicates dropped.

    The inverse of the "Keywords contain a case-insensitive duplicate" veto.
    Tokens are split on commas and stripped (delimiter whitespace and empty
    segments are syntax the save already ignores); the FIRST spelling of
    each case-folded identity is kept, in order; the result is joined with
    ", ". Nothing is truncated or otherwise rewritten, so a keyword the save
    refuses for its length or content still reaches the save as typed.

    Args:
        value: The keywords text as typed.

    Returns:
        The same keyword set, each identity once, first spelling first.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        identity = token.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(token)
    return ", ".join(kept)


def tidy_library_note_field_on_blur(screen: LibraryScreen, event: Any) -> bool:
    """Tidy the note field the user just left, as a visible draft edit.

    Guards, in order: the Input is one of the editor's fields; the editor is
    the visible surface (editor view, on the Notes or Create row); the
    editor is armed and no presentation sync is writing the fields; the
    Input is still attached (a blur caused by a recompose teardown must not
    write); the session's note is the selected note. Then, when the tidy
    changes the value, the Input shows the tidied text and the field's own
    ``Changed`` handler is called with it synchronously -- the same route a
    keystroke takes (mutate, re-arm the autosave, repaint) -- so no sync can
    overwrite the Input from a snapshot that has not received the edit yet.
    The ``Changed`` the value assignment would post is suppressed, as it
    would be a second copy of the same edit.

    Args:
        screen: The Library screen that owns the editor.
        event: The ``Input.Blurred`` event (``event.input`` is the field).

    Returns:
        True when the field was tidied; False when nothing needed to change
        or a guard declined.
    """
    field = event.input
    entry = _TIDY_BY_FIELD_ID.get(field.id or "")
    if entry is None:
        return False
    state = screen._notes_state
    if (
        state.view != "editor"
        or screen._library_selected_row_id not in _NOTES_EDITOR_ROWS
        or not state.editor_armed
        or state.presentation_syncing
        or not field.is_attached
    ):
        return False
    snapshot = screen._library_note_session.snapshot
    if snapshot is None or snapshot.note_id != state.selected_note_id:
        return False
    kind, handler_name = entry
    tidy = tidy_note_title if kind == "title" else tidy_note_keywords
    tidied = tidy(field.value)
    if tidied == field.value:
        return False
    with field.prevent(field.Changed):
        field.value = tidied
    getattr(screen, handler_name)(field.Changed(field, tidied))
    return True
