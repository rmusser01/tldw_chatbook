"""Library note autosave policy: the debounce's maximum wait, and its vetoes.

TASK-34000.1 (review N-01). The 2 s debounce re-armed on every keystroke, so
one key every 1.2 s was never saved at all. ``arm_library_note_autosave``
caps each debounce with a maximum wait measured from the first unsaved
keystroke of a burst.

That cap means an autosave can now fire while the user is still typing. An
autosave the save refuses (a title with a trailing space, unsafe markup, a
duplicate keyword) used to route to the offending field and focus it. That
was harmless only while autosaves waited for a pause. Mid-burst, the next
body keys landed in the title, and the next autosave saved them as the title
(review N-07). ``keep_autosave_veto_in_place`` is that guard: a vetoed or
failed *autosave* never moves focus or switches the pane while an editor
field has focus. It reports the reason in the save status instead. An
explicit Save still routes to the field.

Kept out of ``library_screen.py`` (over its size ratchet) and
``library_notes_controller.py`` (at its ratchet). The screen imports this
lazily, so it stays out of the Library preimport census.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ...Library.library_notes_session import NoteSaveOutcome, NoteSaveOutcomeKind

if TYPE_CHECKING:
    from ...Library.library_notes_state import LibraryNoteSessionSnapshot
    from ..Screens.library_screen import LibraryScreen
    from .library_notes_state import LibraryNotesState

#: The note editor's fields. While one of them has focus the user is typing,
#: and an autosave must not move them.
_EDITOR_FIELD_IDS = frozenset(
    {
        "library-note-title",
        "library-note-body",
        "library-note-keywords",
        "library-note-context-keywords",
    }
)
#: Autosave outcomes that used to route to a field and focus it.
_REFUSED = frozenset({NoteSaveOutcomeKind.VALIDATION_VETO, NoteSaveOutcomeKind.FAILED})


def library_note_autosave_delay(
    state: LibraryNotesState,
    snapshot: LibraryNoteSessionSnapshot,
    *,
    debounce: float,
    max_wait: float,
    now: float | None = None,
) -> float:
    """Return the debounce for this keystroke, capped by the burst's max wait.

    A burst starts at the first unsaved keystroke and is keyed on the note,
    its session and its saved revision, so a landed save, another note or a
    reopened session starts a new one. The timer callback ends the burst
    too (``arm_library_note_autosave``), so a vetoed or failed autosave does
    not turn every later keystroke into an immediate retry.

    Args:
        state: The screen's Notes state, which holds the current burst.
        snapshot: The note session snapshot after the keystroke.
        debounce: The normal quiet period before an autosave.
        max_wait: The longest a burst may stay unsaved.
        now: The monotonic clock reading; defaults to ``time.monotonic()``.

    Returns:
        Seconds until the autosave should fire: the debounce, or less when
        the burst's max wait runs out first (never negative).
    """
    now = time.monotonic() if now is None else now
    key = (snapshot.note_id, snapshot.session_generation, snapshot.saved_revision)
    burst = state.autosave_burst
    if burst is None or burst[0] != key:
        burst = (key, now)
        state.autosave_burst = burst
    return max(0.0, min(debounce, burst[1] + max_wait - now))


def arm_library_note_autosave(
    screen: LibraryScreen,
    snapshot: LibraryNoteSessionSnapshot,
    *,
    debounce: float,
    max_wait: float,
) -> None:
    """Start the autosave timer for the current burst.

    The caller has already invalidated the previous timer. The two durations
    are passed in so the screen keeps reading its own module constants,
    which tests patch to make autosave fast or to turn it off.

    Args:
        screen: The Library screen that owns the timer.
        snapshot: The note session snapshot after the keystroke.
        debounce: ``LIBRARY_NOTES_AUTOSAVE_SECONDS``.
        max_wait: ``LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS``.
    """
    state = screen._notes_state
    generation = state.autosave_generation
    delay = library_note_autosave_delay(
        state, snapshot, debounce=debounce, max_wait=max_wait
    )

    def _fire() -> None:
        if generation == state.autosave_generation:
            state.autosave_burst = None
        screen._notes_controller._fire_library_note_autosave(generation)

    state.autosave_timer = screen.set_timer(delay, _fire)


def keep_autosave_veto_in_place(screen: LibraryScreen, outcome: NoteSaveOutcome) -> bool:
    """Present a refused autosave without moving a typing user (review N-07).

    Args:
        screen: The Library screen whose autosave finished.
        outcome: The autosave's typed outcome.

    Returns:
        True when the outcome was a vetoed or failed autosave of the open note
        while an editor field has focus, and its status is now shown in place.
        The caller must then skip the presentation that routes and focuses.
        False for anything else, which the caller presents as before.
    """
    if outcome.kind not in _REFUSED:
        return False
    focused = screen.focused
    if focused is None or focused.id not in _EDITOR_FIELD_IDS:
        return False
    snapshot = screen._library_note_session.snapshot
    if snapshot is None or snapshot.note_id != screen._notes_state.selected_note_id:
        return False
    screen._notes_state.autosave_state = (
        "validation" if outcome.kind is NoteSaveOutcomeKind.VALIDATION_VETO else "error"
    )
    screen._update_library_note_meta_static(content=snapshot.body)
    return True
