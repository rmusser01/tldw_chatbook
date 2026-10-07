"""Library note autosave policy: the debounce's maximum wait, and its vetoes.

TASK-34000.1 (review N-01). The 2 s debounce re-armed on every keystroke, so
one key every 1.2 s was never saved at all. ``arm_library_note_autosave``
caps each debounce with a maximum wait measured from the first unsaved
keystroke of a burst. The delay it arms is never zero or negative
(``AUTOSAVE_MIN_DELAY_SECONDS``), and a burst ends when a save of the note
starts (``note_save_starts``).

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
#: The shortest delay an autosave timer is ever armed with (final review C1).
#: Textual 8.2.8's ``Timer._run`` divides by the interval when it skips a late
#: tick, so a 0 s Timer dies with ``ZeroDivisionError`` before it calls back,
#: and the app then raises that error when it shuts down. A burst whose max
#: wait has already run out therefore saves "almost at once", never "at once".
AUTOSAVE_MIN_DELAY_SECONDS = 0.05


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
    reopened session starts a new one. A save that starts ends the burst too
    (``note_save_starts``), so a vetoed or failed autosave does not
    turn every later keystroke into an immediate retry.

    A burst can outlive its max wait without its timer firing: a quit flush
    or a rail switch cancels the timer and may leave the draft unsaved. The
    next keystroke then gets ``AUTOSAVE_MIN_DELAY_SECONDS``, not zero.

    Args:
        state: The screen's Notes state, which holds the current burst.
        snapshot: The note session snapshot after the keystroke.
        debounce: The normal quiet period before an autosave.
        max_wait: The longest a burst may stay unsaved.
        now: The monotonic clock reading; defaults to ``time.monotonic()``.

    Returns:
        Seconds until the autosave should fire: the debounce, or less when
        the burst's max wait runs out first. Always positive: at least
        ``AUTOSAVE_MIN_DELAY_SECONDS``.
    """
    now = time.monotonic() if now is None else now
    key = (snapshot.note_id, snapshot.session_generation, snapshot.saved_revision)
    burst = state.autosave_burst
    if burst is None or burst[0] != key:
        burst = (key, now)
        state.autosave_burst = burst
    return max(AUTOSAVE_MIN_DELAY_SECONDS, min(debounce, burst[1] + max_wait - now))


def note_save_starts(state: LibraryNotesState, autosave_generation: int | None) -> bool:
    """A save of the open note is about to reach the session: end its burst.

    The burst ends here and not in the timer callback (final review C1). The
    callback only queues the save worker, and a keystroke handled before that
    worker starts invalidates it. Ending the burst in the callback made that
    keystroke begin a new burst with a full max wait, so the max-wait save
    was dropped and a steady typist could wait a second max wait for it.

    Args:
        state: The screen's Notes state, which holds the current burst.
        autosave_generation: The timer generation an autosave captured, or
            None for an explicit Save.

    Returns:
        False when a newer keystroke superseded this autosave: it must not
        save, and the burst (with what is left of its max wait) stays. True
        otherwise, with the burst ended.
    """
    if (
        autosave_generation is not None
        and autosave_generation != state.autosave_generation
    ):
        return False
    state.autosave_burst = None
    return True


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
        # Queues the save worker only. The burst ends when that save starts
        # (``note_save_starts``), so a key landing first keeps it.
        screen._notes_controller._fire_library_note_autosave(generation)

    state.autosave_timer = screen.set_timer(delay, _fire)


def keep_autosave_veto_in_place(
    screen: LibraryScreen, outcome: NoteSaveOutcome
) -> bool:
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
