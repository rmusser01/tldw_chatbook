"""The Library note autosave debounce has a maximum wait (TASK-34000.1, N-01).

The 2 s debounce re-arms on every keystroke, so one key every 1.2 s was never
saved at all. ``library_note_autosave_delay`` caps it with a maximum wait
measured from the first unsaved keystroke of a burst. These pin the
arithmetic and what starts a new burst; ``Tests/UI/test_library_quit_guard.py``
drives the same seam through the real editor and a real database.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Library_Modules.library_notes_state import LibraryNotesState
from tldw_chatbook.UI.Library_Modules.library_pending_work import (
    arm_library_note_autosave,
    library_note_autosave_delay,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS,
    LIBRARY_NOTES_AUTOSAVE_SECONDS,
)

pytestmark = pytest.mark.unit

DEBOUNCE = LIBRARY_NOTES_AUTOSAVE_SECONDS
MAX_WAIT = LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS


def _snapshot(*, note_id: str = "n1", session: int = 1, saved: int = 0):
    return SimpleNamespace(
        note_id=note_id, session_generation=session, saved_revision=saved
    )


def _delay(state, snapshot, now: float) -> float:
    return library_note_autosave_delay(
        state, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT, now=now
    )


def test_the_production_pair_is_two_and_ten_seconds() -> None:
    assert (DEBOUNCE, MAX_WAIT) == (2.0, 10.0)


def test_a_steady_typist_is_saved_once_the_max_wait_runs_out() -> None:
    """One key every 1.2 s: the debounce alone would never fire."""
    state = LibraryNotesState()
    snapshot = _snapshot()
    delays = [_delay(state, snapshot, 100.0 + 1.2 * key) for key in range(10)]

    # Every key re-arms the full debounce until the burst nears its cap...
    assert delays[:7] == [DEBOUNCE] * 7
    # ...then the remaining time to the cap, which a 1.2 s cadence reaches.
    assert delays[7] == pytest.approx(1.6)
    assert delays[8] == pytest.approx(0.4)
    # A key landing after the cap (the timer was re-armed) saves at once.
    assert delays[9] == 0.0


def test_a_landed_save_starts_a_new_burst() -> None:
    state = LibraryNotesState()
    _delay(state, _snapshot(saved=0), now=100.0)
    assert _delay(state, _snapshot(saved=0), now=109.5) == pytest.approx(0.5)

    assert _delay(state, _snapshot(saved=1), now=109.6) == DEBOUNCE


@pytest.mark.parametrize(
    "other", [_snapshot(note_id="n2"), _snapshot(session=2)], ids=["note", "session"]
)
def test_another_note_or_session_starts_a_new_burst(other) -> None:
    state = LibraryNotesState()
    _delay(state, _snapshot(), now=100.0)
    assert _delay(state, other, now=109.5) == DEBOUNCE


class _Screen:
    """The three seams ``arm_library_note_autosave`` uses, recorded."""

    def __init__(self) -> None:
        self._notes_state = LibraryNotesState()
        self.timers: list[tuple[float, object]] = []
        self.fired: list[int] = []

    def set_timer(self, delay, callback):
        self.timers.append((delay, callback))
        return SimpleNamespace(stop=lambda: None)

    def _fire_library_note_autosave(self, generation: int) -> None:
        self.fired.append(generation)


def test_the_timer_ends_its_burst_so_a_vetoed_save_is_not_retried_per_key() -> None:
    """A failed or vetoed autosave keeps the saved revision, so the burst key
    alone would leave every later keystroke at a zero delay."""
    screen = _Screen()
    snapshot = _snapshot()
    arm_library_note_autosave(screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT)
    (delay, fire) = screen.timers[-1]
    assert delay == DEBOUNCE
    assert screen._notes_state.autosave_burst is not None

    fire()

    assert screen.fired == [screen._notes_state.autosave_generation]
    assert screen._notes_state.autosave_burst is None
    arm_library_note_autosave(screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT)
    assert screen.timers[-1][0] == DEBOUNCE


def test_a_superseded_timer_leaves_the_current_burst_alone() -> None:
    screen = _Screen()
    arm_library_note_autosave(
        screen, _snapshot(), debounce=DEBOUNCE, max_wait=MAX_WAIT
    )
    (_delay_seconds, stale_fire) = screen.timers[-1]
    screen._notes_state.autosave_generation += 1  # invalidated by a newer key
    burst = screen._notes_state.autosave_burst

    stale_fire()

    assert screen._notes_state.autosave_burst == burst
