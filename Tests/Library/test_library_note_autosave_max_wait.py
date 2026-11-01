"""The Library note autosave debounce has a maximum wait (TASK-34000.1, N-01).

The 2 s debounce re-arms on every keystroke, so one key every 1.2 s was never
saved at all. ``library_note_autosave_delay`` caps it with a maximum wait
measured from the first unsaved keystroke of a burst. These pin the
arithmetic and what starts a new burst; ``Tests/UI/test_library_quit_guard.py``
drives the same seam through the real editor and a real database.

Final review C1: a burst whose max wait had run out armed a 0 s timer, and
Textual 8.2.8 kills a 0 s ``Timer`` with ``ZeroDivisionError`` before it calls
back. The old pin here asserted that ``0.0`` against a fake ``set_timer``, so
it enshrined the defect. The delay is now pinned as positive, and one test
arms an expired burst on a real Textual message pump.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from textual.app import App

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
)
from tldw_chatbook.UI.Library_Modules import library_note_autosave
from tldw_chatbook.UI.Library_Modules.library_notes_state import LibraryNotesState
from tldw_chatbook.UI.Library_Modules.library_note_autosave import (
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
    # A key landing after the cap saves almost at once -- but never on a
    # non-positive delay, which Textual's Timer cannot fire (final review C1).
    assert delays[9] > 0.0
    assert delays[9] <= library_note_autosave.AUTOSAVE_MIN_DELAY_SECONDS


@pytest.mark.parametrize("late", [0.0, 0.001, 1.0, 3600.0])
def test_an_expired_burst_never_arms_a_non_positive_delay(late: float) -> None:
    state = LibraryNotesState()
    snapshot = _snapshot()
    _delay(state, snapshot, now=100.0)

    delay = _delay(state, snapshot, now=100.0 + MAX_WAIT + late)

    assert delay > 0.0
    assert delay <= library_note_autosave.AUTOSAVE_MIN_DELAY_SECONDS


@pytest.mark.asyncio
async def test_an_expired_burst_fires_on_a_real_textual_message_pump() -> None:
    """The seam with nothing faked: ``App.set_timer`` and its message pump.

    A burst that began before its max wait ran out is armed again (a key
    after Keep editing on a refused quit, or after a rail switch away and
    back). The timer must call back, and stopping the app must not raise.
    """

    fired: list[int] = []
    called_back = False
    shutdown_error: BaseException | None = None
    app = App()
    try:
        async with app.run_test() as pilot:
            state = LibraryNotesState()
            snapshot = _snapshot()
            key = (
                snapshot.note_id,
                snapshot.session_generation,
                snapshot.saved_revision,
            )
            state.autosave_burst = (key, time.monotonic() - (MAX_WAIT + 1.0))
            screen = SimpleNamespace(
                _notes_state=state,
                set_timer=app.set_timer,  # the real Textual timer
                _notes_controller=SimpleNamespace(
                    _fire_library_note_autosave=fired.append
                ),
            )
            arm_library_note_autosave(
                screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT
            )
            try:
                async with asyncio.timeout(DEBOUNCE):
                    while not fired:
                        await pilot.pause(0.02)
            except TimeoutError:
                pass
            called_back = fired == [state.autosave_generation]
    except ZeroDivisionError as error:  # Textual awaits the dead timer task
        shutdown_error = error

    assert called_back, "the expired burst's timer never called back"
    assert shutdown_error is None, "a dead 0 s timer made app shutdown raise"


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
        self._notes_controller = SimpleNamespace(
            _fire_library_note_autosave=self.fired.append
        )

    def set_timer(self, delay, callback):
        self.timers.append((delay, callback))
        return SimpleNamespace(stop=lambda: None)


def test_a_starting_save_ends_its_burst_so_a_vetoed_save_is_not_retried_per_key() -> (
    None
):
    """A failed or vetoed autosave keeps the saved revision, so the burst key
    alone would leave every later keystroke at the minimum delay."""
    screen = _Screen()
    state = screen._notes_state
    snapshot = _snapshot()
    arm_library_note_autosave(screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT)
    (delay, fire) = screen.timers[-1]
    assert delay == DEBOUNCE
    assert state.autosave_burst is not None

    fire()
    assert screen.fired == [state.autosave_generation]
    # The save worker reaches the session: this is where the burst ends.
    assert library_note_autosave.note_save_starts(state, state.autosave_generation)

    assert state.autosave_burst is None
    arm_library_note_autosave(screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT)
    assert screen.timers[-1][0] == DEBOUNCE


def test_a_key_between_the_timer_and_its_save_keeps_the_max_wait() -> None:
    """Final review C1: the timer ended the burst before the save worker's
    generation check. A key in that gap cancelled the max-wait save and began
    a new burst with a full max wait, so a steady typist could wait twice."""
    screen = _Screen()
    state = screen._notes_state
    snapshot = _snapshot()
    burst_started = time.monotonic() - MAX_WAIT  # the burst's max wait is up
    key = (snapshot.note_id, snapshot.session_generation, snapshot.saved_revision)
    state.autosave_burst = (key, burst_started)
    arm_library_note_autosave(screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT)
    (_delay_seconds, fire) = screen.timers[-1]
    stale_generation = state.autosave_generation

    fire()  # the max-wait timer fires; its save worker has not started yet
    state.autosave_generation += 1  # a key lands first and invalidates it
    arm_library_note_autosave(screen, snapshot, debounce=DEBOUNCE, max_wait=MAX_WAIT)

    assert state.autosave_burst == (key, burst_started), "the burst was restarted"
    assert (
        0.0 < screen.timers[-1][0] <= library_note_autosave.AUTOSAVE_MIN_DELAY_SECONDS
    )
    # The superseded worker then starts, and must leave the burst alone.
    assert not library_note_autosave.note_save_starts(state, stale_generation)
    assert state.autosave_burst == (key, burst_started)


def test_an_explicit_save_ends_the_burst_too() -> None:
    state = LibraryNotesState()
    _delay(state, _snapshot(), now=100.0)

    assert library_note_autosave.note_save_starts(state, None)

    assert state.autosave_burst is None


# -- TASK-34000.9 (review N-06): re-arming at screen resume ------------------


class _ResumableScreen(_Screen):
    """The seams ``resume_library_note_autosave`` reads, with the arm recorded.

    ``on_screen_suspend`` stops and clears the timer for any pushed modal
    (TASK-31521); the resume must put back exactly what a dirty editor note
    needs and nothing for every other state.
    """

    def __init__(
        self,
        *,
        view: str = "editor",
        dirty: bool = True,
        saving: bool = False,
        in_conflict: bool = False,
        note_id: str = "n1",
        selected: str = "n1",
        autosave_state: str = "idle",
        armed: bool = False,
        snapshot: bool = True,
        row: str = LIBRARY_ROW_BROWSE_NOTES,
        covered: bool = False,
    ) -> None:
        super().__init__()
        # Fix round 1: the visible rail row (a parked note on Media must not
        # be saved behind the user's back) and whether another screen still
        # covers this one (a chained dialog).
        self._library_selected_row_id = row
        self.app = SimpleNamespace(screen=object() if covered else self)
        self._notes_state.view = view
        self._notes_state.selected_note_id = selected
        self._notes_state.autosave_state = autosave_state
        if armed:
            self._notes_state.autosave_timer = SimpleNamespace(stop=lambda: None)
        self._library_note_session = SimpleNamespace(
            snapshot=(
                SimpleNamespace(
                    note_id=note_id,
                    dirty=dirty,
                    saving=saving,
                    in_conflict=in_conflict,
                    body="draft",
                )
                if snapshot
                else None
            )
        )
        self.scheduled = 0
        self.status_refreshes = 0

    def _schedule_library_note_autosave(self) -> None:
        self.scheduled += 1
        self._notes_state.autosave_timer = SimpleNamespace(stop=lambda: None)

    def _update_library_note_meta_static(self, *, content: str) -> None:
        self.status_refreshes += 1


@pytest.mark.parametrize("row", [LIBRARY_ROW_BROWSE_NOTES, LIBRARY_ROW_CREATE_NOTE])
def test_resume_re_arms_a_dirty_editor_note_once_and_repaints_its_status(row) -> None:
    """On the Notes row, or on Create ▸ New note while the just-created note's
    editor is showing (TASK-34000.25)."""
    screen = _ResumableScreen(row=row)

    assert library_note_autosave.resume_library_note_autosave(screen) is True

    assert screen.scheduled == 1
    assert screen._notes_state.autosave_timer is not None
    assert screen.status_refreshes == 1, "the promise must be repainted as true"


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"dirty": False}, id="clean"),
        pytest.param({"saving": True}, id="saving"),
        pytest.param({"in_conflict": True}, id="conflict"),
        pytest.param({"autosave_state": "validation"}, id="vetoed"),
        pytest.param({"view": "list"}, id="list-view"),
        pytest.param({"armed": True}, id="already-armed"),
        pytest.param({"selected": "n2"}, id="another-note-selected"),
        pytest.param({"snapshot": False}, id="no-session"),
        pytest.param({"autosave_state": "error"}, id="failed-write"),
        pytest.param({"row": "browse-media"}, id="parked-on-media"),
        pytest.param({"covered": True}, id="still-covered-by-a-dialog"),
    ],
)
def test_resume_arms_nothing_unless_a_dirty_editor_note_is_waiting(state) -> None:
    """A vetoed draft cannot succeed without an edit (the next keystroke
    re-arms it); a clean, saving, conflicted or already-armed note needs no
    second timer; a list view has no editor to save. Fix round 1: a failed
    write keeps its "Save failed" status until Save or the next keystroke (a
    retry at resume could write a draft "Discard and quit" just discarded);
    a note parked on another rail row is saved only when the user lands back
    on Notes (the rail-switch precedent); a screen still covered by a chained
    dialog is not visible yet."""
    screen = _ResumableScreen(**state)
    armed_before = screen._notes_state.autosave_timer

    assert library_note_autosave.resume_library_note_autosave(screen) is False

    assert screen.scheduled == 0
    assert screen._notes_state.autosave_timer is armed_before
    assert screen.status_refreshes == 0


def test_an_autosave_counts_as_scheduled_only_while_armed_or_saving() -> None:
    """The status line's promise reads this, not ``dirty``."""
    state = LibraryNotesState()
    snapshot = SimpleNamespace(saving=False)
    assert library_note_autosave.autosave_is_scheduled(state, snapshot) is False

    state.autosave_timer = SimpleNamespace(stop=lambda: None)
    assert library_note_autosave.autosave_is_scheduled(state, snapshot) is True

    state.autosave_timer = None
    state.autosave_state = "saving"
    assert library_note_autosave.autosave_is_scheduled(state, snapshot) is True

    state.autosave_state = "idle"
    assert (
        library_note_autosave.autosave_is_scheduled(state, SimpleNamespace(saving=True))
        is True
    )
