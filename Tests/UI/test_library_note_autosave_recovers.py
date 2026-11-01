"""Autosave keeps working after its burst's max wait has run out (final review C1).

TASK-34000.1 capped the autosave debounce with a maximum wait. Once a burst's
max wait had passed without its timer firing, the next keystroke armed a 0 s
timer. Textual 8.2.8 kills a 0 s ``Timer`` with ``ZeroDivisionError`` before it
calls back, and the callback was the only thing that ended a burst, so every
later keystroke armed another dead timer. The header went on saying "changes
save automatically" over text that no autosave would ever write, and stopping
the app raised.

Two ways in were deterministic in the real app, and each is one test here:

- Keep editing after a refused quit: the quit flush cancels the pending timer,
  the prompt outlasts the max wait, and the user goes on typing.
- A rail switch away and back: the dirty note session is retained unsaved, its
  timer cancelled, and the return re-arms it after the max wait.

Same harness as ``test_library_quit_guard.py``: the real ``TldwCli``, the real
keys, a real ChaChaNotes database. Every assertion reads the DATABASE ROW, and
each test also fails if stopping the app raises.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, Input, TextArea

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _TITLE,
    SIZE,
    _ctrl_q,
    _keep_editing,
    _library,
    _library_app,
    _open_note,
    _settings_without_splash,
    _type,
    _type_at_end,
    _until,
)

#: Same reason as the quit-guard files: full-app boots need the bootstrap
#: profile locally (``raw_source_selection_changed`` under the per-test redirect).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: Scaled autosave pair (production: 2 s / 10 s), short so both boots fit the
#: UI lane's 60 s per-file rule.
_DEBOUNCE, _MAX_WAIT = 0.5, 1.0
#: How long a test waits for an autosave that should take about one debounce.
#: Generous for a loaded machine; before the fix no wait was long enough.
_AUTOSAVE_WITHIN = 10.0


def _scaled_autosave(monkeypatch) -> None:
    monkeypatch.setattr(
        library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", _DEBOUNCE
    )
    monkeypatch.setattr(
        library_screen_module, "LIBRARY_NOTES_AUTOSAVE_MAX_WAIT_SECONDS", _MAX_WAIT
    )


async def _autosaved(pilot, screen, profile, body: str) -> bool:
    """Whether the note's row holds ``body`` and the draft is clean, in time."""

    def _landed() -> bool:
        snapshot = screen._library_note_session.snapshot
        return (
            profile.note(profile.note_id)["content"] == body
            and snapshot is not None
            and not snapshot.dirty
            and not snapshot.saving
        )

    try:
        async with asyncio.timeout(_AUTOSAVE_WITHIN):
            while not _landed():
                await pilot.pause(0.05)
    except TimeoutError:
        return False
    return True


async def test_typing_after_keep_editing_on_a_refused_quit_is_autosaved(
    tmp_path, monkeypatch
):
    """Review route 2. Nothing but the autosave may write the fixed draft."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    saved = False
    burst = object()
    row: dict = {}
    shutdown_error: BaseException | None = None
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                title = screen.query_one("#library-note-title", Input)
                title.focus()
                title.cursor_position = len(title.value)
                await _type(pilot, " ")  # a trailing space: every save is refused
                prompt = await _ctrl_q(pilot, app, events)  # inside the debounce
                assert prompt is not None, "a refused save quit without asking"
                await pilot.pause(_MAX_WAIT + 0.5)  # the prompt outlasts the max wait
                await _keep_editing(pilot, app, screen, prompt)
                assert profile.note(profile.note_id)["content"] == _BODY

                title.focus()
                title.cursor_position = len(title.value)
                await pilot.press("backspace")  # fix the title
                await _type_at_end(pilot, body, "kept")

                saved = await _autosaved(pilot, screen, profile, _BODY + "kept")
                row = profile.note(profile.note_id)
                burst = screen._notes_state.autosave_burst
    except ZeroDivisionError as error:  # Textual awaits the dead timer task
        shutdown_error = error
    finally:
        profile.db.close_connection()

    assert events == [], "Keep editing must not quit"
    assert saved, (
        "typing after Keep editing was never autosaved: the row still holds "
        f"{row.get('content')!r}"
    )
    assert row["title"] == _TITLE
    assert burst is None, "the burst must end when its save starts"
    assert shutdown_error is None, "a dead 0 s timer made app shutdown raise"


async def test_a_note_left_dirty_by_a_rail_switch_is_autosaved_on_return(
    tmp_path, monkeypatch
):
    """Review route 3. The retained draft and the keys typed after it both land."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    saved = False
    row: dict = {}
    shutdown_error: BaseException | None = None
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, "a")
                assert screen._library_note_session.snapshot.dirty
                # Leave for another Browse row inside the debounce: the dirty
                # session is retained, unsaved, with its timer cancelled.
                screen.query_one("#library-row-browse-media", Button).press()
                await _until(
                    pilot,
                    lambda: not screen.query("#library-note-body"),
                    "the rail to leave Notes",
                )
                retained = screen._library_note_session.snapshot
                assert retained is not None and retained.dirty
                assert profile.note(profile.note_id)["content"] == _BODY
                await pilot.pause(_MAX_WAIT + 0.5)  # away past the max wait

                screen.query_one("#library-row-browse-notes", Button).press()
                await _until(
                    pilot,
                    lambda: bool(screen.query("#library-note-body")),
                    "the retained editor to come back",
                )
                await pilot.pause(0.2)
                body = screen.query_one("#library-note-body", TextArea)
                await _type_at_end(pilot, body, "bc")

                saved = await _autosaved(pilot, screen, profile, _BODY + "abc")
                row = profile.note(profile.note_id)
    except ZeroDivisionError as error:  # Textual awaits the dead timer task
        shutdown_error = error
    finally:
        profile.db.close_connection()

    assert events == []
    assert saved, (
        "typing after the rail switch was never autosaved: the row still holds "
        f"{row.get('content')!r}"
    )
    assert shutdown_error is None, "a dead 0 s timer made app shutdown raise"
