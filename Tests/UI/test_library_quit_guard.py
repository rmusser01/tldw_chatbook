"""Ctrl+Q saves unsaved Library edits, or asks first (TASK-34000.1, review N-01).

Ctrl+Q in Library ▸ Notes used to throw away whatever was typed since the
last autosave: the quit walk asks only the active screen's ``confirm_quit`` /
``prepare_for_quit``, ``LibraryScreen`` had neither, and its ``on_unmount``
cancels the pending debounced save.

This is the lean core that gates pull requests (``scripts/ui_pr_gate_census.txt``):
- the typed tail and a new note reach the database before exit;
- a refused save keeps a typing user in place;
- Ctrl+Q over a refused save asks, with Keep editing focused.

Each test is one real app boot, about 10-20 s. The UI Fast Lane is near its
20-minute cap (TASK-34353), so every other variant lives in
``test_library_quit_guard_extended.py``, which runs with the full
``Tests/UI`` suite.

These drive the real ``TldwCli`` and press the real keys. The quit walk, the
Library screen, its note session coordinator, the save port and a real
ChaChaNotes database on disk are all real. Only the irreversible shutdown is
replaced by a recorder, and it reads the DATABASE at the moment the approved
quit would have exited. Each assertion is therefore about what a relaunch
would find, not about widget text.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, TextArea

from Tests.UI.library_quit_guard_support import (
    _BODY,
    _DEBOUNCE,
    _MAX_WAIT,
    _TITLE,
    SIZE,
    _ctrl_q,
    _keep_editing,
    _library,
    _library_app,
    _new_blank_note,
    _no_autosave,
    _open_note,
    _prompt_title,
    _scaled_autosave,
    _settings_without_splash,
    _type,
    _type_at_end,
    _type_steadily,
    _until,
)
from Tests.app_module_patches import patch_app_global

#: ``bootstrap_profile`` as in ``test_roleplay_quit_guard.py``: a full-app boot
#: under the per-test profile redirect fails locally at ``_build_test_app`` with
#: ``RecoveryRequired: raw_source_selection_changed`` (lessons-testing-evidence).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


# --- AC#1 / AC#5: a clean flush persists the typed text, with no prompt ------


async def test_ctrl_q_inside_the_autosave_window_persists_the_typed_tail(
    tmp_path, monkeypatch
):
    """Verify capture 03: a tail typed just before Ctrl+Q is in the DB at exit."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " verifytailn01")
            assert screen._library_note_session.snapshot.dirty

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is None, "a flushable edit must quit without asking"
            [(kind, row)] = events
            assert kind == "quit"
            assert row["content"] == _BODY + " verifytailn01", (
                "Ctrl+Q exited before the typed tail reached the database"
            )
            assert row["title"] == _TITLE
    profile.db.close_connection()


async def test_ctrl_q_on_a_new_note_keeps_its_typed_title_and_body(
    tmp_path, monkeypatch
):
    """Capture 10 / j6: the new note persists as typed, not as an empty Untitled."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.rows()
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _new_blank_note(screen, pilot)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()
            await _type(pilot, "riley new ctrlq")
            await _type_at_end(
                pilot, screen.query_one("#library-note-body", TextArea), "body typed"
            )

            assert await _ctrl_q(pilot, app, events) is None
            [(kind, rows)] = events
            assert kind == "quit"
            assert ("riley new ctrlq", "body typed") in rows, rows
            assert len(rows) == 3, f"expected the two seeds plus the new note: {rows}"
            assert not [row for row in rows if row[0] == "Untitled"], rows
    profile.db.close_connection()


# --- AC#2 + the N-07 core guard: one refused draft ---------------------------


async def test_a_refused_save_keeps_the_typist_in_place_and_ctrl_q_asks(
    tmp_path, monkeypatch
):
    """A title the save refuses (a trailing space) vetoes every save.

    The steady typing runs past the scaled max wait, so autosaves fire, and
    are vetoed, while the user is still typing (review N-07). If a veto moved
    focus to the title, the next body keys would land there and a later save
    would store them as the title. The final veto is awaited on the status
    (bounded), so a run where no autosave was vetoed fails loudly instead of
    passing.

    Then Ctrl+Q cannot save the draft either, so it asks: Keep editing is
    focused and returns with the text intact, and only Discard and quit exits,
    saving nothing.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            title = screen.query_one("#library-note-title", Input)
            title.focus()
            title.cursor_position = len(title.value)
            await _type(pilot, " ")  # the save refuses a trailing space
            body.focus()
            body.move_cursor(body.document.end)
            await pilot.pause()

            burst = "abcdefghijkl"  # ~3 s of steady keys: past the 2 s max wait
            await _type_steadily(pilot, burst, lambda: None)
            # Typing stopped, so the armed autosave fires and is vetoed; its
            # "validation" state then stays until the next key. Stop waiting
            # early if focus already left the body (or keys reached the title):
            # that is the defect itself, and the asserts below name it.
            await _until(
                pilot,
                lambda: (
                    screen._notes_state.autosave_state == "validation"
                    or screen.focused is not body
                    or title.value != _TITLE + " "
                ),
                "a vetoed autosave, or focus leaving the body",
                timeout=_DEBOUNCE + _MAX_WAIT + 5.0,
            )
            await pilot.pause()
            await pilot.pause(0.2)  # the unguarded veto focused after a refresh
            assert screen.focused is body, (
                "a vetoed autosave moved focus to "
                f"{getattr(screen.focused, 'id', None)!r}"
            )
            assert title.value == _TITLE + " ", (
                f"keys typed into the body landed in the title: {title.value!r}"
            )
            assert screen._notes_state.autosave_state == "validation", (
                "no autosave was vetoed: the focus guard was never exercised"
            )
            status = screen._library_note_session.snapshot.status_message
            assert "whitespace" in status, status
            await _type(pilot, "xyz")
            assert body.text == _BODY + burst + "xyz", (
                "keys typed during the burst landed outside the body"
            )
            assert title.value == _TITLE + " "
            row = profile.note(profile.note_id)
            assert (row["title"], row["content"]) == (_TITLE, _BODY)

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "a vetoed save quit silently"
            assert events == []
            assert _prompt_title(prompt) == (
                f'Quit and discard unsaved changes to "{_TITLE}"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert events == []
            assert body.text == _BODY + burst + "xyz"
            assert title.value == _TITLE + " "
            assert screen._library_note_session.snapshot.dirty

            prompt = await _ctrl_q(pilot, app, events)
            assert prompt is not None, "the second Ctrl+Q must ask again"
            await pilot.click("#confirm-button")
            await _until(pilot, lambda: bool(events), "Discard and quit to quit")
            [(kind, row)] = events
            assert kind == "quit"
            assert (row["title"], row["content"]) == (_TITLE, _BODY), (
                "Discard and quit must not save the discarded draft"
            )
    profile.db.close_connection()
