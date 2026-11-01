"""The slower arms of ``test_library_note_autosave_after_dialog.py`` (TASK-34000.9).

Not in ``scripts/ui_pr_gate_census.txt``; runs with the full ``Tests/UI``
suite. The Move note dialog (the stress journey's repro, j6), the 80x24
repeat of the F1 arm (AC#5's second size), and the two resume vetoes the
review added (fix round 1): a note parked on another rail row is not saved
behind the user's back, and a failed write is not retried behind a dialog.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _MAX_WAIT,
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _open_note,
    _scaled_autosave,
    _settings_without_splash,
    _type_at_end,
    _until,
)
from Tests.UI.test_library_note_autosave_after_dialog import (
    _autosaved,
    _close_dialog,
    _open_help,
)
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderTargetDialog,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

COMPACT = (80, 24)


def _text(screen, selector: str) -> str:
    return " ".join(str(screen.query_one(selector, Static).render()).split())


async def test_the_move_note_dialog_inside_the_autosave_window_still_saves_the_edit(
    tmp_path, monkeypatch
):
    """j6: type, open Move note from the list beside the editor, Escape."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    saved = False
    row: dict = {}
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                # Opening the note from its row selected that placement, so
                # the list's "Move note" action targets the open note.
                selected = screen._selected_library_notes_tree_row()
                assert selected is not None and selected.kind == "note", selected
                assert selected.note_id == profile.note_id
                await _type_at_end(pilot, body, " after move")

                await pilot.click("#library-notes-placement-move")
                await _until(
                    pilot,
                    lambda: isinstance(app.screen, LibraryNoteFolderTargetDialog),
                    "the Move note dialog to open",
                )
                assert screen._notes_state.autosave_timer is None
                assert profile.note(profile.note_id)["content"] == _BODY
                await _close_dialog(pilot, app, screen)

                saved = await _autosaved(pilot, screen, profile, _BODY + " after move")
                row = profile.note(profile.note_id)
    finally:
        profile.db.close_connection()

    assert events == []
    assert saved, (
        "the edit typed before Move note was never autosaved after the dialog "
        f"closed: the row still holds {row.get('content')!r}"
    )


async def test_f1_help_inside_the_autosave_window_still_saves_the_edit_at_80x24(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    saved = False
    row: dict = {}
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=COMPACT) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, " compact")

                await _open_help(pilot, app, screen, profile)
                await _close_dialog(pilot, app, screen)

                saved = await _autosaved(pilot, screen, profile, _BODY + " compact")
                row = profile.note(profile.note_id)
    finally:
        profile.db.close_connection()

    assert events == []
    assert saved, (
        "at 80x24 the edit typed before F1 was never autosaved: the row still "
        f"holds {row.get('content')!r}"
    )


# --- Fix round 1 (I-1): a note parked on another rail row stays parked -----


async def test_a_dialog_on_the_media_row_does_not_save_a_parked_note(
    tmp_path, monkeypatch
):
    """Switching to Media keeps the dirty session and cancels its timer on
    purpose; autosave resumes only on return to Notes. A dialog opened and
    closed on Media must not re-arm it there: a hidden save could flip the
    parked preview state, and a veto could route to a field that is not on
    screen."""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    saved_on_return = False
    row_while_parked: dict = {}
    timer_after_dialog: object = None
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, " parked")
                screen.query_one("#library-row-browse-media", Button).press()
                # Fix round 2: F1 pressed the instant the editor left the DOM
                # was swallowed while the Media route was still settling
                # (reviewer run: "timed out waiting for F1 help to open").
                # Wait for the route to be mounted and for focus to have
                # settled on it before the key.
                await _until(
                    pilot,
                    lambda: (
                        bool(screen.query(".library-media-route"))
                        and not screen.query("#library-note-body")
                        and screen.focused is not None
                    ),
                    "the Media route to settle",
                )
                await pilot.pause()
                parked = screen._library_note_session.snapshot
                assert parked is not None and parked.dirty
                assert screen._notes_state.autosave_timer is None

                await _open_help(pilot, app, screen, profile)
                await _close_dialog(pilot, app, screen)
                await pilot.pause()  # the resume handler has run by now
                timer_after_dialog = screen._notes_state.autosave_timer
                await pilot.pause(_MAX_WAIT + 1.0)  # past any re-armed delay
                row_while_parked = profile.note(profile.note_id)
                assert screen._library_note_session.snapshot.dirty

                screen.query_one("#library-row-browse-notes", Button).press()
                await _until(
                    pilot,
                    lambda: bool(screen.query("#library-note-body")),
                    "the retained editor to come back",
                )
                saved_on_return = await _autosaved(
                    pilot, screen, profile, _BODY + " parked"
                )
    finally:
        profile.db.close_connection()

    assert events == []
    assert timer_after_dialog is None, "a dialog on Media re-armed a parked note"
    assert row_while_parked["content"] == _BODY, (
        "a note parked on Media was saved behind the user's back: "
        f"{row_while_parked['content']!r}"
    )
    assert saved_on_return, "returning to Notes did not resume the parked autosave"


# --- Fix round 1 (I-2): a failed write is not retried behind a dialog ------


async def test_a_failed_write_keeps_its_status_and_is_not_retried_after_a_dialog(
    tmp_path, monkeypatch
):
    """One write fails. Both status channels say so; F1 then Esc must leave
    them saying so, with no timer and no retry, until Save or the next
    keystroke. (A retry at resume could also write a draft that "Discard and
    quit" had just discarded, since a failed flush leaves this state.)"""
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    real_save = profile.scope_service.save_note
    saves: list[str] = []
    failures = {"left": 1}

    async def _save_failing_once(**kwargs):
        if kwargs.get("note_id") is not None and failures["left"] > 0:
            failures["left"] -= 1
            saves.append("failed")
            raise RuntimeError("disk is busy")
        saves.append("ok")
        return await real_save(**kwargs)

    monkeypatch.setattr(profile.scope_service, "save_note", _save_failing_once)
    before: tuple[str, str] = ("", "")
    after: tuple[str, str] = ("", "")
    state_after = ""
    timer_after: object = None
    row: dict = {}
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                await _type_at_end(pilot, body, " fails")
                await _until(
                    pilot,
                    lambda: screen._notes_state.autosave_state == "error",
                    "the autosave to fail",
                )
                await pilot.pause()
                before = (
                    _text(screen, "#library-note-status"),
                    _text(screen, "#library-note-work-authority"),
                )
                assert "failed" in before[0].lower(), before
                assert "failed" in before[1].lower(), before

                await _open_help(pilot, app, screen, profile)
                await _close_dialog(pilot, app, screen)
                await pilot.pause()
                state_after = screen._notes_state.autosave_state
                timer_after = screen._notes_state.autosave_timer
                after = (
                    _text(screen, "#library-note-status"),
                    _text(screen, "#library-note-work-authority"),
                )
                await pilot.pause(_MAX_WAIT + 1.0)  # a retry would have landed
                row = profile.note(profile.note_id)
    finally:
        profile.db.close_connection()

    assert events == []
    assert state_after == "error", f"resume cleared the failure: {state_after!r}"
    assert timer_after is None, "resume armed a retry of a failed write"
    assert after == before, f"the failure stopped showing: {before} -> {after}"
    assert saves == ["failed"], f"a retry fired behind the user's back: {saves}"
    assert row["content"] == _BODY
