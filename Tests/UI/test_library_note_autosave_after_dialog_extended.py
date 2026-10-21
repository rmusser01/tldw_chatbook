"""The slower arms of ``test_library_note_autosave_after_dialog.py`` (TASK-34000.9).

Not in ``scripts/ui_pr_gate_census.txt``; runs with the full ``Tests/UI``
suite. The Move note dialog (the stress journey's repro, j6) and the 80x24
repeat of the F1 arm (AC#5's second size).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
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

                screen.query_one("#library-notes-placement-move", Button).press()
                await _until(
                    pilot,
                    lambda: isinstance(app.screen, LibraryNoteFolderTargetDialog),
                    "the Move note dialog to open",
                )
                assert screen._notes_state.autosave_timer is None
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

                await _open_help(pilot, app, screen)
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
