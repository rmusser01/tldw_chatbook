"""The explicit paths still route to the vetoed field (TASK-34000.10, AC#3).

Companion to ``test_library_note_autosave_veto_in_place.py`` (the gated core);
not in the PR gate census. Both arms are green on base: the explicit Save and
the refused exit are the paths TASK-34000.10 leaves routing and focusing.
They are pinned here so the focus-independent autosave guard can never be
"fixed" by removing the explicit routing too.
"""

from __future__ import annotations

import pytest

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _open_note,
    _scaled_autosave,
    _settings_without_splash,
    _until,
)
from Tests.UI.test_library_note_autosave_veto_in_place import (
    _INVALID,
    _open_info,
    _set_long_title,
    _vetoed,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


# --- AC#3: an explicit Save, or leaving the note, still routes ---------------


async def test_an_explicit_save_still_routes_to_the_vetoed_field(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                _set_long_title(screen)
                await _open_info(pilot, screen)
                await _vetoed(pilot, screen)

                await pilot.click("#library-note-save")
                await _until(
                    pilot,
                    lambda: getattr(screen.focused, "id", None) == "library-note-title",
                    "Save to route to the vetoed title",
                )
                assert screen._notes_state.context is False
                assert screen.query_one("#library-note-editor-region").display
                assert screen._notes_state.autosave_state == "validation"
                assert screen.query_one("#library-note-title").has_class(_INVALID)
                assert profile.note(profile.note_id)["title"] == _TITLE
    finally:
        profile.db.close_connection()
    assert events == []


async def test_leaving_the_note_with_a_veto_standing_still_routes(
    tmp_path, monkeypatch
):
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda notes: None)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                await _open_note(screen, pilot, _TITLE, profile.note_id)
                _set_long_title(screen)
                await _open_info(pilot, screen)
                await _vetoed(pilot, screen)

                # Escape from Info closes Info (focus to the Info button); that
                # is not an exit, and the standing autosave veto still routes
                # nothing. The second Escape is the attempt to leave the note.
                await pilot.press("escape")
                await _until(
                    pilot,
                    lambda: screen._notes_state.context is False
                    and getattr(screen.focused, "id", None) == "library-note-context",
                    "Escape to close Info",
                )
                await pilot.pause(0.3)
                assert getattr(screen.focused, "id", None) == "library-note-context"
                assert screen._notes_state.autosave_state == "validation"

                await pilot.press("escape")
                await _until(
                    pilot,
                    lambda: getattr(screen.focused, "id", None) == "library-note-title",
                    "Escape's refused exit to route to the vetoed title",
                )
                assert screen._notes_state.view == "editor", "Escape left the note"
                assert screen._notes_state.context is False
                assert screen.query_one("#library-note-editor-region").display
                assert profile.note(profile.note_id)["title"] == _TITLE
    finally:
        profile.db.close_connection()
    assert events == []
