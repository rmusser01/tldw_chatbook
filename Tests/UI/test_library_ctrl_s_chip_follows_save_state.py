"""The Ctrl+S chip follows a conflict that no key caused (TASK-34000.40, T9-i).

The footer chips Ctrl+S only while the key would write (ruling T9-j). A save
can stop being possible without the key: Folder files' poll notices the file
changed on disk under a dirty draft, and a Notes autosave meets a newer
version of the note in the database. In both the chip must leave by itself
(and, in Folder files, come back once the draft is dirty again) -- driven by
the save state, not by the key path.

Production ``TldwCli`` booted into Library: a real vault on disk under the
production Folder files workspace, and a real ChaChaNotes database whose row
is changed behind the open note. Real keys; painted footer text.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, TextArea

# Stubs first: they register the optional modules the app imports below probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _open_note,
    _scaled_autosave,
    _settings_without_splash,
    _type,
    _until,
)
from Tests.UI.test_library_ctrl_s_saves_in_every_editor import _footer
from Tests.UI.test_library_file_notes_workspace import _production_workspace_context
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_a_polled_folder_files_conflict_drops_the_chip_and_a_reload_restores_it(
    tmp_path,
):
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "today.md"
    note.write_text("first line\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root, replica=replica, autosave_delay=3600, poll_interval=0.3
    )
    async with _production_workspace_context(workspace, size=SIZE) as pilot:
        screen = pilot.app.screen
        assert await workspace.open_path("today.md")
        await _until(
            pilot, lambda: workspace.current_path == "today.md", "today.md to open"
        )
        editor = workspace.query_one("#file-notes-editor", TextArea)
        await _until(pilot, lambda: editor.region.height > 0, "the editor laid out")
        await pilot.click("#file-notes-editor")
        await _until(pilot, lambda: screen.focused is editor, "focus in the editor")
        await _type(pilot, "mine ")
        await _until(
            pilot,
            lambda: "ctrl+s save file" in _footer(pilot.app),
            "a dirty draft to chip Ctrl+S",
        )

        # No key: the poll finds the file changed on disk under the draft.
        note.write_text("someone else wrote this\n", encoding="utf-8")
        await _until(
            pilot, lambda: workspace.save_state == "conflict", "the poll's conflict"
        )
        await _until(
            pilot,
            lambda: "ctrl+s" not in _footer(pilot.app),
            "a polled conflict to drop the chip",
        )
        assert screen.focused is editor

        # Resolve by reloading the disk version (setup), then type again.
        workspace.query_one("#file-notes-recovery-reload", Button).press()
        confirm = None
        for _ in range(100):
            matches = workspace.query("#file-notes-reload-confirm")
            if matches and matches.first().display:
                confirm = matches.first()
                break
            await pilot.pause(0.05)
        assert confirm is not None, "no reload confirmation"
        confirm.press()
        await _until(
            pilot, lambda: workspace.save_state != "conflict", "the reload to resolve"
        )
        await pilot.click("#file-notes-editor")
        await _until(pilot, lambda: screen.focused is editor, "focus back")
        await _type(pilot, "again ")
        await _until(
            pilot,
            lambda: "ctrl+s save file" in _footer(pilot.app),
            "a new dirty draft to bring the chip back",
        )
    await workspace.shutdown()
    replica.close()


async def test_a_notes_autosave_conflict_drops_the_chip(tmp_path, monkeypatch):
    """The note's row changes in the database behind the open draft; the
    scaled autosave meets the newer version and lands in the conflict, whose
    footer tier offers no ctrl+s -- through the presentation seam every save
    outcome already re-registers the footer from."""
    _scaled_autosave(monkeypatch)
    app, profile = _library_app(tmp_path, monkeypatch, [], lambda notes: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await pilot.click("#library-note-body")
            await _until(pilot, lambda: screen.focused is body, "focus in the body")
            row = profile.note(profile.note_id)
            profile.db.update_note(
                profile.note_id,
                {"content": "written elsewhere"},
                expected_version=row["version"],
            )
            await _type(pilot, "x")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.in_conflict,
                "the autosave to meet the newer version",
                timeout=10.0,
            )
            await _until(
                pilot,
                lambda: "ctrl+s" not in _footer(pilot.app),
                "the autosave's conflict to drop the chip",
            )
    profile.db.close_connection()
