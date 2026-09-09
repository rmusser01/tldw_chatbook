"""Folder files is a mode of Notes, not a screen that replaces it (task-32136).

The critique's question was blunt: "Is Folder files a mode of Notes or a
different screen? The strip that switches into it disappears once you are
there, and the rail goes with it." The user decided: a mode. These tests
pin the wide layout that decision implies, and the empty state that has to
explain the mode before it asks for a folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import Tests.UI._optional_module_stubs  # noqa: F401
from textual.widgets import Button, Static

from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    FOLDER_FILES_EMPTY_COPY,
    LibraryFileNotesWorkspace,
)
import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _static_text,
    _wait_until,
)

WIDE = (235, 52)


@pytest.mark.asyncio
async def test_folder_files_keeps_the_rail_and_the_source_strip(tmp_path) -> None:
    """task-32136 AC1/AC4: the rail and both switches survive the mode change."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("note", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        screen = pilot.app.screen
        assert screen._notes_state.source == "files"

        rail = screen.query_one("#library-file-notes-rail")
        assert rail.display and rail.region.width > 0

        database = screen.query_one("#library-notes-source-database", Button)
        files = screen.query_one("#library-notes-source-files", Button)
        separator = screen.query_one("#library-notes-source-separator", Static)
        assert database.display and files.display and separator.display
        assert str(database.label) == "Library notes"
        assert str(files.label) == "Folder files"
        assert files.has_class("-selected")
        assert not database.has_class("-selected")

        # The back cue is no longer the only way out: the switch is.
        database.press()
        await _wait_until(
            pilot,
            lambda: screen._notes_state.source == "database",
            "the Library notes switch did not leave Folder files",
        )
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_empty_folder_files_explains_itself_and_offers_the_sync_folder(
    tmp_path, monkeypatch
) -> None:
    """task-32136 AC2: the empty state says what the mode is and offers a folder."""
    configured = tmp_path / "synced-notes"
    configured.mkdir()

    def sync_directory_setting(section, key=None, default=None):
        if (section, key) == ("notes", "sync_directory"):
            return str(configured)
        return default

    monkeypatch.setattr(
        workspace_module, "get_cli_setting", sync_directory_setting
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        assert _static_text(workspace, "#file-notes-empty-purpose") == (
            FOLDER_FILES_EMPTY_COPY
        )
        assert workspace.query_one("#file-notes-empty-purpose").display

        use_configured = workspace.query_one("#file-notes-use-sync-folder", Button)
        assert use_configured.display
        assert str(use_configured.label) == "Use synced-notes"

        # Review round 1: a folder with no ``name`` (the filesystem root)
        # used to render the button as a bare "Use ".
        monkeypatch.setattr(
            workspace_module,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                "/" if (section, key) == ("notes", "sync_directory") else default
            ),
        )
        workspace._update_root_surface()
        await pilot.pause()
        assert str(use_configured.label) == "Use /"
        monkeypatch.setattr(
            workspace_module, "get_cli_setting", sync_directory_setting
        )
        workspace._update_root_surface()
        await pilot.pause()
        assert str(use_configured.label) == "Use synced-notes"

        use_configured.press()
        await _wait_until(
            pilot,
            lambda: workspace.root == configured.resolve(),
            "the configured sync folder was never linked",
        )
        assert not workspace.query_one("#file-notes-empty-purpose").display
        assert not workspace.query_one("#file-notes-use-sync-folder", Button).display
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_no_configured_sync_folder_offers_no_button(
    tmp_path, monkeypatch
) -> None:
    """An unset (or stale) ``[notes] sync_directory`` offers nothing to press."""
    monkeypatch.setattr(
        workspace_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(tmp_path / "gone") if (section, key) == ("notes", "sync_directory")
            else default
        ),
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        await pilot.pause()
        assert workspace._configured_sync_folder() is None
        assert not workspace.query_one("#file-notes-use-sync-folder", Button).display
        assert workspace.query_one("#file-notes-empty-purpose").display
    await workspace.shutdown()
    replica.close()
