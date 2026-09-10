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
from textual.widgets import Button, Static

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module
from Tests.UI.test_library_file_notes_workspace import (
    _WorkspaceHarness,
    _production_workspace_context,
    _static_text,
    _wait_until,
)
from tldw_chatbook.config import get_cli_setting as real_get_cli_setting
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    FOLDER_FILES_EMPTY_COPY,
    LibraryFileNotesWorkspace,
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
async def test_a_relative_configured_sync_folder_is_not_offered(
    tmp_path, monkeypatch
) -> None:
    """Review round 2 (Qodo finding 1): the setting goes through validation.

    A relative ``[notes] sync_directory`` resolves against whatever
    directory the app happened to be launched from, so offering it by name
    would link a folder the user never chose.
    """
    (tmp_path / "relative-notes").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        workspace_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            "relative-notes" if (section, key) == ("notes", "sync_directory")
            else default
        ),
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        await pilot.pause()
        assert workspace._configured_sync_folder() is None
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


@pytest.mark.asyncio
async def test_an_abandoned_scan_cannot_overwrite_a_newer_changes_count(
    tmp_path,
) -> None:
    """PR #2549 review, finding 2: a stale worker's count stays out of the row.

    An abandoned scan's thread can still be inside its file loop when the
    next folder change resets the progress counter. Its next progress
    report used to land on the new attempt's row.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("note", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE):
        abandoned = workspace._root_generation
        workspace._root_generation += 1
        workspace._root_scan_entries = 0

        workspace._record_root_scan_progress(1240, generation=abandoned)
        assert workspace._root_scan_entries == 0

        workspace._record_root_scan_progress(
            7, generation=workspace._root_generation
        )
        assert workspace._root_scan_entries == 7
    await workspace.shutdown()
    replica.close()


# --- task-32174 (last-used picker start directory) -------------------------
#
# Folder files already opens at the current root when one is set; the gap
# was no root chosen yet (or an offline one) -- it fell straight to home
# instead of the last directory actually browsed. Keyed independently
# (``file_notes.browse``) from Import once and Keep a folder synced.


@pytest.mark.asyncio
async def test_folder_files_picker_opens_at_last_browsed_directory_without_a_root(
    tmp_path, monkeypatch
) -> None:
    """task-32174 AC#3: no root set yet -> the last-browsed directory wins."""
    remembered = tmp_path / "remembered"
    remembered.mkdir()

    monkeypatch.setattr(
        workspace_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(remembered) if (section, key) == ("file_notes", "browse")
            else default
        ),
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await pilot.pause()
        assert workspace._file_notes_browse_location() == remembered
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_folder_files_picker_falls_back_to_home_without_a_remembered_directory(
    monkeypatch,
) -> None:
    """AC#3: falls back to home when nothing has been browsed yet."""
    monkeypatch.setattr(workspace_module, "get_cli_setting", lambda *a, **k: None)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await pilot.pause()
        assert workspace._file_notes_browse_location() == Path.home()
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_folder_files_remembers_the_picked_root(tmp_path) -> None:
    """AC#3: picking a root becomes the next open's start directory."""
    picked = tmp_path / "vault"
    picked.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await pilot.pause()
        workspace._root_selected(picked)
        await _wait_until(
            pilot,
            lambda: real_get_cli_setting("file_notes", "browse", None)
            == str(picked),
            "the picked root was never remembered as the browse directory",
            # The only wait in this wave that crosses a worker thread AND a
            # real config read-modify-write, which serializes on the shared
            # write lock and grows with the accumulated bootstrap config.
            # The default ~3s budget failed once in a whole-file run.
            attempts=500,
        )
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_folder_files_picker_instance_opens_at_the_remembered_directory(
    tmp_path, monkeypatch
) -> None:
    """PR #2554 review: the production ``_open_root_picker`` must hand the
    remembered directory to the real ``SelectDirectory`` instance -- not
    merely resolve it in a helper nothing consults."""
    remembered = tmp_path / "remembered"
    remembered.mkdir()
    monkeypatch.setattr(
        workspace_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(remembered) if (section, key) == ("file_notes", "browse") else default
        ),
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    captured: dict = {}

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await pilot.pause()

        async def _fake_push(screen, callback=None):
            captured["screen"] = screen
            captured["callback"] = callback

        monkeypatch.setattr(workspace.app, "push_screen", _fake_push)
        await workspace._open_root_picker()

    assert isinstance(captured["screen"], SelectDirectory)
    assert Path(captured["screen"]._location) == remembered.resolve()
    assert captured["callback"] == workspace._root_selected
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_folder_files_refuses_an_unusable_remembered_directory(
    tmp_path, monkeypatch
) -> None:
    """A relative or traversing remembered value used to reach the picker
    after a bare ``is_dir()`` probe; it now falls back to home."""
    (tmp_path / "relative-dir").mkdir()
    monkeypatch.chdir(tmp_path)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await pilot.pause()
        for remembered in ("relative-dir", "../..", str(tmp_path / "deleted")):
            monkeypatch.setattr(
                workspace_module,
                "get_cli_setting",
                lambda *args, _value=remembered, **kwargs: _value,
            )
            assert workspace._file_notes_browse_location() == Path.home(), remembered
    await workspace.shutdown()
    replica.close()
