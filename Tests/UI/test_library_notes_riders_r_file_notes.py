"""Folder files riders: the pre-link rail (task-32173) and wait polish (task-32180).

task-32136 shipped "Folder files is a mode of Notes" only after a folder is
linked -- ``#file-notes-body`` was display-gated on a linked root, so the
empty state dropped to a full-width onboarding step with no rail. task-32180
collects the loose ends of the wave-1 wait work: the busy row's geometry was
never pinned below 120 columns, and ``Use <folder>`` only ever read the
legacy ``[notes] sync_directory`` key.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module
from Tests.UI.test_library_crit8_waits import (  # noqa: F401
    _busy_row_cancel_labels,
    _start_blocked_root_change,
    blocked_root_change,
)
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _wait_until,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

WIDE = (235, 52)
COMPACT = (60, 24)


def _shell_panes(workspace: LibraryFileNotesWorkspace) -> tuple:
    """Every pane the reader shell owns, rail first."""
    shell = workspace._reader_shell
    assert shell is not None
    return (
        shell.library,
        shell.library_grip,
        shell.items,
        shell.items_grip,
        shell.work,
    )


@pytest.mark.asyncio
async def test_folder_files_keeps_the_rail_before_a_folder_is_linked() -> None:
    """task-32173 AC1/AC4: the mode reads as a mode from the first frame."""
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        screen = pilot.app.screen
        rail = screen.query_one("#library-file-notes-rail")
        await _wait_until(
            pilot,
            lambda: rail.display and rail.region.width > 0,
            "the Library rail never painted in the unlinked Folder files state",
        )

        # Only the panes that need a folder wait for one.
        _library, _library_grip, items, items_grip, work = _shell_panes(workspace)
        assert not items.display
        assert not items_grip.display
        assert not work.display
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_compact_folder_files_paints_no_shell_before_linking() -> None:
    """task-32173 AC2: compact terminals keep the full-width empty state.

    The adaptive resolver already closes the rail below the Library's
    compact breakpoint, so at 60 columns the unlinked body holds nothing --
    exactly what the hidden body used to give.
    """
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=COMPACT) as pilot:
        await pilot.pause()
        assert not any(pane.display for pane in _shell_panes(workspace))
        assert workspace.query_one("#file-notes-empty-purpose").display
        assert workspace.query_one("#file-notes-choose-root", Button).display
    await workspace.shutdown()
    replica.close()


@pytest.mark.parametrize("start_linked", [True, False], ids=["linked", "unlinked"])
@pytest.mark.asyncio
async def test_the_slow_wait_row_keeps_every_control_on_pane_at_60_columns(
    blocked_root_change,  # noqa: F811  (the imported fixture, by name)
    start_linked,
) -> None:
    """task-32180 AC1: the busy row's geometry, pinned where it was tightest.

    Both starting states, because they take different CSS paths: the
    ``-empty-root`` class gives the status ``width: auto`` so the empty
    state's short prompt hugs its button (task-2850), and a wait line
    wearing that class hugs its own full length and shoves the decision
    controls off the row (fix round 1). task-32173 made the unlinked case
    reachable in the first place.
    """
    old_root, new_root, blocked = blocked_root_change
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root if start_linked else None, replica=replica
    )
    async with _production_workspace_context(workspace, size=COMPACT) as pilot:
        wait = await _start_blocked_root_change(pilot, workspace, blocked, new_root)
        wait.started_at -= 5.0
        workspace._record_root_scan_progress(
            1240, generation=workspace._root_generation
        )
        workspace._update_root_surface()
        await pilot.pause()
        await pilot.pause()

        row = workspace.query_one("#file-notes-root-row")
        visible = [button for button in row.query(Button) if button.display]
        assert [str(button.label) for button in visible] == [
            "Cancel",
            "Keep waiting",
            "Choose another",
        ]
        off_pane = [
            (str(button.label), button.region)
            for button in visible
            if button.region.right > row.region.right
            or button.region.x < row.region.x
            or button.region.width == 0
        ]
        assert off_pane == [], f"row={row.region!r}"
        # task-32121 AC5 holds at this width too.
        assert _busy_row_cancel_labels(workspace) == ["Cancel"]
        # The row still says what it is doing.
        status = workspace.query_one("#file-notes-root-status")
        assert status.region.width > 0
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_use_folder_offers_the_modern_file_notes_root(
    tmp_path, monkeypatch
) -> None:
    """task-32180 AC2: the key this mode itself writes is offered first."""
    modern = tmp_path / "modern-vault"
    modern.mkdir()
    legacy = tmp_path / "legacy-sync"
    legacy.mkdir()

    def both_keys(section, key=None, default=None):
        if (section, key) == ("file_notes", "root"):
            return str(modern)
        if (section, key) == ("notes", "sync_directory"):
            return str(legacy)
        return default

    monkeypatch.setattr(workspace_module, "get_cli_setting", both_keys)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        await pilot.pause()
        assert workspace._configured_sync_folder() == modern
        button = workspace.query_one("#file-notes-use-sync-folder", Button)
        assert button.display
        assert str(button.label) == "Use modern-vault"

        # The legacy key is still the fallback for a profile that never
        # linked a folder in this mode.
        monkeypatch.setattr(
            workspace_module,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                str(legacy) if (section, key) == ("notes", "sync_directory") else None
            ),
        )
        workspace._update_root_surface()
        await pilot.pause()
        assert str(button.label) == "Use legacy-sync"
    await workspace.shutdown()
    replica.close()
