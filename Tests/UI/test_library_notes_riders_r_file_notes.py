"""Folder files riders: the pre-link rail (task-32173) and wait polish (task-32180).

task-32136 shipped "Folder files is a mode of Notes" only after a folder is
linked -- ``#file-notes-body`` was display-gated on a linked root, so the
empty state dropped to a full-width onboarding step with no rail.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
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
