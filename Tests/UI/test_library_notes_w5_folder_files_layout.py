"""Folder-files pane geometry at the critique's three sizes (task-32614).

The navigator's title row is a ``Horizontal``, which Textual defaults to
``height: 1fr``, so it split the pane's spare rows with the tree: 19 rows of
it at 235x52 and 9 at 100x30, all but one blank -- the capture's "rows 6
through 26 are entirely blank, the Search box sits at row 27 and the tree
begins at row 30" exactly. And at 100x30 the panes divided 56/34, in which
an absolute vault path folded over SIX rows broken mid-token.

Every assertion here was first run against the unfixed tree and recorded
failing; the RED text is on the task.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _wait_until,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_FILE_NOTES_READER_PROFILE,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

pytestmark = pytest.mark.asyncio

WIDE = (235, 52)
COMPACT = (100, 30)
NARROW = (60, 24)
CRITIQUE_SIZES = (WIDE, COMPACT, NARROW)

#: A nested vault path long enough to overflow every pane under test.
DEEP_PATH = "Daily notes/2026/September/2026-09-14 morning pages and reflections.md"


def folder_files_vault(tmp_path: Path) -> Path:
    """Build a vault whose deepest path overflows every pane under test."""
    root = tmp_path / "vault"
    deep = root / "Daily notes" / "2026" / "September"
    deep.mkdir(parents=True)
    (deep / "2026-09-14 morning pages and reflections.md").write_text(
        "hi\n", encoding="utf-8"
    )
    for index in range(20):
        (root / f"note-{index:02d}.md").write_text(
            f"# note {index}\n", encoding="utf-8"
        )
    return root


def painted_lines(widget) -> list[str]:
    """Return what ``widget`` actually paints, row by row."""
    return [
        widget.render_line(row).text for row in range(max(widget.region.height, 0))
    ]


@pytest.mark.parametrize("size", CRITIQUE_SIZES)
async def test_the_folder_files_title_row_claims_one_row_not_a_share_of_the_pane(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """task-32614 AC#1/AC#4: the navigator fills the height it claims.

    Born red at 235x52 with ``height=19`` and at 100x30 with ``height=9``:
    ``Horizontal``'s default is ``height: 1fr``, so the title row and the
    tree split the pane's spare rows between them. That is the capture's
    "rows 6 through 26 are entirely blank -- the Search box sits at row 27
    and the tree begins at row 30" exactly.
    """
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=folder_files_vault(tmp_path), replica=replica)
    async with _production_workspace_context(workspace, size=size) as pilot:
        await pilot.pause()
        header = workspace.query_one("#file-notes-tree-header")
        tree = workspace.query_one("#file-notes-tree")
        navigator = workspace.query_one("#file-notes-navigator")
        assert header.outer_size.height == 1, (
            f"the title row took {header.outer_size.height} rows at {size}"
        )
        if navigator.display:
            # Everything the title row is not spending now belongs to the
            # tree: search row (3) + title (1) is the whole of the rest.
            assert tree.outer_size.height == navigator.outer_size.height - 4, (
                f"tree {tree.outer_size.height} of pane "
                f"{navigator.outer_size.height} at {size}"
            )
    await workspace.shutdown()
    replica.close()


@pytest.mark.parametrize("size", CRITIQUE_SIZES)
async def test_neither_folder_files_path_line_is_ever_folded_mid_token(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """task-32614 AC#2/AC#3/AC#4: one row each, elided, basename intact.

    Born red at 100x30: the breadcrumb painted four rows broken at
    "2026"/"-09-14" and the absolute path six, broken at
    "…m80n2j152t"/"9gw3w8qwk…".
    """
    root = folder_files_vault(tmp_path)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=size) as pilot:
        assert await workspace.open_path(DEEP_PATH)
        await _wait_until(
            pilot,
            lambda: workspace.current_path == DEEP_PATH,
            f"{DEEP_PATH} did not open",
        )
        workspace.query_one("#file-notes-manage", Button).press()
        await pilot.pause()
        await pilot.pause()
        for selector in ("#file-notes-breadcrumb", "#file-notes-exact-path"):
            painted = painted_lines(workspace.query_one(selector, Static))
            assert len(painted) == 1, f"{selector} took {len(painted)} rows at {size}"
            line = painted[0].rstrip()
            # The basename is the half a reader recognizes the file by, so
            # it is the half the elision keeps.
            assert line.endswith("reflections.md"), f"{selector} painted {line!r}"
            assert "…" in line or len(line) <= workspace.query_one(
                selector, Static
            ).content_region.width
    await workspace.shutdown()
    replica.close()


def test_folder_files_no_longer_holds_the_narrowest_work_pane_in_the_library() -> None:
    """task-32614 AC#3: the floor that collapses 60x24 now applies at 100x30.

    Born red at ``work_min_width=30`` -- the only Library destination under
    44 -- which is what let a 100-column terminal divide 56/34.
    """
    assert LIBRARY_FILE_NOTES_READER_PROFILE.work_min_width == 44
