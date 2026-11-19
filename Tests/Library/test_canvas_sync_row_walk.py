"""The shared Library row walker treats the Notes tree as ONE list (TASK-34000.28, N-22).

Review N-22 (notes-library-ux-review-2026-10-02): ``g`` focused ``▸
Agent_Lessons`` and Down did nothing -- folder rows (``library-notes-folder-
row``) and the tree's "more" pager (``library-notes-tree-pager``) were not in
``_LIBRARY_LIST_ROW_CLASSES``, so ``_move_library_list_row_focus`` returned
False on them and Textual's Button ignored the key. Folder rows, note rows and
the pager are siblings under ``#library-notes-list``; one table entry per
missing class makes them one walk.

Pure Pilot tests against a bare Textual ``App`` -- the walker is a module-level
function so it can be proven without a ``LibraryScreen`` (its docstring says
so). No database, no profile.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button

from tldw_chatbook.UI.Library_Modules.canvas_sync import (
    _move_library_list_row_focus,
    _previous_folder_row_above,
)


class _NotesTreeHost(App[None]):
    """The Notes tree's exact sibling shape: folder rows, note rows, a pager.

    ``tree_depth`` mirrors what ``LibraryNotesCanvas._set_tree_row_metadata``
    stamps on every tree row (the walker compares integers, not indent
    spaces).
    """

    def compose(self) -> ComposeResult:
        rows = (
            ("journal", "library-notes-folder-row", 0),
            ("projects", "library-notes-folder-row", 0),
            ("thesis", "library-notes-folder-row", 1),
            ("thesis-note", "library-notes-row library-notes-tree-note-row", 2),
            ("more", "library-notes-tree-pager library-canvas-action", 2),
            ("unfiled", "library-notes-folder-row library-notes-unfiled-row", 0),
            ("loose", "library-notes-row library-notes-tree-note-row", 1),
        )
        with Vertical(id="library-notes-list"):
            for row_id, classes, depth in rows:
                button = Button(row_id, id=row_id, classes=classes)
                button.tree_depth = depth
                yield button


@pytest.mark.asyncio
async def test_down_walks_folder_rows_note_rows_and_the_pager_in_order() -> None:
    """AC#3: from a folder row, Down reaches every sibling kind in visual order.

    Base: the walker returns False on the folder row (its class is not in
    the table), so focus never leaves ``#journal``.
    """
    app = _NotesTreeHost()
    async with app.run_test() as pilot:
        app.query_one("#journal", Button).focus()
        await pilot.pause()
        visited: list[str] = []
        for _ in range(6):
            assert _move_library_list_row_focus(app.focused, "down") is True, (
                f"the walker refused Down on {app.focused!r}"
            )
            await pilot.pause()
            visited.append(app.focused.id)
        assert visited == [
            "projects",
            "thesis",
            "thesis-note",
            "more",
            "unfiled",
            "loose",
        ]
        # Up from the pager lands back on the note row above it.
        app.query_one("#more", Button).focus()
        await pilot.pause()
        assert _move_library_list_row_focus(app.focused, "up") is True
        await pilot.pause()
        assert app.focused.id == "thesis-note"


@pytest.mark.asyncio
async def test_previous_folder_row_above_is_the_nearest_shallower_folder() -> None:
    """Left on a collapsed folder row climbs to its PARENT, not to any row above.

    The parent is the nearest earlier folder-row sibling with a smaller depth:
    a sibling folder at the same depth and the note rows in between are
    skipped, and a root folder has no parent.
    """
    app = _NotesTreeHost()
    async with app.run_test():
        thesis = app.query_one("#thesis", Button)
        projects = app.query_one("#projects", Button)
        journal = app.query_one("#journal", Button)
        unfiled = app.query_one("#unfiled", Button)
        assert _previous_folder_row_above(thesis) is projects
        assert _previous_folder_row_above(projects) is None
        assert _previous_folder_row_above(journal) is None
        # Unfiled sits after Thesis (depth 1) but is itself at depth 0.
        assert _previous_folder_row_above(unfiled) is None
        # The helper reads depth only, so a note row's own folder is found the
        # same way (nothing binds a key to that yet).
        assert _previous_folder_row_above(app.query_one("#loose", Button)) is unfiled
        # A row that is not in the tree (no parent) has no parent folder.
        assert _previous_folder_row_above(Button("stray")) is None
