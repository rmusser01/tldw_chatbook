"""TASK-34000.14: the journey wait witnesses paint, not just mounted rows.

Exercise the actual journey helpers on the production Notes canvas and CSS.
AST loading avoids importing the separate full-app/service journey harness.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    empty_notes_slice,
)
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.UI.Library_Modules.library_notes_state import LibraryNotesState
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas
from tldw_chatbook.css.build_css import screen_css_paths

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_ROOT = Path(__file__).resolve().parents[2]
_TITLE = "Capture from console"
_LIST_STATE = LibraryNotesListState(
    rows=(), header_copy="Notes (1)", status_copy="", empty_copy="", total_count=1
)


def _journey_helpers() -> SimpleNamespace:
    """Execute the current helper bodies without the full-app imports."""
    namespace = {"asyncio": asyncio, "NotesBranchKey": NotesBranchKey}
    for filename, names in (
        ("library_quit_guard_support.py", {"_SETTLE_SECONDS", "_until"}),
        (
            "test_library_notes_tree_keeps_folders_after_create.py",
            {
                "FOLDERS",
                "ROOT_FOLDERS",
                "ROOT_PLACEMENTS",
                "_tree_settled",
                "_painted_list",
                "_assert_folders_above_unfiled",
            },
        ),
    ):
        path = Path(__file__).with_name(filename)
        statements = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in names
                    for target in node.targets
                )
            )
        ]
        exec(
            compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return SimpleNamespace(**namespace)


class _CanvasApp(App[None]):
    css_root = _ROOT / "tldw_chatbook/css"
    CSS_PATH = [
        str(screen_css_paths(css_root)[0]),
        str(css_root / "tldw_cli_modular.tcss"),
        str(screen_css_paths(css_root)[1]),
    ]

    def __init__(self, *, folders: bool) -> None:
        super().__init__()
        self.folders = folders

    def compose(self) -> ComposeResult:
        rows = (
            [
                LibraryNotesTreeRow(
                    placement_id=f"folder:{name}",
                    kind="folder",
                    label=name,
                    depth=0,
                    folder_id=name,
                )
                for name in ("Journal", "Meetings", "Projects")
            ]
            if self.folders
            else []
        )
        rows.extend(
            (
                LibraryNotesTreeRow(
                    placement_id="virtual:unfiled",
                    kind="unfiled",
                    label="Unfiled",
                    depth=0,
                    expanded=True,
                ),
                LibraryNotesTreeRow(
                    placement_id="note:1",
                    kind="note",
                    label=_TITLE,
                    depth=1,
                    note_id="1",
                ),
            )
        )
        yield LibraryNotesCanvas(
            _LIST_STATE,
            mode="list",
            tree_projection=LibraryNotesTreeProjection(rows=tuple(rows)),
            id="library-notes-canvas",
        )

    def on_mount(self) -> None:
        state = LibraryNotesState()
        kinds = ("placements", "folders") if self.folders else ("placements",)
        state.tree_branches = {
            key: replace(empty_notes_slice(key), freshness="fresh")
            for kind in kinds
            for key in (NotesBranchKey(None, kind),)
        }
        self.screen._notes_state = state


@pytest.mark.parametrize(
    "folders", [True, False], ids=["folders-present", "folders-absent"]
)
async def test_tree_settle_waits_for_paint_without_hiding_missing_folders(
    monkeypatch, folders: bool
) -> None:
    """A fixed post-state pause returns while the replacement has no layout."""
    helpers = _journey_helpers()
    app = _CanvasApp(folders=folders)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = app.screen
        await helpers._until(
            pilot,
            lambda: bool(screen.query("#library-notes-list"))
            and "▾ Unfiled" in helpers._painted_list(app, screen),
            "the initial list to paint",
            timeout=8.0,
        )
        previous = screen.query_one("#library-notes-list")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        mounted = asyncio.Event()
        refresh_layout = screen._refresh_layout
        monkeypatch.setattr(screen, "_refresh_layout", lambda *args, **kwargs: None)
        canvas.queue_after_recompose(mounted.set)
        canvas.refresh(recompose=True)
        async with asyncio.timeout(8.0):
            await mounted.wait()
        current = screen.query_one("#library-notes-list")
        assert current is not previous and current.is_attached
        assert current.region.area == 0
        assert helpers._painted_list(app, screen) == ""

        pause_returned = asyncio.Event()

        async def observed_pause(delay=None):
            await pilot.pause(delay)
            pause_returned.set()

        waiter = asyncio.create_task(
            helpers._tree_settled(screen, SimpleNamespace(pause=observed_pause))
        )
        try:
            # Await a real helper pause, not a wall-clock guess about its work.
            # The old fixed pause finishes here despite the still-empty crop.
            async with asyncio.timeout(8.0):
                await pause_returned.wait()
            assert (
                not waiter.done()
            ), "tree settle returned before the current list painted"
            monkeypatch.setattr(screen, "_refresh_layout", refresh_layout)
            screen.refresh(layout=True)
            async with asyncio.timeout(8.0):
                await waiter
            assert screen.query_one("#library-notes-list") is current
            painted = helpers._painted_list(app, screen)
            if folders:
                helpers._assert_folders_above_unfiled(painted, note_title=_TITLE)
            else:
                # Absence is the original defect: readiness must let the
                # unchanged semantic assertion name it, rather than waiting
                # for those expected folder labels or passing an empty crop.
                with pytest.raises(AssertionError, match="Journal.*not in the tree"):
                    helpers._assert_folders_above_unfiled(painted, note_title=_TITLE)
        finally:
            monkeypatch.setattr(screen, "_refresh_layout", refresh_layout)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
