"""TASK-1479: the Evals results-grid export path drives a real ``FileSave``
dialog through a keyboard-only flow -- the dialog seeds the Input with a
default filename, and the user should be able to press Enter immediately to
confirm it. Live UAT at 235x52 found the filename Input rendered essentially
invisible (observed ~5 painted columns), and initial focus landed on the
directory listing so that first Enter activated ``..`` instead of confirming.

Two independent defects, exercised here under the REAL app CSS bundle
(mirrors ``_CardHarnessAppWithBundledCSS`` in test_console_mcp_approval.py):
a bare test ``App`` with no ``CSS_PATH`` never loads
``tldw_cli_modular.tcss``'s bare, unscoped ``Select { width: 100%; }`` rule
(``features/_conversations.tcss``), which Textual's cascade always ranks
ABOVE any widget's own ``DEFAULT_CSS`` regardless of selector specificity --
so it silently wins over ``BaseFileDialog``'s own ``Select`` rule and
squeezes the sibling filename Input down to a few columns. This is the same
defect class already fixed for ``#mcp-tools-filter-server-slot Select`` /
``.approval-row-decision`` elsewhere in this bundle; the width collapse only
reproduces with the bundle loaded, while the focus defect reproduces even in
a bare host app.

A third, related defect surfaced while writing these tests: ``file_dialog.py``'s
``_confirm_file``/``_select_file`` read the filename back via a bare
``self.query_one(Input)``, which is ambiguous -- the screen also carries a
hidden ``#path-input`` (Ctrl+L) and ``#search-input`` (Ctrl+F), both earlier
in the compose tree than the filename Input, so the unscoped query silently
grabbed one of those instead. Even with focus correctly on the filename
Input, pressing Enter read back the *other*, empty Input and rejected with
"A file must be chosen" -- so the keyboard export path was unusable even
before considering the width/focus bugs. ``test_enter_immediately_after_mount_confirms_default_path``
below pins the fix for all three at once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

import tldw_chatbook
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave, Filters
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import InputBar
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

_BUNDLED_STYLESHEET = Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"


def _export_filters() -> Filters:
    """Mirrors ``ResultsGrid.action_export``'s exact three filters
    (tldw_chatbook/UI/Evals/results_grid.py:781-788), so this test exercises
    the same call shape the Evals export path actually drives."""
    return Filters(
        ("JSON (full run group)", lambda p: p.suffix.lower() == ".json"),
        ("CSV (active lens)", lambda p: p.suffix.lower() == ".csv"),
        ("All files", lambda p: True),
    )


class _DialogHost(App[None]):
    """Loads the real generated bundle as ``CSS_PATH`` so the InputBar's
    Input/Select contest the actual CSS-origin priority battle exactly as
    they do in the live Console/Evals screens (mirrors
    ``_CardHarnessAppWithBundledCSS`` in test_console_mcp_approval.py)."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, dialog):
        super().__init__()
        self._dialog = dialog
        self.result: object = "NOT_SET"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        def _capture(result):
            self.result = result

        await self.push_screen(self._dialog, callback=_capture)


def _make_save_dialog(tmp_path: Path) -> FileSave:
    return FileSave(
        location=str(tmp_path),
        title="Export results grid",
        default_file="mybench.json",
        filters=_export_filters(),
    )


def _filename_input(dialog) -> Input:
    """The filename Input, scoped through InputBar so the hidden
    ``#path-input``/``#search-input`` widgets elsewhere on the screen can't
    be picked up by an unscoped ``query_one(Input)``."""
    return dialog.query_one(InputBar).query_one(Input)


@pytest.mark.asyncio
async def test_filesave_input_seeds_default_filename_and_is_not_starved(tmp_path):
    """Regression guard for the ~5-column collapse observed live at 235x52:
    the filename Input must both carry the seeded default filename AND
    paint with a real, usable width -- not just have a non-empty ``.value``
    while rendering invisibly narrow."""
    dialog = _make_save_dialog(tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await pilot.pause()

        filename_input = _filename_input(dialog)
        assert filename_input.value == "mybench.json"

        bar = dialog.query_one(InputBar)
        # Painted geometry, not styles -- region.width is what the
        # compositor actually drew, which is what the "~5 cells" defect was
        # observed against.
        assert filename_input.region.width >= bar.region.width // 2, (
            f"filename Input painted at width={filename_input.region.width} "
            f"against an InputBar of width={bar.region.width} -- starved by "
            f"a higher-priority Select rule"
        )


@pytest.mark.asyncio
async def test_filesave_focuses_filename_input_on_mount(tmp_path):
    """Initial focus must land on the filename Input so a keyboard user can
    press Enter immediately -- not on the directory listing, where Enter
    would activate the highlighted row (usually '..')."""
    dialog = _make_save_dialog(tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()

        filename_input = _filename_input(dialog)
        assert dialog.focused is filename_input


@pytest.mark.asyncio
async def test_fileopen_still_focuses_directory_listing_on_mount(tmp_path):
    """Scope guard: FileOpen's initial-focus behaviour must be untouched --
    only FileSave gets the new focus-on-mount steering (task-1479)."""
    dialog = FileOpen(location=str(tmp_path), title="Open")
    app = _DialogHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(DirectoryNavigation)
        assert dialog.focused is dir_nav


@pytest.mark.asyncio
async def test_enter_immediately_after_mount_confirms_default_path(tmp_path):
    """The keyboard-only export path: mount FileSave, press Enter right
    away, and the dialog must dismiss with the seeded default filename --
    not navigate '..' (the focus bug) and not reject with "A file must be
    chosen" (the ambiguous-Input bug: even with focus correctly on the
    filename field, ``_confirm_file`` must read back that SAME Input, not
    one of the screen's other hidden Inputs)."""
    dialog = _make_save_dialog(tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()

        await pilot.press("enter")

        for _ in range(20):
            if app.result != "NOT_SET":
                break
            await pilot.pause()

        assert app.result == tmp_path / "mybench.json"


@pytest.mark.asyncio
async def test_directory_navigation_still_works_after_focus_change(tmp_path):
    """Scope guard: descending into a directory via a highlighted row +
    Enter must still work once the directory listing regains focus (e.g.
    after Ctrl+L path-bar navigation), proving this fix didn't disturb
    normal directory navigation for FileSave (task-1479)."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = _make_save_dialog(tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(DirectoryNavigation)
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        index = None
        for i in range(dir_nav.option_count):
            option = dir_nav.get_option_at_index(i)
            if option.location == subdir:
                index = i
                break
        assert index is not None, "subdir should appear in the directory list"

        dir_nav.highlighted = index
        dir_nav.focus()
        await pilot.pause()

        await pilot.press("enter")

        for _ in range(20):
            if dir_nav.location == subdir.resolve():
                break
            await pilot.pause()

        assert dir_nav.location == subdir.resolve(), (
            "a real Enter key press must still descend into a highlighted "
            "directory once the listing has focus"
        )
