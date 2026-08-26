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
from textual.widgets._select import SelectOverlay

import tldw_chatbook
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    FileSave,
    Filters,
    SelectDirectory,
)
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileFilter
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
        self.results: list[object] = []

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        def _capture(result):
            self.result = result
            self.results.append(result)

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


def _make_picker(dialog_type, tmp_path: Path):
    if dialog_type is FileOpen:
        return FileOpen(
            location=tmp_path,
            filters=_export_filters(),
            must_exist=False,
            default_file="chosen.json",
        )
    if dialog_type is FileSave:
        return FileSave(
            location=tmp_path,
            filters=_export_filters(),
            default_file="chosen.json",
        )
    return SelectDirectory(location=tmp_path)


async def _wait_for_picker_result(app: _DialogHost, pilot) -> None:
    for _ in range(20):
        if app.results:
            return
        await pilot.pause()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_type", [FileOpen, FileSave, SelectDirectory])
@pytest.mark.parametrize("source", ["escape", "visible", "backdrop"])
async def test_file_picker_cancel_sources_return_none_once(
    tmp_path: Path,
    dialog_type,
    source: str,
) -> None:
    """Every concrete picker exposes one bounded content root and one-shot cancel."""
    dialog = _make_picker(dialog_type, tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        content = dialog.query_one("#file-system-picker-dialog", Dialog)
        assert content is dialog.query_one(Dialog)
        assert content.region.width < dialog.region.width
        assert content.region.height < dialog.region.height

        if source == "escape":
            await pilot.press("escape")
        elif source == "visible":
            await pilot.click("#cancel")
        else:
            await pilot.click(offset=(0, 0))
        await _wait_for_picker_result(app, pilot)

    assert app.results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_type", [FileOpen, FileSave, SelectDirectory])
async def test_file_picker_success_results_remain_paths(
    tmp_path: Path,
    dialog_type,
) -> None:
    dialog = _make_picker(dialog_type, tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#select")
        await _wait_for_picker_result(app, pilot)

    expected = (
        [tmp_path]
        if dialog_type is SelectDirectory
        else [tmp_path / "chosen.json"]
    )
    assert app.results == expected
    assert isinstance(app.results[0], Path)


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_type", [FileOpen, FileSave, SelectDirectory])
async def test_file_picker_non_primary_and_inside_clicks_stay_open(
    tmp_path: Path,
    dialog_type,
) -> None:
    dialog = _make_picker(dialog_type, tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(offset=(0, 0), button=3)
        await pilot.click("#file-system-picker-dialog")
        await pilot.click("#path-breadcrumbs")
        await pilot.click(DirectoryNavigation, offset=(2, 2))
        await pilot.pause()

        assert app.screen is dialog
        assert app.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_type", [FileOpen, FileSave, SelectDirectory])
async def test_file_picker_escape_peels_path_search_recent_in_order(
    tmp_path: Path,
    dialog_type,
) -> None:
    dialog = _make_picker(dialog_type, tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.press("ctrl+f")
        await pilot.press("ctrl+r")
        await pilot.pause()

        path_container = dialog.query_one("#path-input-container")
        assert not path_container.has_class("hidden")
        assert dialog.search_active
        assert dialog.show_recent

        await pilot.click("#path-input")
        await pilot.click("#search-input")
        await pilot.click("#recent-list")
        await pilot.click(DirectoryNavigation, offset=(2, 2))
        await pilot.pause()
        assert app.screen is dialog
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()
        assert path_container.has_class("hidden")
        assert dialog.search_active
        assert dialog.show_recent
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()
        assert not dialog.search_active
        assert dialog.show_recent
        assert app.results == []

        await pilot.press("escape")
        await pilot.pause()
        assert not dialog.show_recent
        assert app.results == []

        await pilot.press("escape")
        await _wait_for_picker_result(app, pilot)

    assert app.results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_type", [FileOpen, FileSave, SelectDirectory])
@pytest.mark.parametrize("source", ["visible", "backdrop"])
async def test_file_picker_direct_cancel_is_terminal_with_transients_open(
    tmp_path: Path,
    dialog_type,
    source: str,
) -> None:
    dialog = _make_picker(dialog_type, tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.press("ctrl+f")
        await pilot.press("ctrl+r")
        await pilot.pause()

        if source == "visible":
            await pilot.click("#cancel")
        else:
            await pilot.click(offset=(0, 0))
        await _wait_for_picker_result(app, pilot)

    assert app.results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("dialog_type", [FileOpen, FileSave])
async def test_fspicker_expanded_select_overlay_consumes_first_escape(
    tmp_path: Path,
    dialog_type,
) -> None:
    dialog = _make_picker(dialog_type, tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        file_filter = dialog.query_one(FileFilter)
        file_filter.focus()
        await pilot.press("enter")
        await pilot.pause()
        overlay = file_filter.query_one(SelectOverlay)
        assert file_filter.expanded

        await pilot.click(overlay, offset=(1, 1))
        await pilot.pause()
        assert app.screen is dialog
        assert app.results == []

        file_filter.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert file_filter.expanded
        await pilot.press("escape")
        await pilot.pause()

        assert not file_filter.expanded
        assert app.screen is dialog
        assert app.results == []


@pytest.mark.asyncio
async def test_fspicker_select_directory_changed_dispatches_each_base_effect_once(
    tmp_path: Path,
) -> None:
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    dialog = SelectDirectory(location=tmp_path)
    app = _DialogHost(dialog)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        counts = {"error": 0, "breadcrumb": 0, "recent": 0, "path": 0}
        real_set_error = dialog._set_error
        real_update_breadcrumbs = dialog._update_breadcrumbs
        current_path = dialog.query_one("#current_path_display")
        real_path_update = current_path.update

        def count_error(message: str = "") -> None:
            counts["error"] += 1
            real_set_error(message)

        def count_breadcrumb(path: Path) -> None:
            counts["breadcrumb"] += 1
            real_update_breadcrumbs(path)

        def count_recent(path: Path, file_type: str) -> None:
            del path, file_type
            counts["recent"] += 1

        def count_path(value: object = "") -> None:
            counts["path"] += 1
            real_path_update(value)

        dialog._set_error = count_error
        dialog._update_breadcrumbs = count_breadcrumb
        dialog._add_to_recent = count_recent
        current_path.update = count_path

        dialog.query_one(DirectoryNavigation).location = subdir
        for _ in range(20):
            if counts["path"]:
                break
            await pilot.pause()

        assert counts == {
            "error": 1,
            "breadcrumb": 1,
            "recent": 1,
            "path": 1,
        }
        assert dialog.query_one("#path_input", Input).value == str(subdir.resolve())
