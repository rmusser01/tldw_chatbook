"""Mount/smoke tests for the refactored enhanced file picker.

These tests push ``EnhancedFileOpen`` and ``EnhancedFileSave`` through a
Textual pilot and verify that:

* the dialogs mount without ``MountError``/``QueryError``,
* the base layout IDs the hooks rely on are present,
* the recent/bookmarks panels and search input can be toggled without crashing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Input

from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    EnhancedFileSave,
    MultiSelectDirectoryEntry,
    SearchableDirectoryNavigation,
)


def _make_filters() -> Filters:
    return Filters(
        ("All Files", lambda _path: True),
    )


class _DialogHost(App[None]):
    """Minimal host that immediately pushes the dialog under test."""

    def __init__(self, dialog):
        super().__init__()
        self._dialog = dialog
        self._result: object = None

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        def _capture(result):
            self._result = result
            self.exit()

        await self.push_screen(self._dialog, callback=_capture)


@pytest.mark.asyncio
async def test_enhanced_file_open_mounts_and_widgets_exist():
    dialog = EnhancedFileOpen(
        location=".",
        title="Test Open",
        filters=_make_filters(),
        context="test_open",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Base layout IDs required by the hooks must exist.
        assert dialog.query_one("#path-breadcrumbs") is not None
        assert dialog.query_one("#recent-list") is not None
        assert dialog.query_one("#bookmarks-panel") is not None
        assert dialog.query_one("#bookmarks-list") is not None
        assert dialog.query_one("#search-input") is not None
        assert dialog.query_one(SearchableDirectoryNavigation) is not None
        assert dialog.query_one("#select") is not None
        assert dialog.query_one("#cancel") is not None

        # Panels and search should toggle without error.
        dialog.action_show_recent()
        await pilot.pause()
        dialog.action_toggle_bookmarks()
        await pilot.pause()
        dialog.action_focus_search()
        await pilot.pause()


@pytest.mark.asyncio
async def test_enhanced_file_save_mounts_and_widgets_exist():
    dialog = EnhancedFileSave(
        location=".",
        title="Test Save",
        filters=_make_filters(),
        default_filename="test.txt",
        context="test_save",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert dialog.query_one("#path-breadcrumbs") is not None
        assert dialog.query_one("#recent-list") is not None
        assert dialog.query_one("#bookmarks-panel") is not None
        assert dialog.query_one("#bookmarks-list") is not None
        assert dialog.query_one("#search-input") is not None
        assert dialog.query_one(SearchableDirectoryNavigation) is not None
        assert dialog.query_one("#select") is not None
        assert dialog.query_one("#cancel") is not None

        dialog.action_show_recent()
        await pilot.pause()
        dialog.action_toggle_bookmarks()
        await pilot.pause()


@pytest.mark.asyncio
async def test_search_filter_filters_directory_entries():
    """Search input updates ``SearchableDirectoryNavigation.search_filter``."""
    dialog = EnhancedFileOpen(
        location=".",
        title="Test Search",
        context="test_search",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        search_input = dialog.query_one("#search-input")

        # Typing a query that almost certainly does not match every entry.
        search_input.value = "__unlikely_name__"
        await pilot.pause()
        assert dir_nav.search_filter == "__unlikely_name__"

        # Clearing the search resets the filter.
        dialog.query_one("#clear-search").press()
        await pilot.pause()
        assert dir_nav.search_filter == ""


@pytest.mark.asyncio
async def test_selecting_file_populates_filename_input(tmp_path):
    """Selecting a file in the directory list fills the filename input.

    Regression guard for the filename-input ambiguity bug: the dialog
    contains ``#path-input``, ``#search-input``, and the filename input, so
    ``query_one(Input)`` is ambiguous. Selecting a file must populate the
    filename input, not one of the other inputs.
    """
    test_file = tmp_path / "sample_file.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Select File",
        context="test_select_file",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        # Wait for the directory navigation worker to populate the list.
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        # Find the option that corresponds to our test file.
        file_index = None
        for index in range(dir_nav.option_count):
            option = dir_nav.get_option_at_index(index)
            if option.location == test_file:
                file_index = index
                break

        assert file_index is not None, "test file should appear in the directory list"

        dir_nav.highlighted = file_index
        dir_nav.action_select()
        await pilot.pause()

        filename_input = dialog.query_one("#filename-input", Input)
        assert filename_input.value == "sample_file.txt"


@pytest.mark.asyncio
async def test_selecting_file_does_not_populate_hidden_path_input(tmp_path):
    """Selecting a file must not touch the hidden ``#path-input``.

    Regression guard for the MRO handler shadowing bug: the base
    ``BaseFileDialog._select_file`` runs ``query_one(Input)`` and would
    populate whichever input is found first. With the hidden ``#path-input``
    present, selecting a file must leave it empty.
    """
    test_file = tmp_path / "sample_file.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Hidden Path Input",
        context="test_hidden_path_input",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        file_index = None
        for index in range(dir_nav.option_count):
            option = dir_nav.get_option_at_index(index)
            if option.location == test_file:
                file_index = index
                break

        assert file_index is not None, "test file should appear in the directory list"

        dir_nav.highlighted = file_index
        dir_nav.action_select()
        await pilot.pause()

        path_input = dialog.query_one("#path-input", Input)
        assert path_input.value == ""



@pytest.mark.asyncio
async def test_open_dialog_confirms_selected_file(tmp_path):
    """Pushing EnhancedFileOpen and clicking Select returns the chosen file."""
    test_file = tmp_path / "confirm_me.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Confirm Open",
        context="test_confirm_open",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        file_index = None
        for index in range(dir_nav.option_count):
            option = dir_nav.get_option_at_index(index)
            if option.location == test_file:
                file_index = index
                break
        assert file_index is not None

        dir_nav.highlighted = file_index
        dir_nav.action_select()
        await pilot.pause()

        dialog.query_one("#select").press()
        await pilot.pause()

        # The host app exits once the screen is dismissed.
        result.append(app._result)

    assert result[0] == test_file


@pytest.mark.asyncio
async def test_save_dialog_confirms_new_file(tmp_path):
    """Pushing EnhancedFileSave and clicking Select returns the entered path."""
    dialog = EnhancedFileSave(
        location=str(tmp_path),
        title="Test Confirm Save",
        default_filename="new_file.txt",
        context="test_confirm_save",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        filename_input = dialog.query_one("#filename-input", Input)
        assert filename_input.value == "new_file.txt"

        dialog.query_one("#select").press()
        await pilot.pause()

        result.append(app._result)

    assert result[0] == tmp_path / "new_file.txt"


@pytest.mark.asyncio
async def test_cancel_dismisses_with_none(tmp_path):
    """Pushing EnhancedFileOpen and clicking Cancel dismisses with None."""
    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Cancel",
        context="test_cancel",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()
        dialog.query_one("#cancel").press()
        await pilot.pause()
        result.append(app._result)

    assert result[0] is None


@pytest.mark.asyncio
async def test_multi_select_hides_filename_input(tmp_path):
    """Multi-select mode does not render the single-filename input."""
    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Multi Select UI",
        multi_select=True,
        context="test_multi_select_ui",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        with pytest.raises(Exception):
            dialog.query_one("#filename-input", Input)


@pytest.mark.asyncio
async def test_multi_select_toggles_and_returns_list(tmp_path):
    """Multi-select mode returns a list of selected files."""
    file_a = tmp_path / "alpha.txt"
    file_b = tmp_path / "beta.txt"
    file_a.write_text("a")
    file_b.write_text("b")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Multi Select",
        multi_select=True,
        context="test_multi_select",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        # Find the two files in the option list.
        indices = {}
        for index in range(dir_nav.option_count):
            option = dir_nav.get_option_at_index(index)
            if option.location == file_a:
                indices["alpha"] = index
            elif option.location == file_b:
                indices["beta"] = index
        assert "alpha" in indices and "beta" in indices

        # Toggle both files via Space/action_toggle_selection.
        for key in ("alpha", "beta"):
            dir_nav.highlighted = indices[key]
            await pilot.pause()
            dialog.action_toggle_selection()
            await pilot.pause()
            option = dir_nav.get_option_at_index(indices[key])
            assert isinstance(option, MultiSelectDirectoryEntry)
            assert option.selected is True

        dialog.query_one("#select").press()
        await pilot.pause()
        result.append(app._result)

    assert isinstance(result[0], list)
    assert set(result[0]) == {file_a, file_b}


@pytest.mark.asyncio
async def test_type_ahead_jumps_to_file_prefix(tmp_path):
    """Typing a letter in the directory list jumps to the matching file."""
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    alpha.write_text("a")
    beta.write_text("b")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Type Ahead",
        context="test_type_ahead",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        dir_nav.focus()
        await pilot.pause()
        # Jump to the file whose name starts with "b".
        await pilot.press("b")
        await pilot.pause()

        assert dir_nav.highlighted is not None
        option = dir_nav.get_option_at_index(dir_nav.highlighted)
        assert option.location.name.startswith("b")


@pytest.mark.asyncio
async def test_directory_change_updates_breadcrumbs_and_bookmark_button(tmp_path):
    """Changing directory refreshes breadcrumbs and the bookmark button state."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Directory Change",
        context="test_dir_change",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        for _ in range(20):
            if dir_nav.option_count > 0:
                break
            await pilot.pause()

        # Sanity check: initial breadcrumbs show the starting directory.
        initial_breadcrumbs = [str(btn.label) for btn in dialog.query("#path-breadcrumbs Button")]
        assert tmp_path.name in initial_breadcrumbs or "🏠" in initial_breadcrumbs

        # Navigate into the subdirectory.
        dir_nav.location = subdir
        await pilot.pause()
        await pilot.pause()

        # Breadcrumbs should now include the subdirectory name.
        breadcrumbs = [str(btn.label) for btn in dialog.query("#path-breadcrumbs Button")]
        assert "subdir" in breadcrumbs, f"Expected 'subdir' in breadcrumbs, got {breadcrumbs}"

        # The bookmark button tooltip should reflect the new path (even if not bookmarked).
        add_bookmark = dialog.query_one("#add-bookmark")
        assert add_bookmark.tooltip is not None


async def _wait_for_options(dir_nav, pilot, attempts: int = 20) -> None:
    """Poll the threaded directory-navigation loader until it populates."""
    for _ in range(attempts):
        if dir_nav.option_count > 0:
            break
        await pilot.pause()


def _index_of(dir_nav, target: Path) -> Optional[int]:
    """Find the option-list index whose entry location matches ``target``."""
    for index in range(dir_nav.option_count):
        option = dir_nav.get_option_at_index(index)
        if option.location == target:
            return index
    return None


def _option_click_offset(dir_nav, index: int) -> tuple[int, int]:
    """Compute a widget-local ``(x, y)`` offset that lands inside the row
    rendered for the option at ``index``, for driving a real mouse click via
    ``pilot.click(dir_nav, offset=...)``.

    ``DirectoryNavigation`` sets ``border: blank`` (invisible, but still
    reserves a row/column), so a click offset can't be derived from the raw
    option index alone -- it must be adjusted by the widget's content-region
    inset (``content_region`` vs. ``region``) and any active scroll offset.
    """
    content_local = dir_nav.content_region.offset - dir_nav.region.offset
    line_number = dir_nav._index_to_line[index]
    y = content_local.y + line_number - dir_nav.scroll_offset.y
    x = content_local.x + 2
    return x, y


@pytest.mark.asyncio
async def test_single_select_on_dir_does_not_navigate(tmp_path):
    """Highlighting/selecting a directory (single-click semantics) must not
    auto-navigate into it (task-430 AC#2: select is select-only; opening a
    directory is a separate action)."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Single Select Dir",
        context="test_single_select_dir",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        subdir_index = _index_of(dir_nav, subdir)
        assert subdir_index is not None, "subdir should appear in the directory list"

        start_location = dir_nav.location
        dir_nav.highlighted = subdir_index
        dir_nav.action_select()
        await pilot.pause()

        assert dir_nav.location == start_location, "select must not descend into the directory"
        assert dir_nav.highlighted == subdir_index, "the directory should still be highlighted"


@pytest.mark.asyncio
async def test_open_highlighted_descends_dir(tmp_path):
    """``action_open_highlighted`` (Enter / double-click / Go) descends into
    a highlighted directory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Open Highlighted Dir",
        context="test_open_highlighted_dir",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        subdir_index = _index_of(dir_nav, subdir)
        assert subdir_index is not None, "subdir should appear in the directory list"

        dir_nav.highlighted = subdir_index
        dir_nav.action_open_highlighted()
        await pilot.pause()

        for _ in range(20):
            if dir_nav.location == subdir.resolve():
                break
            await pilot.pause()

        assert dir_nav.location == subdir.resolve(), "open must descend into the directory"


@pytest.mark.asyncio
async def test_double_click_opens_highlighted_dir(tmp_path):
    """A double-click (``Click`` event with ``chain >= 2``) opens the
    highlighted directory, exercising ``SearchableDirectoryNavigation.on_click``
    directly (task-430 AC#2)."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Double Click Dir",
        context="test_double_click_dir",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        subdir_index = _index_of(dir_nav, subdir)
        assert subdir_index is not None, "subdir should appear in the directory list"

        # A single click (chain=1) only highlights/selects -- must not navigate.
        dir_nav.highlighted = subdir_index
        dir_nav.on_click(
            events.Click(
                dir_nav, 0, 0, 0, 0, button=1, shift=False, meta=False, ctrl=False, chain=1
            )
        )
        await pilot.pause()
        assert dir_nav.location == tmp_path.resolve(), "single click must not navigate"

        # The second click of the chain (chain=2) opens the directory.
        dir_nav.on_click(
            events.Click(
                dir_nav, 0, 0, 0, 0, button=1, shift=False, meta=False, ctrl=False, chain=2
            )
        )
        await pilot.pause()

        for _ in range(20):
            if dir_nav.location == subdir.resolve():
                break
            await pilot.pause()

        assert dir_nav.location == subdir.resolve(), "double-click must descend into the directory"


@pytest.mark.asyncio
async def test_open_highlighted_returns_file(tmp_path):
    """``action_open_highlighted`` (Enter / double-click / Go) on a
    highlighted file confirms and returns it, dismissing the dialog."""
    test_file = tmp_path / "open_me.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Open Highlighted File",
        context="test_open_highlighted_file",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        file_index = _index_of(dir_nav, test_file)
        assert file_index is not None, "test file should appear in the directory list"

        dir_nav.highlighted = file_index
        dir_nav.action_open_highlighted()
        await pilot.pause()

        result.append(app._result)

    assert result[0] == test_file


@pytest.mark.asyncio
async def test_single_select_on_file_fills_filename(tmp_path):
    """Single-select (``action_select``) on a highlighted file fills the
    filename input but does not confirm/dismiss the dialog -- confirming
    still requires Enter/double-click/Go (``action_open_highlighted``) or
    the Select button."""
    test_file = tmp_path / "pick_me.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Single Select File",
        context="test_single_select_file",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        file_index = _index_of(dir_nav, test_file)
        assert file_index is not None, "test file should appear in the directory list"

        dir_nav.highlighted = file_index
        dir_nav.action_select()
        await pilot.pause()

        filename_input = dialog.query_one("#filename-input", Input)
        assert filename_input.value == "pick_me.txt"
        assert app._result is None, "select-only must not dismiss the dialog"


# --- Real-driver regression coverage (task-430) -----------------------------
#
# The tests above exercise ``action_open_highlighted()``/``action_select()``/
# ``on_click(events.Click(...))`` directly. That's valuable for pinning down
# exact semantics, but it never proves the real UI routes (an actual Enter
# key press, an actual Go/Select button press with no filename typed, and an
# actual double-click delivered through Textual's render/hit-test pipeline)
# are wired to those methods. The tests below drive the dialog the way a
# user actually would, so a future refactor that silently detaches a
# binding/handler will be caught here even if the underlying action methods
# still work in isolation.


@pytest.mark.asyncio
async def test_real_enter_key_descends_highlighted_dir(tmp_path):
    """A real ``pilot.press("enter")`` -- not a direct
    ``action_open_highlighted()`` call -- descends into a highlighted
    directory, exercising ``SearchableDirectoryNavigation``'s Enter
    binding end-to-end (task-430 AC#2)."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Real Enter Dir",
        context="test_real_enter_dir",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        subdir_index = _index_of(dir_nav, subdir)
        assert subdir_index is not None, "subdir should appear in the directory list"

        dir_nav.highlighted = subdir_index
        dir_nav.focus()
        await pilot.pause()

        await pilot.press("enter")

        for _ in range(20):
            if dir_nav.location == subdir.resolve():
                break
            await pilot.pause()

        assert dir_nav.location == subdir.resolve(), (
            "a real Enter key press must descend into the highlighted directory"
        )


@pytest.mark.asyncio
async def test_real_enter_key_opens_highlighted_file(tmp_path):
    """A real ``pilot.press("enter")`` on a highlighted file confirms and
    returns it, dismissing the dialog end-to-end (task-430 AC#2)."""
    test_file = tmp_path / "enter_open_me.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Real Enter File",
        context="test_real_enter_file",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        file_index = _index_of(dir_nav, test_file)
        assert file_index is not None, "test file should appear in the directory list"

        dir_nav.highlighted = file_index
        dir_nav.focus()
        await pilot.pause()

        await pilot.press("enter")

        for _ in range(20):
            if app._result is not None:
                break
            await pilot.pause()

        result.append(app._result)

    assert result[0] == test_file


@pytest.mark.asyncio
async def test_go_button_descends_highlighted_dir_without_filename(tmp_path):
    """Pressing the Go/Select button (a real ``Button.Pressed`` message, via
    ``Button.press()``) with no filename typed and a directory highlighted
    descends into it instead of erroring or dismissing -- exercising
    ``_confirm_single()``'s highlighted-directory branch through the actual
    button handler (task-430 AC#2)."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Go Button Dir",
        context="test_go_button_dir",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        subdir_index = _index_of(dir_nav, subdir)
        assert subdir_index is not None, "subdir should appear in the directory list"

        dir_nav.highlighted = subdir_index
        await pilot.pause()

        filename_input = dialog.query_one("#filename-input", Input)
        assert filename_input.value == "", "no filename should be typed for this scenario"

        dialog.query_one("#select").press()

        for _ in range(20):
            if dir_nav.location == subdir.resolve():
                break
            await pilot.pause()

        assert dir_nav.location == subdir.resolve(), (
            "Go/Select with no filename typed must descend into the highlighted directory"
        )
        assert app._result is None, "descending must not dismiss the dialog"


@pytest.mark.asyncio
async def test_real_double_click_opens_highlighted_dir(tmp_path):
    """A double-click delivered through Textual's pilot (``times=2``) opens
    the highlighted directory, exercising the real
    ``MouseDown``/``MouseUp``/``Click`` dispatch and the compositor's
    ``event.style.meta["option"]`` hit-test -- not a hand-built
    ``events.Click(chain=2)`` object passed straight to ``on_click()``
    (task-430 AC#2).

    ``pilot.click(..., times=N)`` is Textual's own supported mechanism for
    emulating an N-times click (see ``Pilot.click``'s docstring: "times: ...
    2 will double-click"); it drives ``chain=1`` then ``chain=2`` through the
    full event pipeline, which is why this is a genuine double-click and not
    an ad-hoc probe. See ``test_double_click_opens_highlighted_dir`` above
    for a lower-level, single-event-object check of the same
    ``on_click`` chain-based branching.
    """
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Real Double Click Dir",
        context="test_real_double_click_dir",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        subdir_index = _index_of(dir_nav, subdir)
        assert subdir_index is not None, "subdir should appear in the directory list"

        dir_nav.highlighted = subdir_index
        await pilot.pause()

        offset = _option_click_offset(dir_nav, subdir_index)
        start_location = dir_nav.location

        # A real single click (chain=1) only highlights/selects -- it must
        # not navigate.
        await pilot.click(dir_nav, offset=offset, times=1)
        await pilot.pause()
        assert dir_nav.location == start_location, "a single click must not navigate"

        # A real double-click (chain=1 then chain=2, delivered through the
        # actual render/hit-test pipeline) opens the directory.
        await pilot.click(dir_nav, offset=offset, times=2)

        for _ in range(20):
            if dir_nav.location == subdir.resolve():
                break
            await pilot.pause()

        assert dir_nav.location == subdir.resolve(), (
            "a real double-click must descend into the highlighted directory"
        )


@pytest.mark.asyncio
async def test_legacy_list_filters_are_normalized(tmp_path):
    """A list of glob strings can be passed as ``filters`` without crashing."""
    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Legacy Filters",
        filters=["*.zip"],
        context="test_legacy_filters",
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        # The dialog should mount and expose the filter Select.
        select = dialog.query_one("#file-filter")
        assert select is not None


@pytest.mark.asyncio
async def test_open_must_exist_rejects_missing_file(tmp_path):
    """EnhancedFileOpen with must_exist=True refuses a non-existent filename."""
    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Must Exist",
        must_exist=True,
        context="test_must_exist",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        filename_input = dialog.query_one("#filename-input", Input)
        filename_input.value = "does_not_exist.txt"
        await pilot.pause()

        dialog.query_one("#select").press()
        await pilot.pause()

        result.append(app._result)

    assert result[0] is None


@pytest.mark.asyncio
async def test_path_bar_opens_a_file(tmp_path):
    """Submitting the full path to an *existing file* in the Ctrl+L path bar
    confirms and returns that file, instead of silently navigating to its
    parent directory and dropping the filename (task-430 AC#3)."""
    test_file = tmp_path / "typed_path.txt"
    test_file.write_text("hello")

    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Path Bar Opens File",
        context="test_path_bar_opens_file",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        dialog.action_focus_path_input()
        await pilot.pause()

        path_input = dialog.query_one("#path-input", Input)
        path_input.value = str(test_file)
        await pilot.pause()

        # A real Button.Pressed message, driven through the actual Go button.
        dialog.query_one("#go-to-path").press()
        await pilot.pause()

        result.append(app._result)

    assert result[0] == test_file


@pytest.mark.asyncio
async def test_escape_closes_recent_overlay_first(tmp_path):
    """Esc closes the topmost open overlay (Recent) before dismissing the
    picker; a second Esc then dismisses the picker (task-430 AC#4)."""
    dialog = EnhancedFileOpen(
        location=str(tmp_path),
        title="Test Escape Layered Overlay",
        context="test_escape_layered_overlay",
    )
    app = _DialogHost(dialog)
    result = []

    async with app.run_test() as pilot:
        await pilot.pause()

        dialog.action_toggle_recent()
        await pilot.pause()
        assert dialog.show_recent is True

        # A real key press, not a direct action call.
        await pilot.press("escape")
        await pilot.pause()
        assert dialog.show_recent is False, "first Esc must close the Recent overlay"
        assert dialog.is_current, "first Esc must not dismiss the picker"

        await pilot.press("escape")
        await pilot.pause()

        result.append(app._result)

    assert result[0] is None, "second Esc must dismiss the picker"


def test_file_list_highlight_is_visible():
    """The file list's selected-row highlight must not blend into $surface.

    ``tldw_cli_modular.tcss`` carries a generic
    ``OptionList > .option-list--option-highlighted`` rule that paints the
    highlighted row with ``$surface`` (close to the dialog background). That
    app-level rule wins over ``SearchableDirectoryNavigation``'s own
    ``EnhancedFileDialog SearchableDirectoryNavigation > .option-list--...``
    DEFAULT_CSS via Textual's origin-priority cascade, so the cursor row is
    effectively invisible in the running app (task-430 AC#1). Fix: an
    id-scoped override for ``#file-list-pane`` using the sanctioned
    ``$ds-focus-bg``/``$ds-focus-fg`` non-obscuring focus tokens.

    This asserts against the CSS *source* rather than a rendered component
    style. ``_DialogHost`` above is a bare ``App`` with no ``CSS_PATH`` set,
    so ``tldw_cli_modular.tcss`` (where both the offending generic rule and
    the fix live) is never loaded in-test; a pilot-based
    ``get_component_styles`` probe against it only resolves
    ``SearchableDirectoryNavigation``'s own DEFAULT_CSS ($primary 30%/50%),
    never the bundle rule this fix targets, so it can't exercise the bug or
    the fix. The source assertion is the only way to pin this down in this
    harness.
    """
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )
    text = bundle_path.read_text(encoding="utf-8")

    match = re.search(
        r"#file-list-pane\s+\.option-list--option-highlighted\s*\{([^}]*)\}",
        text,
    )
    assert match is not None, (
        "Expected an id-scoped '#file-list-pane .option-list--option-highlighted' "
        "rule in tldw_cli_modular.tcss that beats the generic "
        "'OptionList > .option-list--option-highlighted' rule for the file picker's "
        "list pane."
    )

    block = match.group(1)
    assert "$ds-focus-bg" in block, "Highlighted row must use the focus-bg token"
    assert "$ds-focus-fg" in block, "Highlighted row must use the focus-fg token"
    assert "$surface" not in block, (
        "Highlighted row must not fall back to the near-invisible $surface color"
    )
