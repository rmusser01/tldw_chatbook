"""Mount/smoke tests for the refactored enhanced file picker.

These tests push ``EnhancedFileOpen`` and ``EnhancedFileSave`` through a
Textual pilot and verify that:

* the dialogs mount without ``MountError``/``QueryError``,
* the base layout IDs the hooks rely on are present,
* the recent/bookmarks panels and search input can be toggled without crashing.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import Dialog
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    EnhancedFileSave,
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
