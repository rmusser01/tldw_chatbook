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

import pytest
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
