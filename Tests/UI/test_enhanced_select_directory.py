"""Tests for the enhanced-family directory selection dialog (TASK-16477).

``EnhancedSelectDirectory`` mirrors the vendored ``SelectDirectory``
contract (directory-only listing; Select returns the directory currently
viewed) on the ``EnhancedFileDialog`` chrome, so the Roleplay screen's
export-marked-JSON flow matches the pickers used everywhere else.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Widgets import enhanced_file_picker as efp_module
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedSelectDirectory,
    SearchableDirectoryNavigation,
)


class _DialogHost(App[None]):
    """Minimal host that immediately pushes the dialog under test."""

    def __init__(self, dialog):
        super().__init__()
        self._dialog = dialog
        self.result: object = None
        self.result_seen = False

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self._dialog, callback=self._capture)

    def _capture(self, result) -> None:
        self.result = result
        self.result_seen = True


async def _wait_for_options(nav: SearchableDirectoryNavigation, pilot) -> None:
    for _ in range(20):
        if nav.option_count > 0:
            break
        await pilot.pause()


def _entry_paths(nav: SearchableDirectoryNavigation) -> list[Path]:
    return [
        nav.get_option_at_index(index).location for index in range(nav.option_count)
    ]


@pytest.mark.asyncio
async def test_mounts_listing_directories_only(tmp_path):
    """The listing shows directories (and ``..``) but no file entries."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "readme.txt").write_text("x")

    dialog = EnhancedSelectDirectory(
        location=tmp_path, title="Export 2 items as JSON", context="t_dirs_only"
    )
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(nav, pilot)

        paths = _entry_paths(nav)
        assert any(p.name == "subdir" for p in paths)
        assert not any(p.name == "readme.txt" for p in paths)


@pytest.mark.asyncio
async def test_mount_populates_breadcrumbs_and_dir_path_input(tmp_path):
    """The enhanced chrome is live on open: breadcrumbs populated, path input synced."""
    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_mount_chrome")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        nav = dialog.query_one(SearchableDirectoryNavigation)

        crumbs = dialog.query_one("#path-breadcrumbs")
        assert len(list(crumbs.walk_children(with_self=False))) > 0

        path_input = dialog.query_one("#dir-path-input", Input)
        assert path_input.value == str(nav.location)


@pytest.mark.asyncio
async def test_select_button_returns_viewed_directory(tmp_path):
    """Select dismisses with the directory being viewed, like the vendored dialog."""
    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_select_dir")
    app = _DialogHost(dialog)

    with (
        patch.object(efp_module, "save_settings_to_cli_config"),
        patch.object(efp_module, "save_setting_to_cli_config"),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            dialog.query_one("#select", Button).press()
            await pilot.pause()

    assert app.result == tmp_path.resolve()


@pytest.mark.asyncio
async def test_cancel_dismisses_with_none(tmp_path):
    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_cancel_dir")
    app = _DialogHost(dialog)

    with (
        patch.object(efp_module, "save_settings_to_cli_config"),
        patch.object(efp_module, "save_setting_to_cli_config"),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            dialog.query_one("#cancel", Button).press()
            await pilot.pause()

    assert app.result_seen
    assert app.result is None


@pytest.mark.asyncio
async def test_dir_path_input_navigates_to_typed_directory(tmp_path):
    (tmp_path / "nested").mkdir()

    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_path_nav")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(nav, pilot)

        path_input = dialog.query_one("#dir-path-input", Input)
        path_input.value = str(tmp_path / "nested")
        path_input.focus()
        await pilot.press("enter")
        for _ in range(20):
            if nav.location == (tmp_path / "nested").resolve():
                break
            await pilot.pause()

        assert nav.location == (tmp_path / "nested").resolve()
        # The input keeps tracking the viewed directory after navigation.
        assert path_input.value == str(nav.location)


@pytest.mark.asyncio
async def test_dir_path_input_bad_path_shows_error_and_stays_open(tmp_path):
    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_bad_path")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        path_input = dialog.query_one("#dir-path-input", Input)
        path_input.value = str(tmp_path / "definitely-not-here")
        path_input.focus()
        await pilot.press("enter")
        await pilot.pause()

        error_line = dialog.query_one("#error-line", Static)
        assert "Path not found" in str(error_line.renderable)
        assert not app.result_seen


@pytest.mark.asyncio
async def test_dir_path_input_nul_byte_is_rejected_not_raised(tmp_path):
    """A NUL byte must set an error line, not raise ValueError (Qodo TASK-16478)."""
    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_nul_path")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        path_input = dialog.query_one("#dir-path-input", Input)
        path_input.value = str(tmp_path / "bad\x00name")
        path_input.focus()
        await pilot.press("enter")
        await pilot.pause()

        error_line = dialog.query_one("#error-line", Static)
        assert "null characters" in str(error_line.renderable)
        assert not app.result_seen


@pytest.mark.asyncio
async def test_dir_path_input_existing_file_reports_not_a_directory(tmp_path):
    """An existing file path is a different error than a missing one (Qodo)."""
    (tmp_path / "a_file.txt").write_text("x")

    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_file_path")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        path_input = dialog.query_one("#dir-path-input", Input)
        path_input.value = str(tmp_path / "a_file.txt")
        path_input.focus()
        await pilot.press("enter")
        await pilot.pause()

        error_line = dialog.query_one("#error-line", Static)
        rendered = str(error_line.renderable)
        assert "Not a directory" in rendered
        assert "Path not found" not in rendered
        assert not app.result_seen


@pytest.mark.asyncio
async def test_dir_path_input_accepts_relative_paths(tmp_path):
    (tmp_path / "relative-target").mkdir()

    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_relative")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        nav = dialog.query_one(SearchableDirectoryNavigation)

        path_input = dialog.query_one("#dir-path-input", Input)
        path_input.value = "relative-target"
        path_input.focus()
        await pilot.press("enter")
        for _ in range(20):
            if nav.location == (tmp_path / "relative-target").resolve():
                break
            await pilot.pause()

        assert nav.location == (tmp_path / "relative-target").resolve()


@pytest.mark.asyncio
async def test_shortcut_hints_name_the_folder_action(tmp_path):
    """Directory-mode hints must not advertise the file-flow Enter-confirm."""
    dialog = EnhancedSelectDirectory(location=tmp_path, context="t_hints")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        hints = dialog.query_one("#shortcut-hints", Static).renderable
        text = str(hints)
        assert "use this folder" in text
        assert "Enter Open" in text


def test_construction_has_no_filters_or_multi_select():
    """Directory picking never grows a file filter or multi-select set."""
    dialog = EnhancedSelectDirectory(context="t_ctor")
    assert dialog.filters is None
    assert dialog.multi_select is False


def test_filters_argument_is_not_accepted():
    """The vendored ``SelectDirectory`` shape stays: no ``filters`` kwarg."""
    with pytest.raises(TypeError):
        EnhancedSelectDirectory(filters=Filters(("All", lambda p: True)))  # type: ignore[call-arg]
