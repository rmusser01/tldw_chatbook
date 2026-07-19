"""Integration tests for the flattened evaluation file picker wrappers.

These tests push each eval-specific dialog through a Textual pilot and verify
open/cancel flows and callback delivery.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.enhanced_file_picker import SearchableDirectoryNavigation
from tldw_chatbook.Widgets.file_picker_dialog import (
    DatasetFilePickerDialog,
    EvalFilePickerDialog,
    ExportFilePickerDialog,
    QuickPickerWidget,
    TaskFilePickerDialog,
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


class _WidgetHost(App[None]):
    """Host for inline QuickPickerWidget tests."""

    def __init__(self, widget):
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


async def _wait_for_options(dir_nav, pilot, attempts=20):
    for _ in range(attempts):
        if dir_nav.option_count > 0:
            break
        await pilot.pause()


def _index_of(dir_nav, target):
    for index in range(dir_nav.option_count):
        option = dir_nav.get_option_at_index(index)
        if option.location == target:
            return index
    return None


@pytest.mark.asyncio
async def test_eval_file_picker_selects_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_file = tmp_path / "eval.yaml"
    test_file.write_text("test: true")

    picked = []
    dialog = EvalFilePickerDialog(callback=lambda p: picked.append(p))
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        index = _index_of(dir_nav, test_file)
        assert index is not None, "eval file should appear in the filtered list"

        dir_nav.highlighted = index
        dir_nav.action_select()
        await pilot.pause()
        dialog.query_one("#select").press()
        await pilot.pause()

    assert picked and picked[0] == str(test_file)


@pytest.mark.asyncio
async def test_eval_file_picker_cancel_returns_none(tmp_path):
    picked = ["not-called"]
    dialog = EvalFilePickerDialog(callback=lambda p: picked.append(p))
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dialog.query_one("#cancel").press()
        await pilot.pause()

    assert picked[-1] is None


@pytest.mark.asyncio
async def test_task_file_picker_selects_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_file = tmp_path / "task.yaml"
    test_file.write_text("task: test")

    picked = []
    dialog = TaskFilePickerDialog(callback=lambda p: picked.append(p))
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        index = _index_of(dir_nav, test_file)
        assert index is not None, "YAML task file should appear"

        dir_nav.highlighted = index
        dir_nav.action_select()
        await pilot.pause()
        dialog.query_one("#select").press()
        await pilot.pause()

    assert picked and picked[0] == str(test_file)


@pytest.mark.asyncio
async def test_dataset_file_picker_selects_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_file = tmp_path / "data.csv"
    test_file.write_text("a,b\n1,2\n")

    picked = []
    dialog = DatasetFilePickerDialog(callback=lambda p: picked.append(p))
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        index = _index_of(dir_nav, test_file)
        assert index is not None, "CSV dataset file should appear"

        dir_nav.highlighted = index
        dir_nav.action_select()
        await pilot.pause()
        dialog.query_one("#select").press()
        await pilot.pause()

    assert picked and picked[0] == str(test_file)


@pytest.mark.asyncio
async def test_export_file_picker_returns_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    picked = []
    dialog = ExportFilePickerDialog(callback=lambda p: picked.append(p))
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Input

        filename_input = dialog.query_one("#filename-input", Input)
        filename_input.value = "results.json"
        await pilot.pause()
        dialog.query_one("#select").press()
        await pilot.pause()

    assert picked and picked[0] == str(tmp_path / "results.json")


@pytest.mark.asyncio
async def test_quick_picker_widget_browse_selects_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_file = tmp_path / "task.yaml"
    test_file.write_text("task: test")

    picked = []
    widget = QuickPickerWidget(
        file_types="task files", callback=lambda p: picked.append(p), context="test_quick"
    )
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()
        widget.query_one("#browse-button").press()
        await pilot.pause()

        dialog = app.screen
        dir_nav = dialog.query_one(SearchableDirectoryNavigation)
        await _wait_for_options(dir_nav, pilot)

        index = _index_of(dir_nav, test_file)
        assert index is not None, "task file should appear for QuickPickerWidget"

        dir_nav.highlighted = index
        dir_nav.action_select()
        await pilot.pause()
        dialog.query_one("#select").press()
        await pilot.pause()

        assert picked and picked[0] == str(test_file)
        display = widget.query_one("#selected-file-display")
        assert "task.yaml" in str(display.renderable)


@pytest.mark.asyncio
async def test_quick_picker_widget_clear_selection(tmp_path):
    widget = QuickPickerWidget(file_types="evaluation files", context="test_quick_clear")
    widget.selected_file = str(tmp_path / "old.txt")
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()
        widget.clear_selection()
        await pilot.pause()

        assert widget.get_selected_file() is None
        display = widget.query_one("#selected-file-display")
        assert "No file selected" in str(display.renderable)
