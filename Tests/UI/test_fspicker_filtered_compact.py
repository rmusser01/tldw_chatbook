"""Filtered Open/Save controls remain usable under the production CSS cascade."""

from typing import ClassVar

import pytest
from textual.events import Paste
from textual.widgets import Button, Input

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_fspicker_compact_journeys import _loaded, _tab_to
from Tests.UI.test_library_ingest_entry_journeys import _painted
from Tests.UI.test_library_shell import _wait_for_condition
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave, Filters
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileFilter


class _Host(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.results = []

    async def on_mount(self):
        await self.push_screen(self.dialog, self.results.append)


def _picker(location, kind):
    filters = Filters(
        ("GGUF model files", lambda p: p.suffix == ".gguf"),
        ("All files", lambda p: True),
    )
    if kind == "save":
        return FileSave(location, filters=filters, default_file="new-model.gguf")
    return FileOpen(location, filters=filters, offer_select_folder=kind == "folder")


def _controls_visible(host, nav):
    dialog = host.dialog
    bounds = dialog.query_one(Dialog).content_region
    bar = dialog.query_one(InputBar)
    field = bar.query_one(Input)
    widgets = [field, bar.query_one(FileFilter), *bar.query(Button)]
    geometry = {str(w): str(w.region) for w in widgets}
    label = dialog.query_one("#file-name-label")
    assert "File name:" in _painted(host, label), {
        "label": str(label.region),
        **geometry,
    }
    # Enough of a path to read and edit, independently of the chosen layout.
    assert field.content_region.width >= 30, geometry
    for widget in widgets:
        assert bounds.contains_region(widget.region), geometry
    for button in bar.query(Button):
        assert str(button.label) in _painted(host, button), geometry
    assert "GGUF model files" in _painted(host, bar.query_one(FileFilter)), geometry
    assert nav.content_region.height >= 3, geometry
    assert "model.gguf" in _painted(host, nav), geometry


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("kind", ["open", "folder", "save"])
async def test_compact_filtered_picker_shows_and_submits_pasted_path(
    tmp_path, monkeypatch, theme, kind
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    tmp_path = tmp_path / "sources"
    tmp_path.mkdir()
    target = tmp_path / "model.gguf"
    target.write_text("private synthetic fixture")
    host = _Host(_picker(tmp_path, kind))
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        nav = await _loaded(host.dialog, pilot, tmp_path)
        _controls_visible(host, nav)
        field = host.dialog.query_one(InputBar).query_one(Input)
        await pilot.click(field)
        await pilot.press("ctrl+a")
        field.post_message(Paste(str(target)))
        await _wait_for_condition(
            pilot, lambda: field.value == str(target), message="pasted path"
        )
        if kind == "save":
            # A new private destination avoids the overwrite confirmation flow.
            await pilot.press("ctrl+a")
            await pilot.press(*"new-model.gguf")
            target = tmp_path / "new-model.gguf"
        await _tab_to(
            host.dialog, host, pilot, "#select", "Save" if kind == "save" else "Open"
        )
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: bool(host.results), message="path returned"
        )
        assert host.results == [target]
        assert (tmp_path / "model.gguf").read_text() == "private synthetic fixture"
        assert not (tmp_path / "new-model.gguf").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("kind", ["open", "save"])
async def test_filter_change_and_resize_keep_editor_selection_focus_and_cancel(
    tmp_path, monkeypatch, theme, kind
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    tmp_path = tmp_path / "sources"
    tmp_path.mkdir()
    (tmp_path / "model.gguf").touch()
    (tmp_path / "notes.txt").touch()
    host = _Host(_picker(tmp_path, kind))
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        dialog = host.dialog
        nav = await _loaded(dialog, pilot, tmp_path)
        _controls_visible(host, nav)
        field = dialog.query_one(InputBar).query_one(Input)
        select = dialog.query_one(FileFilter)
        await _tab_to(dialog, host, pilot, "FileFilter", "GGUF model files")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("end", "enter")
        await _wait_for_condition(
            pilot,
            lambda: select.value == 1 and nav.option_count == 3,
            message=lambda: (
                f"filter={select.value} expanded={select.expanded} count={nav.option_count} focus={dialog.focused} status={nav.listing_status}"
            ),
        )
        await pilot.press("end")
        selected_path = nav.get_option_at_index(nav.highlighted).location
        assert {
            nav.get_option_at_index(i).location.resolve()
            for i in range(nav.option_count)
        } == {tmp_path.parent, tmp_path / "model.gguf", tmp_path / "notes.txt"}
        assert selected_path.name in {"model.gguf", "notes.txt"}
        await pilot.click(field)
        await pilot.press("ctrl+a", *"unsaved path.gguf", "shift+left", "shift+left")
        selection = field.selection
        for size in [(170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert dialog.query_one(InputBar).query_one(Input) is field
            assert dialog.query_one(FileFilter) is select
            assert dialog.focused is field
            assert field.value == "unsaved path.gguf"
            assert field.selection == selection
            assert select.value == 1
            assert nav.get_option_at_index(nav.highlighted).location == selected_path
            assert field.content_region.width >= 30
            assert "unsaved path.gguf" in _painted(host, field)
        await _tab_to(dialog, host, pilot, "#cancel", "Cancel")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: bool(host.results), message="cancel returned"
        )
        assert host.results == [None]
        assert not (tmp_path / "unsaved path.gguf").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_filtered_listing_keyboard_selection_returns_file(
    tmp_path, monkeypatch, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    tmp_path = tmp_path / "sources"
    tmp_path.mkdir()
    target = tmp_path / "model.gguf"
    target.touch()
    (tmp_path / "hidden.txt").touch()
    host = _Host(_picker(tmp_path, "open"))
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        nav = await _loaded(host.dialog, pilot, tmp_path)
        _controls_visible(host, nav)
        assert nav.option_count == 2
        await pilot.press("end", "enter")
        field = host.dialog.query_one(InputBar).query_one(Input)
        await _wait_for_condition(
            pilot,
            lambda: field.value == "model.gguf" and host.dialog.focused is field,
            message="file selected",
        )
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: bool(host.results), message="file returned"
        )
        assert host.results == [target]
