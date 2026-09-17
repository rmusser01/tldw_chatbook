"""Folder listing, actions and resize continuity under production app CSS."""

from typing import ClassVar

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_library_ingest_entry_journeys import _painted
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import _wait_for_condition
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import Dialog, InputBar
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation


class _PickerHost(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, location):
        super().__init__()
        self.dialog = SelectDirectory(location, title="Choose model folder")
        self.results = []

    async def on_mount(self):
        await self.push_screen(self.dialog, self.results.append)


async def _loaded(dialog, pilot, location):
    nav = dialog.query_one(DirectoryNavigation)
    await _wait_for_condition(
        pilot,
        lambda: nav.location == location and nav.listing_status.startswith("Loaded"),
        message="Directory listing did not finish",
    )
    return nav


async def _tab_to(dialog, host, pilot, selector, label):
    for _ in range(24):
        if dialog.focused is dialog.query_one(selector):
            break
        await pilot.press("tab")
    await _focus(dialog, host, pilot, selector, label)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_folder_rows_paint_and_keyboard_navigation_selects_child(
    tmp_path, monkeypatch, size, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    child = tmp_path / "alpha folder"
    child.mkdir()
    (tmp_path / "omega folder").mkdir()
    host = _PickerHost(tmp_path)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        dialog = host.dialog
        nav = await _loaded(dialog, pilot, tmp_path)
        # TASK-32606 opens folder-returning pickers on their path field.
        assert dialog.focused is dialog.query_one("#path_input", Input)
        await _tab_to(dialog, host, pilot, DirectoryNavigation, "alpha folder")
        assert dialog.focused is nav
        # The source of the initial regression: loaded options with no list paint.
        geometry = {
            type(widget).__name__ + (f"#{widget.id}" if widget.id else ""): {
                "region": str(widget.region),
                "content": str(widget.content_region),
            }
            for widget in [
                dialog.query_one(Dialog),
                *dialog.query_one(Dialog).children,
                nav,
            ]
        }
        assert "alpha folder" in _painted(host, nav), geometry
        assert "omega folder" in _painted(host, nav), geometry
        assert nav.content_region.height >= 3, geometry
        await pilot.press("home", "down")
        assert nav.get_option_at_index(nav.highlighted).location == child
        assert "alpha folder" in _painted(host, nav)
        await pilot.press("enter")
        await _loaded(dialog, pilot, child)
        assert dialog.query_one("#path_input", Input).value == str(child)
        await pilot.press("home", "enter")
        await _loaded(dialog, pilot, tmp_path)
        await pilot.press("home", "down", "enter")
        await _loaded(dialog, pilot, child)
        await _tab_to(dialog, host, pilot, "#select", "Select")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: bool(host.results), message="Select did not return folder"
        )
        assert host.results == [child]
        assert list(child.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_compact_error_and_resize_keep_path_actions_and_listing(
    tmp_path, monkeypatch, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    (tmp_path / "alpha folder").mkdir()
    host = _PickerHost(tmp_path)
    host.theme = theme
    async with host.run_test(size=(170, 48)) as pilot:
        dialog = host.dialog
        nav = await _loaded(dialog, pilot, tmp_path)
        await _tab_to(dialog, host, pilot, "#path_input", tmp_path.name)
        field = dialog.query_one("#path_input", Input)
        field.value = "missing-folder"
        field.cursor_position = 4
        await _tab_to(dialog, host, pilot, "#select", "Select")
        await pilot.press("enter")
        error = dialog.query_one("#picker-error-line", Static)
        await _wait_for_condition(
            pilot, lambda: error.display, message="Invalid folder did not show error"
        )
        selection = field.selection
        highlighted = nav.get_option_at_index(nav.highlighted).location
        for size in [(80, 24), (170, 48), (80, 24)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert host.screen is dialog and host.results == []
            assert dialog.query_one("#path_input") is field
            assert field.value == "missing-folder"
            assert field.selection == selection
            assert nav.get_option_at_index(nav.highlighted).location == highlighted
            # Validation returns focus to the path; resize must keep it there.
            await _focus(dialog, host, pilot, "#path_input", "missing-folder")
            assert "Path not found: missing-folder" in _painted(host, error)
            assert "alpha folder" in _painted(host, nav)
            bar = dialog.query_one(InputBar)
            # The explicit "Select folder" label takes more space than
            # "Select"; the path must still paint the whole edited value.
            assert field.region.width >= len(field.value) + 2
            assert field.region.width < bar.region.width
            for selector in ("#select", "#cancel"):
                button = dialog.query_one(selector, Button)
                assert button.region.right <= dialog.query_one(Dialog).region.right
                assert str(button.label) in _painted(host, button)
        await _tab_to(dialog, host, pilot, "#cancel", "Cancel")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: bool(host.results), message="Cancel did not dismiss"
        )
        assert host.results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("count", [0, 24], ids=["empty", "scrolling"])
async def test_compact_empty_or_scrolled_folder_keeps_visible_selection(
    tmp_path, monkeypatch, theme, count
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    listing = tmp_path / "listing"
    listing.mkdir()
    for index in range(count):
        (listing / f"folder-{index:02}").mkdir()
    host = _PickerHost(listing)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        dialog = host.dialog
        nav = await _loaded(dialog, pilot, listing)
        assert nav.option_count == count + 1
        assert dialog.focused is dialog.query_one("#path_input", Input)
        for _ in range(24):
            if dialog.focused is nav:
                break
            await pilot.press("tab")
        assert dialog.focused is nav
        await pilot.press("end")
        selected = nav.get_option_at_index(nav.highlighted).location
        label = selected.name if count else ".."
        assert dialog.focused is nav
        assert nav.highlighted == count
        await _wait_for_condition(
            pilot,
            lambda: label in _painted(host, nav),
            message="Highlighted folder was not painted after End",
        )
        assert nav.content_region.height >= 3
        if count:
            await pilot.press("enter")
            await _loaded(dialog, pilot, selected)
        await pilot.press("escape")
        await _wait_for_condition(
            pilot, lambda: bool(host.results), message="Escape did not cancel"
        )
        assert host.results == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("segments", [1, 36], ids=["short", "long"])
async def test_long_path_error_keeps_listing_and_same_folder_correction_clears_it(
    tmp_path, monkeypatch, theme, segments
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    listing = tmp_path / "listing"
    listing.mkdir()
    (listing / "alpha folder").mkdir()
    host = _PickerHost(listing)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        dialog = host.dialog
        nav = await _loaded(dialog, pilot, listing)
        field = dialog.query_one("#path_input", Input)
        field.focus()
        missing = "/".join(["missing-folder"] * segments)
        field.value = missing
        await pilot.press("enter")
        error = dialog.query_one("#picker-error-line", Static)
        await _wait_for_condition(
            pilot, lambda: error.display, message="Long path error did not show"
        )
        assert "Path not found" in _painted(host, error)
        assert nav.content_region.height >= 3
        assert "alpha folder" in _painted(host, nav)
        # The full diagnostic stays available; compact layout abbreviates paint.
        assert missing in str(error.renderable)
        field.value = str(listing)
        await pilot.press("enter")
        await pilot.pause()
        assert nav.location == listing
        assert not error.display
        assert dialog.focused is field
        assert host.results == []


def test_overlong_folder_name_returns_validation_error(tmp_path):
    from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
        resolve_typed_directory,
    )

    result = resolve_typed_directory("x" * 260, tmp_path)
    assert isinstance(result, str)
    assert "too long" in result.lower()
