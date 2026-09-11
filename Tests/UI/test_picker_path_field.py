"""One field behaviour for every picker path field (task-32251).

Live reproduction on dev 4a14b3f36f: Library > Notes > Folder files >
"Choose folder", click the pre-filled "Folder path:" field, type an
absolute path -- the field held
``/Users/macbook-dev/Users/macbook-dev/.cache/...`` and pressing Enter
painted ``Path not found: ...`` INSIDE the dialog's bottom border.

Both halves are fixed at the shared seam (``base_dialog.PathInput`` and
``FileSystemPickerScreen._set_error``), so the assertions below run
against all three families: the vendored ``SelectDirectory`` (Folder
files), the vendored ``FileOpen`` (Import once / Keep a folder synced),
and ``EnhancedSelectDirectory``.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, SelectDirectory
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
    Dialog,
    InputBar,
    PathInput,
)


class _DialogHost(App[None]):
    """Minimal host that immediately pushes the dialog under test."""

    def __init__(self, dialog):
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self._dialog)


async def _click_to_focus(pilot, field: Input) -> None:
    """Click an UNFOCUSED field the way a terminal delivers it.

    The pointer crosses the field (``MouseMove``), the click focuses it
    (``Focus``), and only then does the widget see the ``MouseDown``.
    `Pilot.click` alone posts no ``MouseMove``, which is precisely the
    signal `PathInput` uses to tell a deliberate click-to-place-the-caret
    from the click that focused the field -- so a test that skips the
    hover is not testing the real path (review F2).
    """
    assert not field.has_focus, "this helper is for the click-to-focus path"
    await pilot.hover(field, offset=(3, 0))
    await pilot.pause()
    await pilot.click(field, offset=(3, 0))
    await pilot.pause()


async def _click_focused_field(pilot, field: Input) -> None:
    """Click a field that ALREADY has focus, to place the caret."""
    assert field.has_focus
    await pilot.hover(field, offset=(3, 0))
    await pilot.pause()
    await pilot.click(field, offset=(3, 0))
    await pilot.pause()


@pytest.mark.asyncio
async def test_folder_picker_field_is_replaced_by_a_typed_absolute_path(tmp_path):
    """AC#1: click into the pre-filled field and type -- no concatenation."""
    (tmp_path / "vault").mkdir()
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = dialog.query_one("#path_input", Input)
        assert field.value == str(tmp_path), "the field arrives pre-filled"
        await _click_to_focus(pilot, field)
        await pilot.press(*str(tmp_path / "vault"))
        await pilot.pause()
        assert field.value == str(tmp_path / "vault")


@pytest.mark.asyncio
async def test_file_picker_field_is_replaced_by_a_typed_absolute_path(tmp_path):
    """The same seam under ``FileOpen`` (Import once / Keep a folder synced)."""
    (tmp_path / "vault").mkdir()
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = dialog.query_one(InputBar).query_one(Input)
        assert isinstance(field, PathInput)
        field.value = str(tmp_path)
        await pilot.pause()
        await _click_to_focus(pilot, field)
        await pilot.press(*str(tmp_path / "vault"))
        await pilot.pause()
        assert field.value == str(tmp_path / "vault")


@pytest.mark.asyncio
async def test_a_second_click_positions_the_cursor_instead_of_reselecting(tmp_path):
    """Select-on-focus must not make the field un-editable by mouse."""
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = dialog.query_one("#path_input", Input)
        await _click_to_focus(pilot, field)
        await _click_focused_field(pilot, field)
        assert not field.selected_text


@pytest.mark.asyncio
async def test_clicking_into_a_tab_focused_field_places_the_caret(tmp_path):
    """Review F2: Tab in, then click to place the caret -- and it stays placed.

    Tab focus already selects the whole value (Textual's own
    ``select_on_focus``), so a re-select on the next click is invisible
    until the following keystroke wipes the path.
    """
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = dialog.query_one("#path_input", Input)
        field.focus()
        await pilot.pause()
        assert field.selected_text == field.value, "Tab focus selects the value"

        await _click_focused_field(pilot, field)
        assert not field.selected_text

        await pilot.press("x")
        await pilot.pause()
        assert field.value != "x", "the click-placed caret was not honoured"
        assert str(tmp_path) in field.value.replace("x", "", 1)


@pytest.mark.asyncio
async def test_a_bad_typed_path_reports_under_the_field_not_in_the_border(tmp_path):
    """AC#2: the reason gets a row of its own; the border stays clean."""
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = dialog.query_one("#path_input", Input)
        field.value = str(tmp_path / "definitely-not-here")
        field.focus()
        await pilot.press("enter")
        await pilot.pause()

        error_line = dialog.query_one("#picker-error-line", Static)
        assert error_line.display
        assert "Path not found" in str(error_line.renderable)
        assert not (dialog.query_one(Dialog).border_subtitle or "")


@pytest.mark.asyncio
async def test_the_error_row_clears_once_navigation_succeeds(tmp_path):
    (tmp_path / "vault").mkdir()
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = dialog.query_one("#path_input", Input)
        field.value = str(tmp_path / "definitely-not-here")
        field.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert dialog.query_one("#picker-error-line", Static).display

        field.value = str(tmp_path / "vault")
        await pilot.press("enter")
        await pilot.pause()
        assert not dialog.query_one("#picker-error-line", Static).display
