"""Typed-path Select/hint/label contracts for the vendored ``SelectDirectory``.

task-32122: this is the dialog behind Library > Notes > Folder files' "Choose
folder" affordance (``library_file_notes_workspace.py``'s ``#file-notes-
choose-root`` button). Its ``#select`` button used to return only
``DirectoryNavigation.location`` -- the directory merely being browsed --
silently discarding a path the user typed but never pressed Enter on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation


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


@pytest.mark.asyncio
async def test_select_resolves_typed_but_unsubmitted_path(tmp_path):
    (tmp_path / "vault").mkdir()
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        path_input = dialog.query_one("#path_input", Input)
        path_input.value = str(tmp_path / "vault")
        dialog.query_one("#select", Button).press()
        await pilot.pause()

    assert app.result == (tmp_path / "vault").resolve()


@pytest.mark.asyncio
async def test_select_with_bad_typed_path_shows_error_and_stays_open(tmp_path):
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        path_input = dialog.query_one("#path_input", Input)
        path_input.value = str(tmp_path / "does-not-exist")
        dialog.query_one("#select", Button).press()
        await pilot.pause()
        # task-32251 AC#2: the reason now renders on its own row under
        # the field, not inside the dialog's bottom border.
        assert "Path not found" in str(
            dialog.query_one("#picker-error-line", Static).renderable
        )

    assert not app.result_seen


@pytest.mark.asyncio
async def test_select_with_no_typed_edit_still_returns_viewed_directory(tmp_path):
    """No-typing case is unchanged: Select confirms the browsed directory."""
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dialog.query_one("#select", Button).press()
        await pilot.pause()

    assert app.result == tmp_path.resolve()


@pytest.mark.asyncio
async def test_field_is_labelled_folder_path(tmp_path):
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        label = dialog.query_one("#path-input-label")
        text = getattr(label.renderable, "plain", str(label.renderable))
        assert "Folder path" in text
        assert "File name" not in text


@pytest.mark.asyncio
async def test_hint_line_names_enter_open_and_select(tmp_path):
    """AC#4: the Enter-vs-Select hint must actually render (task-32122)."""
    dialog = SelectDirectory(tmp_path, title="Choose File Notes Folder")
    app = _DialogHost(dialog)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        hint = dialog.query_one("#picker-hint-line", Static)
        text = str(hint.renderable)
        assert "Open" in text
        assert "use this folder" in text


def test_default_location_is_home_not_cwd():
    """The Import once picker (and any other default-'.' picker) currently

    opens at the process cwd; the fix belongs in the shared base so every
    vendored dialog gets it, not just this one caller.
    """
    dialog = SelectDirectory(title="Choose File Notes Folder")
    assert dialog._location == Path.home()
