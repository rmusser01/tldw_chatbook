"""Typed-path Select folder/hint/label contracts for ``FileOpen(offer_select_folder=True)``.

task-32122: this is the dialog behind Library > Notes' "Import once" and
"Keep a folder synced" flows (``library_screen.py``'s
``_push_library_note_import_picker`` and
``library_notes_controller.py``'s ``handle_library_notes_lasting_folder_
requested``). Its "Select folder" button used to always return
``DirectoryNavigation.location`` -- the directory merely being browsed --
and never read the "File name" field at all, even when the user had typed
an absolute path into it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import Dialog, InputBar


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


def _field(dialog) -> Input:
    return dialog.query_one(InputBar).query_one(Input)


@pytest.mark.asyncio
async def test_select_folder_resolves_typed_directory(tmp_path):
    (tmp_path / "vault").mkdir()
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        _field(dialog).value = str(tmp_path / "vault")
        dialog.query_one("#select-current-folder", Button).press()
        await pilot.pause()

    assert app.result == (tmp_path / "vault").resolve()


@pytest.mark.asyncio
async def test_select_folder_with_bad_typed_path_shows_error_and_stays_open(tmp_path):
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        _field(dialog).value = str(tmp_path / "does-not-exist")
        dialog.query_one("#select-current-folder", Button).press()
        await pilot.pause()
        assert "Path not found" in (dialog.query_one(Dialog).border_subtitle or "")

    assert not app.result_seen


@pytest.mark.asyncio
async def test_select_folder_with_a_filename_reports_not_a_directory(tmp_path):
    """A typed path that exists but isn't a directory is a distinct error."""
    (tmp_path / "notes.md").write_text("x")
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        _field(dialog).value = str(tmp_path / "notes.md")
        dialog.query_one("#select-current-folder", Button).press()
        await pilot.pause()
        subtitle = dialog.query_one(Dialog).border_subtitle or ""
        assert "Not a directory" in subtitle

    assert not app.result_seen


@pytest.mark.asyncio
async def test_select_folder_with_empty_field_returns_viewed_directory(tmp_path):
    """No-typing case is unchanged: Select folder confirms the browsed one."""
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        dialog.query_one("#select-current-folder", Button).press()
        await pilot.pause()

    assert app.result == tmp_path.resolve()


@pytest.mark.asyncio
async def test_enter_still_navigates_not_selects(tmp_path):
    """Keep Enter's existing meaning: it descends, it does not confirm."""
    (tmp_path / "vault").mkdir()
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        field = _field(dialog)
        field.value = str(tmp_path / "vault")
        field.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert not app.result_seen


@pytest.mark.asyncio
async def test_label_switches_to_folder_path_when_typed_value_is_a_directory(
    tmp_path,
):
    """AC#2: 'Folder path', not 'File name', once the typed text resolves

    to a directory -- that's exactly when "Select folder" would act on it.
    """
    (tmp_path / "vault").mkdir()
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        label = dialog.query_one("#file-name-label")

        def _text() -> str:
            return getattr(label.renderable, "plain", str(label.renderable))

        assert "File name" in _text()

        field = _field(dialog)
        field.value = str(tmp_path / "vault")
        await pilot.pause()

        assert "Folder path" in _text()
        assert "File name" not in _text()


@pytest.mark.asyncio
async def test_hint_line_names_open_and_select_folder(tmp_path):
    """AC#4: the Enter-vs-Select hint must actually render (task-32122)."""
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        hint = dialog.query_one("#picker-hint-line", Static)
        text = str(hint.renderable)
        assert "Open" in text
        assert "Select folder" in text
        assert "use this folder" in text


def test_default_location_is_home_not_cwd():
    """AC#4: 'Import once' currently opens at the process cwd."""
    dialog = FileOpen(title="Import once", offer_select_folder=True)
    assert dialog._location == Path.home()
