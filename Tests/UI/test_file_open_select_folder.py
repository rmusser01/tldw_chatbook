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
async def test_select_folder_after_clicking_a_file_falls_back_to_browsed_directory(
    tmp_path,
):
    """Important 1 (review round 2): a single click on a file pre-fills the

    field with its basename (``file_dialog.py``'s ``_select_file``, so the
    user can then press Open) -- Select folder must not mistake that for a
    typed folder path and error with "Not a directory: notes.md" where it
    used to just return the directory being browsed. Arrives via the same
    OptionList selection path a real click uses (``action_select`` on the
    highlighted row), not by setting the Input directly.
    """
    (tmp_path / "notes.md").write_text("x")
    dialog = FileOpen(tmp_path, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        nav = dialog.query_one(DirectoryNavigation)
        for _ in range(20):
            if nav.option_count > 0:
                break
            await pilot.pause()
        index = next(
            i
            for i in range(nav.option_count)
            if nav.get_option_at_index(i).location.name == "notes.md"
        )
        nav.highlighted = index
        nav.action_select()
        await pilot.pause()
        assert _field(dialog).value == "notes.md"

        dialog.query_one("#select-current-folder", Button).press()
        await pilot.pause()

    assert app.result == tmp_path.resolve()


@pytest.mark.asyncio
async def test_select_folder_with_typed_dotdot_and_no_click_resolves_to_parent(
    tmp_path,
):
    """Round-2 fix's regression (review round 3): ``DirectoryNavigation.

    _settle_highlight`` defaults ``highlighted`` to 0 on every repopulate,
    including the first load, and ".." is always option 0 in a non-root
    directory -- so comparing the typed value against "whatever's
    highlighted" (instead of an explicit click-provenance flag) falsely
    treated a typed, never-clicked ".." as a click-fill echo and silently
    swallowed it. Opened on ``sub/``, typed ".." with no click, Select
    folder must resolve to ``sub``'s parent, not dismiss with ``sub``
    itself.
    """
    sub = tmp_path / "sub"
    sub.mkdir()
    dialog = FileOpen(sub, title="Import once", offer_select_folder=True)
    app = _DialogHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()
        _field(dialog).value = ".."
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
