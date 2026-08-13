"""Tests for Database Notes folder name and destination dialogs."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input, Select, Static

from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderNameDialog,
    LibraryNoteFolderTargetDialog,
)


@pytest.mark.asyncio
async def test_name_dialog_echoes_action_and_initial_name():
    app = App()
    async with app.run_test() as pilot:
        await app.push_screen(
            LibraryNoteFolderNameDialog(
                title="Rename folder", initial_name="Ideas"
            )
        )
        await pilot.pause()
        assert str(app.screen.query_one("#library-note-folder-dialog-title", Static).renderable) == (
            "Rename folder"
        )
        assert app.screen.query_one("#library-note-folder-name", Input).value == "Ideas"


@pytest.mark.asyncio
async def test_target_dialog_includes_root_and_bounded_folder_choices():
    app = App()
    async with app.run_test() as pilot:
        await app.push_screen(
            LibraryNoteFolderTargetDialog(
                title="Move folder",
                folders=(("Personal", "personal"), ("Personal / Ideas", "ideas")),
                include_root=True,
            )
        )
        await pilot.pause()
        select = app.screen.query_one("#library-note-folder-target", Select)
        assert select.value == ""
        assert len(select._options) == 3
