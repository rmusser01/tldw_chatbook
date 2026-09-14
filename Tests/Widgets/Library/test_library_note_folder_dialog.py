"""Tests for Database Notes folder name and destination dialogs."""

from __future__ import annotations

import pytest
from textual import events
from textual.app import App
from textual.widgets import Input, Select, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderNameDialog,
    LibraryNoteFolderTargetDialog,
)


@pytest.mark.asyncio
async def test_name_dialog_echoes_action_and_initial_name():
    app = ConsolidatedCSSApp()
    async with app.run_test() as pilot:
        await app.push_screen(
            LibraryNoteFolderNameDialog(title="Rename folder", initial_name="Ideas")
        )
        await pilot.pause()
        assert str(
            app.screen.query_one("#library-note-folder-dialog-title", Static).renderable
        ) == ("Rename folder")
        assert app.screen.query_one("#library-note-folder-name", Input).value == "Ideas"


@pytest.mark.asyncio
async def test_target_dialog_includes_root_and_bounded_folder_choices():
    app = ConsolidatedCSSApp()
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


@pytest.mark.asyncio
async def test_target_dialog_without_root_mounts_with_a_blank_selection():
    """TASK-32533: the P0 site. Add note to folder / Move note open this dialog
    with ``include_root=False``; its blank value was ``Select.BLANK``, which is
    not a Select attribute at all and resolves to ``Widget.BLANK`` (False) --
    an illegal value that raised InvalidSelectValueError at mount and, before
    this task, exited the whole app."""
    app = ConsolidatedCSSApp()
    async with app.run_test() as pilot:
        await app.push_screen(
            LibraryNoteFolderTargetDialog(
                title="Add note to folder",
                folders=(("Personal", "personal"), ("Personal / Ideas", "ideas")),
            )
        )
        await pilot.pause()
        assert app._exception is None
        select = app.screen.query_one("#library-note-folder-target", Select)
        assert select.value is Select.NULL
        # Both folders, plus the blank prompt row `allow_blank` prepends.
        assert [value for _, value in select._options] == [
            Select.NULL,
            "personal",
            "ideas",
        ]


@pytest.mark.asyncio
async def test_target_dialog_choose_with_nothing_selected_stays_open():
    """Choose with no folder picked must not dismiss with a bogus folder id."""
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(
        title="Add note to folder",
        folders=(("Personal", "personal"),),
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.click("#library-note-folder-target-confirm")
        await pilot.pause()
        assert app.screen is modal
        assert results == []


def _note_folder_modal(kind: str):
    return (
        LibraryNoteFolderNameDialog(title="Rename folder", initial_name="Ideas")
        if kind == "name"
        else LibraryNoteFolderTargetDialog(
            title="Move folder",
            folders=(("Personal", "personal"), ("Ideas", "ideas")),
            include_root=True,
        )
    )


def _note_folder_control(kind: str, control: str) -> str:
    prefix = (
        "library-note-folder-dialog"
        if kind == "name"
        else "library-note-folder-target"
    )
    return f"#{prefix}-{control}"


@pytest.mark.parametrize("kind", ["name", "target"])
@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_note_folder_library_modal_contract_exact_negative_once(
    kind: str,
    source: str,
) -> None:
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = _note_folder_modal(kind)

    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        assert modal.query_one(f"#library-note-folder-{kind}-dialog")

        if source == "visible":
            await pilot.click(_note_folder_control(kind, "cancel"))
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert results == [None]


@pytest.mark.parametrize("kind", ["name", "target"])
@pytest.mark.asyncio
async def test_note_folder_library_modal_contract_inside_and_non_primary_stay_open(
    kind: str,
) -> None:
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = _note_folder_modal(kind)

    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.click(_note_folder_control(kind, "title"))
        event = events.Click(
            modal,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=3,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=0,
            screen_y=0,
        )
        await modal._dispatch_message(event)
        await pilot.pause()

        assert app.screen is modal
        assert results == []


@pytest.mark.parametrize("kind", ["name", "target"])
@pytest.mark.asyncio
async def test_note_folder_library_modal_contract_positive_is_str(kind: str) -> None:
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = _note_folder_modal(kind)

    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        if kind == "name":
            modal.query_one("#library-note-folder-name", Input).value = "Archive"
        else:
            modal.query_one("#library-note-folder-target", Select).value = "ideas"
        await pilot.click(_note_folder_control(kind, "confirm"))
        await pilot.pause()

    assert len(results) == 1
    assert type(results[0]) is str
    assert results[0] == ("Archive" if kind == "name" else "ideas")


@pytest.mark.parametrize("kind", ["name", "target"])
@pytest.mark.asyncio
async def test_note_folder_repeated_input_dismisses_once(kind: str) -> None:
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = _note_folder_modal(kind)

    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.press("escape", "escape")
        await pilot.pause()

    assert results == [None]
