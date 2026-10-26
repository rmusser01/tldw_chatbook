"""Tests for Database Notes folder name and destination dialogs."""

from __future__ import annotations

import asyncio

import pytest
from textual import events
from textual.app import App
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Notes.note_folder_models import NoteFolder, NoteFolderChildPage
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


class _FolderLoader:
    """A picker loader over a fixed folder list, recording every request.

    Matches the way ``LocalNoteFolderRepository.page_folders`` does: the
    query anywhere in the case-folded path, path order, exact total.
    """

    def __init__(self, *paths: str, page_size: int = 20) -> None:
        self.folders = tuple(
            NoteFolder(
                folder_id=path.strip("/").replace("/", "-").casefold(),
                parent_id=None,
                name=path.rsplit("/", 1)[-1],
                path=path,
                normalized_path=path.casefold(),
                version=1,
                deleted=False,
            )
            for path in paths
        )
        self.page_size = page_size
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, query: str, offset: int) -> NoteFolderChildPage:
        self.calls.append((query, offset))
        matching = sorted(
            (f for f in self.folders if query.casefold() in f.normalized_path),
            key=lambda folder: folder.normalized_path,
        )
        page = tuple(matching[offset : offset + self.page_size])
        end = offset + len(page)
        return NoteFolderChildPage(
            folders=page,
            total_folders=len(matching),
            start_offset=offset,
            previous_offset=max(0, offset - self.page_size) if offset else None,
            next_offset=end if end < len(matching) else None,
        )


def _row_labels(screen) -> list[str]:
    return [
        str(row.label)
        for row in screen.query(".library-note-folder-target-row").results(Button)
    ]


def _total(screen) -> str:
    return str(screen.query_one("#library-note-folder-target-total", Static).render())


async def _until(pilot, predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await pilot.pause(0.02)


@pytest.mark.asyncio
async def test_target_dialog_includes_root_and_every_folder_the_loader_pages():
    loader = _FolderLoader("/Personal", "/Personal/Ideas")
    app = ConsolidatedCSSApp()
    async with app.run_test() as pilot:
        await app.push_screen(
            LibraryNoteFolderTargetDialog(
                title="Move folder", loader=loader, include_root=True
            )
        )
        await _until(pilot, lambda: _total(app.screen) == "2 folders")
        assert _row_labels(app.screen) == ["Top level", "Personal", "Personal / Ideas"]
        assert loader.calls == [("", 0)]
        # The filter owns focus on open; the pager hides on a single page.
        assert app.screen.focused is app.screen.query_one(
            "#library-note-folder-target-filter", Input
        )
        assert not app.screen.query_one("#library-note-folder-target-pager").display


@pytest.mark.asyncio
async def test_target_dialog_without_root_mounts_with_no_preselected_folder():
    """TASK-32533's site, without its ``Select``: no folder is pre-chosen."""
    loader = _FolderLoader("/Personal", "/Personal/Ideas")
    app = ConsolidatedCSSApp()
    async with app.run_test() as pilot:
        await app.push_screen(
            LibraryNoteFolderTargetDialog(title="Add note to folder", loader=loader)
        )
        await _until(pilot, lambda: _total(app.screen) == "2 folders")
        assert app._exception is None
        assert _row_labels(app.screen) == ["Personal", "Personal / Ideas"]
        assert not app.screen.query(".library-note-folder-target-row.-highlighted")


@pytest.mark.asyncio
async def test_target_dialog_choose_with_nothing_selected_stays_open():
    """Choose with no folder picked must not dismiss with a bogus folder id."""
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(
        title="Add note to folder", loader=_FolderLoader("/Personal")
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: _total(modal) == "1 folder")
        await pilot.click("#library-note-folder-target-confirm")
        await pilot.pause()
        assert app.screen is modal
        assert results == []


@pytest.mark.asyncio
async def test_target_dialog_typing_narrows_the_list_after_a_debounce():
    """TASK-34000.28 (N-23): type-to-filter over every folder, exact total."""
    loader = _FolderLoader(
        "/Projects", "/Projects/Thesis", "/Projects/App launch", "/Study"
    )
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(LibraryNoteFolderTargetDialog(title="Move", loader=loader))
        await _until(pilot, lambda: _total(app.screen) == "4 folders")
        await pilot.press("t", "h", "e", "s")
        await _until(pilot, lambda: _total(app.screen) == "1 folder")
        assert _row_labels(app.screen) == ["Projects / Thesis"]
        # One request for the settled text, not one per keystroke.
        assert loader.calls == [("", 0), ("thes", 0)]

        await pilot.press("z")
        await _until(pilot, lambda: _total(app.screen) == "No folder matches “thesz”")
        assert _row_labels(app.screen) == []


@pytest.mark.asyncio
async def test_target_dialog_pages_with_the_exact_total():
    loader = _FolderLoader(*(f"/Folder {index:02d}" for index in range(37)))
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(90, 40)) as pilot:
        await app.push_screen(LibraryNoteFolderTargetDialog(title="Move", loader=loader))
        await _until(pilot, lambda: _total(app.screen) == "Showing 1–20 of 37 folders")
        assert len(_row_labels(app.screen)) == 20
        assert app.screen.query_one("#library-note-folder-target-pager").display
        assert app.screen.query_one("#library-note-folder-target-previous").disabled

        app.screen.query_one("#library-note-folder-target-next", Button).press()
        await _until(pilot, lambda: _total(app.screen) == "Showing 21–37 of 37 folders")
        assert _row_labels(app.screen)[0] == "Folder 20"
        assert app.screen.query_one("#library-note-folder-target-next").disabled
        assert loader.calls == [("", 0), ("", 20)]


@pytest.mark.asyncio
async def test_target_dialog_down_from_the_filter_and_enter_choose_a_row():
    loader = _FolderLoader("/Projects", "/Projects/Thesis")
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(
            LibraryNoteFolderTargetDialog(title="Move", loader=loader),
            callback=results.append,
        )
        await _until(pilot, lambda: _total(app.screen) == "2 folders")
        await pilot.press("down")
        await pilot.pause()
        assert str(app.screen.focused.label) == "Projects"
        await pilot.press("down")
        await pilot.pause()
        assert str(app.screen.focused.label) == "Projects / Thesis"
        await pilot.press("up", "up")
        await pilot.pause()
        assert app.screen.focused.id == "library-note-folder-target-filter"
        await pilot.press("down", "down", "enter")
        await pilot.pause()

    assert results == ["projects-thesis"]


@pytest.mark.asyncio
async def test_target_dialog_paints_user_text_literally():
    """A title or folder name that looks like markup is painted as typed."""
    # (A folder NAME cannot hold "/", the path separator.)
    loader = _FolderLoader("/[b]Bold", "/x [red] y")
    app = ConsolidatedCSSApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(
            LibraryNoteFolderTargetDialog(title='Move "[/]odd[b]" to…', loader=loader)
        )
        await _until(pilot, lambda: _total(app.screen) == "2 folders")
        assert app._exception is None
        title = app.screen.query_one("#library-note-folder-target-title", Static)
        assert str(title.render()) == 'Move "[/]odd[b]" to…'
        labels = [
            str(row.label.plain)
            for row in app.screen.query(".library-note-folder-target-row").results(
                Button
            )
        ]
        assert labels == ["[b]Bold", "x [red] y"]


def _note_folder_modal(kind: str):
    return (
        LibraryNoteFolderNameDialog(title="Rename folder", initial_name="Ideas")
        if kind == "name"
        else LibraryNoteFolderTargetDialog(
            title="Move folder",
            loader=_FolderLoader("/Personal", "/Ideas"),
            include_root=True,
        )
    )


def _note_folder_control(kind: str, control: str) -> str:
    prefix = (
        "library-note-folder-dialog" if kind == "name" else "library-note-folder-target"
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
            await _until(pilot, lambda: len(_row_labels(modal)) == 3)
            next(
                row
                for row in modal.query(".library-note-folder-target-row").results(Button)
                if row.choice_value == "ideas"
            ).focus()
            await pilot.pause()
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
