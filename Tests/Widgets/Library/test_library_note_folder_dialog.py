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


#: The review's probe tree (fix round 1, I-1): "stu" matches only Study,
#: while row 0 of the unfiltered list is Projects (or Top level for a folder).
_PROBE_PATHS = ("/Projects", "/Projects/Thesis", "/Study")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include_root", "first_row"),
    [(False, "Projects"), (True, "Top level")],
    ids=["move-note", "move-folder"],
)
async def test_down_then_enter_before_the_filter_lands_never_chooses_a_pre_filter_row(
    include_root: bool, first_row: str
):
    """TASK-34000.28 fix round 1 (I-1), the reviewer's probe replayed exactly.

    Typing "stu" and pressing Down then Enter with no pause used to focus row
    0 of the OLD list (Projects; Top level for a folder move) while the
    debounced filter was still pending, and Enter moved the item there
    (``['projects']`` / ``['']`` on ``114213d212``). Now Down asks for the
    narrowed list at once and lands on ITS first row, so the Enter chooses
    Study -- or, had it beaten the page (the held arm below), nothing yet.
    """
    loader = _FolderLoader(*_PROBE_PATHS)
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(
        title="Move", loader=loader, include_root=include_root
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        assert _row_labels(modal)[0] == first_row

        await pilot.press("s", "t", "u", "down", "enter")
        await _until(
            pilot,
            lambda: bool(results)
            or (
                _row_labels(modal) == ["Study"]
                and str(getattr(modal.focused, "label", "")) == "Study"
            ),
        )
        assert results in ([], ["study"]), f"chose a pre-filter row: {results}"
        if not results:
            await pilot.press("enter")
            await pilot.pause()

    assert results == ["study"]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_root", [False, True], ids=["move-note", "move-folder"])
async def test_an_enter_that_beats_the_narrowed_page_chooses_nothing(include_root: bool):
    """I-1, deterministic: the narrowed page is held, so Down and Enter both
    arrive while the rows on screen are the unfiltered list."""
    loader = _HeldLoader(*_PROBE_PATHS)
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(
        title="Move", loader=loader, include_root=include_root
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.press("s", "t", "u", "down", "enter")
        await _until(pilot, lambda: bool(results) or len(loader.calls) == 2)
        await pilot.pause()
        assert results == [], f"chose a pre-filter row: {results}"
        assert app.screen is modal
        # Down asked for "stu" at once rather than waiting out the debounce.
        assert loader.calls[-1] == ("stu", 0)

        loader.release.set()
        await _until(pilot, lambda: str(getattr(modal.focused, "label", "")) == "Study")
        assert results == []
        await pilot.press("enter")
        await pilot.pause()

    assert results == ["study"]


@pytest.mark.asyncio
async def test_choose_while_the_filter_is_pending_never_confirms_a_pre_filter_row():
    """I-1's Choose half: Top level is preselected for a folder move, but
    once "stu" is typed it is no longer listed, so Choose before the narrowed
    list lands must not move the folder to the top level."""
    loader = _FolderLoader(*_PROBE_PATHS)
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(title="Move", loader=loader, include_root=True)
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.press("s", "t", "u")
        modal.query_one("#library-note-folder-target-confirm", Button).press()
        await _until(pilot, lambda: bool(results) or _row_labels(modal) == ["Study"])
        await pilot.pause()
        assert results == [], f"Choose confirmed a pre-filter row: {results}"
        assert app.screen is modal


@pytest.mark.asyncio
async def test_a_stale_row_reached_by_tab_does_not_choose_while_the_filter_is_pending():
    """I-1's row half: Tab reaches the old list's first row before the
    debounce answers; Enter there asks for the narrowed list instead."""
    loader = _FolderLoader(*_PROBE_PATHS)
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(title="Move", loader=loader)
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.press("s", "t", "u", "tab", "enter")
        await _until(pilot, lambda: bool(results) or _row_labels(modal) == ["Study"])
        await pilot.pause()
        assert results == [], f"a stale row chose: {results}"


@pytest.mark.asyncio
async def test_tab_from_the_filter_reaches_the_first_row():
    """Fix round 1 (M-1): the row list itself is not a Tab stop."""
    app = ConsolidatedCSSApp()
    modal = LibraryNoteFolderTargetDialog(
        title="Move", loader=_FolderLoader(*_PROBE_PATHS)
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.press("tab")
        await pilot.pause()
        focused = modal.focused
        assert isinstance(focused, Button), focused
        assert str(focused.label) == "Projects"


class _HeldLoader(_FolderLoader):
    """Answers the first page at once; holds (or fails) every later one."""

    def __init__(self, *paths: str, fail: bool = False) -> None:
        super().__init__(*paths)
        self.release = asyncio.Event()
        self.fail = fail

    async def __call__(self, query: str, offset: int) -> NoteFolderChildPage:
        if self.calls:
            self.calls.append((query, offset))
            if self.fail:
                raise RuntimeError("folder page offline")
            await self.release.wait()
            self.calls.pop()
        return await super().__call__(query, offset)


@pytest.mark.asyncio
async def test_the_total_reads_loading_while_a_narrowed_page_is_on_its_way():
    """Fix round 1 (M-4): no stale total beside a list that is about to change."""
    loader = _HeldLoader(*_PROBE_PATHS)
    app = ConsolidatedCSSApp()
    modal = LibraryNoteFolderTargetDialog(title="Move", loader=loader)
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.press("s", "t", "u")
        await _until(pilot, lambda: len(loader.calls) == 2)
        await pilot.pause()
        assert _total(modal) == "Loading folders…"
        loader.release.set()
        await _until(pilot, lambda: _total(modal) == "1 folder")
        assert _row_labels(modal) == ["Study"]


@pytest.mark.asyncio
async def test_a_failed_page_leaves_nothing_to_choose():
    """Fix round 1 (M-4): after "Could not load folders." no row of the
    previous query is listed, and Choose is disabled."""
    loader = _HeldLoader(*_PROBE_PATHS, fail=True)
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(title="Move", loader=loader, include_root=True)
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.press("s", "t", "u")
        await _until(pilot, lambda: _total(modal) == "Could not load folders.")
        await pilot.pause()
        assert _row_labels(modal) == []
        confirm = modal.query_one("#library-note-folder-target-confirm", Button)
        assert confirm.disabled
        await pilot.press("down", "enter")
        await pilot.pause()
        assert results == []
        assert app.screen is modal


class _HeldFirstPageLoader(_FolderLoader):
    """Holds even the FIRST page until ``release`` is set."""

    def __init__(self, *paths: str) -> None:
        super().__init__(*paths)
        self.release = asyncio.Event()

    async def __call__(self, query: str, offset: int) -> NoteFolderChildPage:
        self.calls.append((query, offset))
        await self.release.wait()
        self.calls.pop()
        return await super().__call__(query, offset)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_root", [False, True], ids=["move-note", "move-folder"])
async def test_a_down_the_user_then_tabbed_away_from_never_lands_on_row_zero(
    include_root: bool,
):
    """Final review M-10: Down before the first page lands arms "focus the
    first row on load". Tab then takes the user to Cancel; the landing page
    used to pull focus back to row 0 anyway, so the Enter meant for Cancel
    chose that row (a move the user never asked for). A move of the user's
    own to anything but a row or the filter withdraws the arm."""
    loader = _HeldFirstPageLoader(*_PROBE_PATHS)
    app = ConsolidatedCSSApp()
    results: list[str | None] = []
    modal = LibraryNoteFolderTargetDialog(
        title="Move", loader=loader, include_root=include_root
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: loader.calls == [("", 0)])
        assert modal.focused is modal.query_one(
            "#library-note-folder-target-filter", Input
        )
        await pilot.press("down", "tab")
        await _until(
            pilot,
            lambda: getattr(modal.focused, "id", None)
            == "library-note-folder-target-cancel",
        )

        loader.release.set()
        await _until(pilot, lambda: _total(modal) == "3 folders")
        await pilot.pause()
        assert getattr(modal.focused, "id", None) == "library-note-folder-target-cancel", (
            f"the landing page took focus to {modal.focused!r}"
        )
        await pilot.press("enter")
        await _until(pilot, lambda: bool(results))

    assert results == [None], f"Enter on Cancel chose a folder: {results}"


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
