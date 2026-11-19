"""TASK-34000.14: the Library Notes tree shows its folders after a Create arrival.

Review finding N-13 (notes-library-ux-review-2026-10-02): a power user
capturing from Console through the palette ("New Note" → Blank note → Esc)
came back to a tree that started at "▾ Unfiled" -- Agent_Lessons, Journal,
Meetings and the rest were missing, "g go to folder" landed on Unfiled, and
"Add note to folder" opened with nothing to choose. The tree projects folder
rows only from the root folders slice, and only the rail's Notes row ever
requested it: the create path's reconcile located the new note through root
PLACEMENTS alone. Every walk here runs the real Library hierarchy with the
production stylesheet over a real ChaChaNotes database holding three
top-level folders, at the review's size (160x45). The palette walk applies
its navigation context the way ``app.py`` does -- before the screen mounts;
the full-app ``NavigateToScreen`` version, the Move-note arm and the 80x24
repeat live in the ``_extended`` sibling, which is not in the PR lane.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _armed_editor,
    _new_blank_note,
    _NotesProfile,
    _type,
    _until,
)
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
    _JourneyHarness,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_NOTES_CREATE
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderTargetDialog,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Three top-level folders: enough to assert presence AND order.
FOLDERS = ("Journal", "Meetings", "Projects")

#: The title the review typed into its capture.
TITLE = "Capture from console"

ROOT_FOLDERS = NotesBranchKey(None, "folders")
ROOT_PLACEMENTS = NotesBranchKey(None, "placements")


def _seed_folders(profile: _NotesProfile) -> None:
    repository = LocalNoteFolderRepository(profile.db)
    for name in FOLDERS:
        repository.create_folder(name=name, parent_id=None)


def _profiled_app(tmp_path) -> tuple:
    """The production app object over a real notes database with folders."""
    app = _build_test_app()
    _seed_conversations(app, [])
    profile = _NotesProfile(tmp_path)
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    _seed_folders(profile)
    return app, profile


def _mounted(tmp_path, *, navigation_context: dict | None = None) -> tuple:
    """A Library host ready to run; ``navigation_context`` is applied pre-mount.

    Mirrors ``app.py``'s ordering: ``handle_screen_navigation`` applies the
    palette's context before the destination screen mounts.
    """
    app, profile = _profiled_app(tmp_path)
    screen = LibraryScreen(app)
    if navigation_context:
        screen.apply_navigation_context(navigation_context)
    return _JourneyHarness(app, screen=screen), profile


async def _shell(host, pilot) -> LibraryScreen:
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    return screen


def _folder_rows(screen) -> list[Button]:
    """The tree's real folder rows: the automatic Unfiled group excluded."""
    return [
        row
        for row in screen.query(".library-notes-folder-row").results(Button)
        if not row.has_class("library-notes-unfiled-row")
    ]


def _painted_list(host, screen) -> str:
    """Whatever the compositor paints inside the notes list, unwrapped."""
    strips = list(host.screen._compositor.render_strips())
    region = screen.query_one("#library-notes-list").region
    rows = [
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, min(region.bottom, len(strips)))
    ]
    return " ".join(" ".join(rows).split())


def _count_root_placement_pages(monkeypatch, profile) -> list[tuple]:
    """Record every root placements page the real scope service serves."""
    pages: list[tuple] = []
    real_page = profile.scope_service.page_note_placements

    async def counted(**kwargs):
        if kwargs.get("parent_id") is None:
            pages.append((kwargs.get("offset"), kwargs.get("order")))
        return await real_page(**kwargs)

    monkeypatch.setattr(profile.scope_service, "page_note_placements", counted)
    return pages


async def _tree_settled(screen, pilot) -> None:
    """Wait until the root slices have answered, or were never asked for.

    A root folders slice that is absent is the defect itself, so the wait
    lets the assertions name it; one that is loading, or that a mutation
    fence left ``uninitialized``, is still owed and is waited for.
    """

    def settled() -> bool:
        branches = screen._notes_state.tree_branches
        placements = branches.get(ROOT_PLACEMENTS)
        if placements is None or placements.loading:
            return False
        folders = branches.get(ROOT_FOLDERS)
        if folders is None:
            return True
        return not folders.loading and folders.freshness != "uninitialized"

    # A settled response schedules a canvas recompose; its mounted list
    # can still have a zero region. Witness baseline paint in the same
    # budget, leaving missing folders to the independent assertions below.
    await _until(
        pilot,
        lambda: settled()
        and bool(screen.query("#library-notes-list"))
        and "▾ Unfiled" in _painted_list(screen.app, screen),
        "the root slices to settle and the current Notes list to paint",
        timeout=15.0,
    )


def _assert_folders_above_unfiled(painted: str, *, note_title: str = "") -> None:
    assert "▾ Unfiled" in painted, painted
    unfiled = painted.index("▾ Unfiled")
    for name in FOLDERS:
        assert f"▸ {name}" in painted, f"{name!r} is not in the tree: {painted}"
        assert painted.index(f"▸ {name}") < unfiled, painted
    assert (
        painted.index("▸ Journal")
        < painted.index("▸ Meetings")
        < painted.index("▸ Projects")
    ), painted
    if note_title:
        assert note_title in painted, painted
        assert painted.index(note_title) > unfiled, painted


async def _painted_folders_above_unfiled(
    host, screen, pilot, *, note_title: str = ""
) -> str:
    """Wait for the painted list to show every folder above Unfiled; assert it.

    TASK-34000.95: the root slices can settle in state a frame before the
    compositor paints them, so one read on a slow CI runner saw an empty list
    (``assert '▾ Unfiled' in ''``). This retries the same assertion against
    fresh paints. If it never holds -- the TASK-34000.14 defect, a tree of
    Unfiled plus the new note -- the assertion runs once more after the wait,
    so the failure still names the missing folder rather than a timeout.
    """

    def holds() -> bool:
        if not screen.query("#library-notes-list"):
            return False
        try:
            _assert_folders_above_unfiled(
                _painted_list(host, screen), note_title=note_title
            )
        except AssertionError:
            return False
        return True

    try:
        await _until(
            pilot, holds, "the painted list to show every folder", timeout=10.0
        )
    except AssertionError:
        pass
    painted = _painted_list(host, screen)
    _assert_folders_above_unfiled(painted, note_title=note_title)
    return painted


async def _title_then_escape(screen, pilot, title: str) -> None:
    """Give the blank note a title (so it is kept) and leave the editor."""
    screen.query_one("#library-note-title", Input).focus()
    await pilot.pause()
    await _type(pilot, title)
    body = screen.query_one("#library-note-body", TextArea)
    body.focus()
    await pilot.pause(0.1)
    await pilot.press("escape")
    await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")


async def _open_folder_picker(host, screen, pilot, button_id: str) -> list[str]:
    """Press a toolbar folder action; return the picker's folder rows.

    TASK-34000.28 (N-23): the picker pages every active folder into row
    buttons rather than a ``Select`` of the loaded branches, so its folders
    are read once its first page has painted (the total line has replaced
    "Loading folders…").
    """
    button = await _wait_for_selector(screen, pilot, button_id)
    button.press()
    await _until(
        pilot,
        lambda: isinstance(host.screen, LibraryNoteFolderTargetDialog),
        "the folder picker to open",
    )
    await _until(
        pilot,
        lambda: any(
            "Loading folders…" not in str(total.render())
            for total in host.screen.query("#library-note-folder-target-total")
        ),
        "the folder picker's first page",
    )
    prompts = [
        str(row.label)
        for row in host.screen.query(".library-note-folder-target-row").results(
            Button
        )
    ]
    await pilot.press("escape")
    await _until(pilot, lambda: host.screen is screen, "the folder picker to close")
    return prompts


async def test_palette_new_note_then_blank_then_escape_shows_every_top_level_folder(
    tmp_path, monkeypatch
) -> None:
    """AC#1 and AC#4 (palette route).

    ``execute_quick_action("new_note")`` posts ``NavigateToScreen(TAB_LIBRARY,
    {notes_create: True})`` and ``app.py`` applies that context before the
    Library mounts; the same context arrives here the same way, so the
    Library opens straight on Create.
    """
    host, profile = _mounted(
        tmp_path, navigation_context={LIBRARY_NAV_CONTEXT_NOTES_CREATE: True}
    )
    root_placement_pages = _count_root_placement_pages(monkeypatch, profile)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _shell(host, pilot)
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank.press()
        await _armed_editor(screen, pilot)
        await _title_then_escape(screen, pilot, TITLE)
        await _tree_settled(screen, pilot)

        await _painted_folders_above_unfiled(host, screen, pilot, note_title=TITLE)
        # Fix round 1 (M-1): the locator already loads root placements for
        # the new note; the create arrival must not fetch them a second time.
        assert len(root_placement_pages) == 1, root_placement_pages
        assert [
            name
            for name in FOLDERS
            if any(name in str(row.label) for row in _folder_rows(screen))
        ] == list(FOLDERS)


async def test_rail_new_note_then_escape_shows_the_folders_and_add_to_folder_lists_them(
    tmp_path,
) -> None:
    """AC#1 (rail route), AC#2 and AC#4: the picker lists the folders."""
    host, _profile = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _shell(host, pilot)
        await _new_blank_note(screen, pilot)
        await _title_then_escape(screen, pilot, TITLE)
        await _tree_settled(screen, pilot)

        prompts = await _open_folder_picker(
            host, screen, pilot, "#library-notes-placement-add"
        )
        assert prompts == list(FOLDERS), (
            f"Add note to folder lists {prompts!r}; tree: {_painted_list(host, screen)}"
        )
        await _painted_folders_above_unfiled(host, screen, pilot, note_title=TITLE)


async def test_a_template_create_lands_in_a_tree_with_its_folders(tmp_path) -> None:
    """AC#1 "or a template": the template rows share the create seam."""
    host, _profile = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _shell(host, pilot)
        screen.query_one("#library-row-create-note", Button).press()
        disclosure = await _wait_for_selector(
            screen, pilot, "#library-note-from-template"
        )
        disclosure.press()
        template = await _wait_for_selector(
            screen, pilot, ".library-notes-template-row"
        )
        template.press()
        await _armed_editor(screen, pilot)
        screen.query_one("#library-note-body", TextArea).focus()
        await pilot.pause(0.1)
        await pilot.press("escape")
        await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
        await _tree_settled(screen, pilot)

        await _painted_folders_above_unfiled(host, screen, pilot)


async def test_loading_row_paints_while_the_root_slice_loads_and_g_is_not_advertised_on_unfiled(
    tmp_path, monkeypatch
) -> None:
    """Sharpened AC#3: a slow root folders page is a visible loading state.

    The real scope service answers every page except the ROOT folders page,
    which is held open until the test releases it. While it is held the list
    shows "Loading folders…" above Unfiled, and "g go to folder" is neither
    advertised nor live -- Unfiled is not a folder to go to. Once the page
    lands the folders paint, the chip returns, and ``g`` lands on the first
    real folder row.
    """
    host, profile = _mounted(tmp_path)
    gate = asyncio.Event()
    real_page = profile.scope_service.page_note_folder_children

    async def held_root_page(**kwargs):
        if kwargs.get("parent_id") is None:
            await gate.wait()
        return await real_page(**kwargs)

    monkeypatch.setattr(
        profile.scope_service, "page_note_folder_children", held_root_page
    )
    async with host.run_test(size=SIZE) as pilot:
        screen = await _shell(host, pilot)
        await _new_blank_note(screen, pilot)
        note_id = screen._notes_state.selected_note_id
        assert note_id
        # Fix round 1 (M-2): the create reconcile itself owes the root
        # folders slice -- by the time the editor is armed it is requested
        # (held here) or already fresh; without that arm the list would
        # only get it from the editor-return belt.
        requested = screen._notes_state.tree_branches.get(ROOT_FOLDERS)
        assert requested is not None and (
            requested.loading or requested.freshness == "fresh"
        ), requested
        await _title_then_escape(screen, pilot, TITLE)
        await _until(
            pilot,
            lambda: (
                screen.focused is not None
                and screen.focused.has_class("library-notes-tree-note-row")
                and getattr(screen.focused, "note_id", "") == note_id
            ),
            "focus to return to the new note's row",
        )

        assert screen.check_action("library_notes_focus_folders", ()) is False, (
            "g is advertised while the root folder slice has not loaded: "
            f"{_painted_list(host, screen)}"
        )
        assert "g" not in dict(screen._library_notes_footer_shortcuts())
        await _until(
            pilot,
            lambda: "Loading folders…" in _painted_list(host, screen),
            "the Loading folders… row",
            timeout=10.0,
        )
        painted = _painted_list(host, screen)
        assert painted.index("Loading folders…") < painted.index("▾ Unfiled"), painted
        assert not _folder_rows(screen)

        gate.set()
        await _until(
            pilot,
            lambda: len(_folder_rows(screen)) == len(FOLDERS),
            "the folder rows to paint",
        )
        painted = await _painted_folders_above_unfiled(
            host, screen, pilot, note_title=TITLE
        )
        assert "Loading folders…" not in painted
        assert screen.check_action("library_notes_focus_folders", ()) is True
        assert dict(screen._library_notes_footer_shortcuts()).get("g") == "go to folder"

        await pilot.press("g")
        await pilot.pause()
        focused = screen.focused
        assert isinstance(focused, Button), focused
        assert focused.has_class("library-notes-folder-row"), focused
        assert not focused.has_class("library-notes-unfiled-row"), str(focused.label)
        assert "Journal" in str(focused.label), str(focused.label)
