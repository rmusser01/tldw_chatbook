"""The Notes tree is keyboard-complete (TASK-34000.28: N-22, N-30, N-32).

Review notes-library-ux-review-2026-10-02, verified on ``bc64fb2272``:

- N-22: ``g`` focused ``▸ Agent_Lessons``; Down and Right then changed
  nothing, and the first note was about 17 Tab stops away. Folder rows and
  the tree's "more" pager were missing from the one shared row-walk table.
- N-30: Enter on **Sort: Newest** opened the ``✓ Newest  Oldest  Title``
  strip but left focus where it was, while the footer read "enter choose
  sort".
- N-32: F6 in the note body SELECTED the current line (Textual's own
  ``TextArea`` f6 binding) instead of leaving the editor.

Three boots of the real ``LibraryScreen`` with the production stylesheet over
a real ChaChaNotes database holding nested folders and a paged Unfiled group
(about 20 s locally, 2026-10-24). Every claim is a real keypress and what the
user then sees: painted text, the focused control, ``Button.label``, and the
database unchanged. The F6 arm (N-32) boots the real ``TldwCli`` and lives in
``test_library_notes_tree_keys_extended.py``, which the full ``Tests/UI`` run
covers -- one more real-app boot put this file past the PR lane's 30 s budget.

Fix round 1 (2026-10-25) rides the same three boots: Enter on the pager hands
focus to the first row it added (M2); Right/Left toggle a folder without
selecting it (I-3); Left/Right walk the open Sort strip (I-2); and a re-page
held until the strip has closed still leaves focus on Sort (I-4). One
boot-free arm pins M2's other half -- a page that added no row keeps focus
on its pager -- which no seeded tree reaches by keys.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _NotesProfile,
    _open_notes,
    _until,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackRecoveryResult,
)
from tldw_chatbook.Library.library_notes_tree_paging import (
    LIBRARY_NOTES_TREE_PAGE_SIZE,
    NotesBranchKey,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The three Library list-row kinds the Notes tree composes as siblings.
ROW_CLASSES = ("library-notes-folder-row", "library-notes-row", "library-notes-tree-pager")

#: A behaviour wait: long enough for a slice load under CI contention, short
#: enough that a RED arm names its reason in seconds, not half a minute.
WAIT = 10.0


async def _wait(pilot, predicate, what, timeout: float = WAIT) -> None:
    """``_until`` with a message that may be a callable (read at raise time)."""
    try:
        await _until(pilot, predicate, "", timeout=timeout)
    except AssertionError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


@pytest.fixture(autouse=True)
def _isolate_actor_pack_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unrelated startup recovery off this test's database."""

    monkeypatch.setattr(
        "tldw_chatbook.app.first_profile_created_this_session", lambda: False
    )
    monkeypatch.setattr(
        PersonaActorPackCoordinator,
        "recover",
        lambda _self: PersonaActorPackRecoveryResult(0, 0, 0, ()),
    )


class _Tree:
    """``Journal``, ``Projects / Thesis`` (holding one note) and a paged Unfiled.

    ``_NotesProfile`` seeds two unfiled notes; ``LIBRARY_NOTES_TREE_PAGE_SIZE``
    more make the Unfiled group one page plus a "more" pager, so the walk has
    every row kind to reach.
    """

    def __init__(self, tmp_path) -> None:
        self.profile = _NotesProfile(tmp_path)
        repository = LocalNoteFolderRepository(self.profile.db)
        self.journal_id = repository.create_folder(name="Journal", parent_id=None).folder_id
        self.projects_id = repository.create_folder(
            name="Projects", parent_id=None
        ).folder_id
        self.thesis_id = repository.create_folder(
            name="Thesis", parent_id=self.projects_id
        ).folder_id
        repository.attach_manual(folder_id=self.thesis_id, note_id=self.profile.note_id)
        for index in range(LIBRARY_NOTES_TREE_PAGE_SIZE):
            assert self.profile.db.add_note(f"Loose {index:02d}", f"body {index}")


def _mounted(tmp_path) -> tuple[LibraryProductionCSSHarness, _Tree]:
    """The real Library hierarchy over the seeded tree, production CSS."""
    app = _build_test_app()
    _seed_conversations(app, [])
    tree = _Tree(tmp_path)
    app.chachanotes_db = tree.profile.db
    app.notes_scope_service = tree.profile.scope_service
    app.notes_service = tree.profile.interop
    return LibraryProductionCSSHarness(app, screen=LibraryScreen(app)), tree


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


def _folder_id(widget) -> str:
    return str(getattr(widget, "folder_id", "") or "")


def _data_shape(tmp_path, note_id: str) -> tuple:
    """Row counts of the three tables a tree key could touch, plus one body.

    Read through a second connection to the test's own database file, so
    the claim is about what a relaunch would find, not widget state.
    """
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("notes", "note_folders", "note_folder_memberships")
        )
        content = connection.execute(
            "SELECT content FROM notes WHERE id = ?", (note_id,)
        ).fetchone()[0]
    return counts + (content,)


def _list_rows(screen) -> list[Button]:
    """The tree's rows in DOM (= visual) order, every kind included."""
    return [
        row
        for row in screen.query("#library-notes-list > Button").results(Button)
        if any(row.has_class(name) for name in ROW_CLASSES)
    ]


def _tree_selection(screen) -> tuple:
    """What a folder action would act on, as the user sees it.

    The selected placement, the rows painted as selected, and the folder
    toolbar's buttons (Rename / Move / Remove appear only for a selected
    folder) -- navigation alone must leave all three alone.
    """
    return (
        screen._notes_state.tree_selected_placement_id,
        tuple(row.id for row in screen.query("#library-notes-list > .is-selected")),
        tuple(
            (button.id, str(button.label))
            for button in screen.query(".library-notes-tree-action-row Button").results(
                Button
            )
        ),
    )


def _hold_root_pages(tree, monkeypatch) -> tuple[asyncio.Event, list[str]]:
    """Gate the root folder and placement pages on an ``asyncio.Event``.

    The event starts set (pages pass straight through); clearing it holds
    every ROOT page request until it is set again, which is how the sort
    test lands a re-page after the strip has closed, deterministically
    (the reviewer's probe B, fix round 1 I-4).
    """
    gate = asyncio.Event()
    gate.set()
    held: list[str] = []
    service = tree.profile.scope_service
    for name in ("page_note_placements", "page_note_folder_children"):
        real = getattr(service, name)

        def held_page(*, _real=real, _name=name, **kwargs):
            async def page():
                if kwargs.get("parent_id") is None and not gate.is_set():
                    held.append(_name)
                    await gate.wait()
                return await _real(**kwargs)

            return page()

        monkeypatch.setattr(service, name, held_page)
    return gate, held


def _focused_id(screen) -> str | None:
    return getattr(screen.focused, "id", None)


async def _notes_tree(host, pilot) -> LibraryScreen:
    """Boot into Notes and wait for the root slices and the pager to settle."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _open_notes(screen, pilot)

    def settled() -> bool:
        branches = screen._notes_state.tree_branches
        folders = branches.get(NotesBranchKey(None, "folders"))
        placements = branches.get(NotesBranchKey(None, "placements"))
        if folders is None or folders.loading or folders.freshness != "fresh":
            return False
        if placements is None or placements.loading:
            return False
        pagers = list(screen.query(".library-notes-tree-pager").results(Button))
        return bool(pagers) and not any(pager.disabled for pager in pagers)

    await _wait(pilot, settled, "the root slices and the Unfiled pager")
    await _wait(
        pilot,
        lambda: "▸ Projects" in _painted(host) and "▾ Unfiled" in _painted(host),
        "the folder rows to paint",
    )
    return screen


async def _leave_text_fields(screen, pilot) -> None:
    """Tab until a non-text control holds focus.

    Entering Notes parks focus on the first note row once the list has
    loaded, or on the filter while it is still empty (the entry-focus arm's
    own fallback); a printable key in the filter would type, so the walks
    start from a button either way.
    """
    for _ in range(12):
        if screen.focused is not None and not isinstance(
            screen.focused, (Input, TextArea)
        ):
            return
        await pilot.press("tab")
        await pilot.pause()
    raise AssertionError(f"no non-text control took focus (focused={screen.focused!r})")


async def _press_g(screen, pilot) -> None:
    """``g`` from a non-text control: the existing go-to-folder gate."""
    await _leave_text_fields(screen, pilot)
    await pilot.press("g")
    await _wait(
        pilot,
        lambda: screen.focused is not None
        and screen.focused.has_class("library-notes-folder-row"),
        lambda: f"g to focus a folder row (focused={screen.focused!r})",
    )


async def test_down_from_a_folder_row_reaches_the_note_rows_and_the_pager(
    tmp_path,
) -> None:
    """AC#3: Up/Down walk folder rows, note rows and the pager in visual order.

    Base: after ``g`` focus is on the first folder row and Down leaves it
    there -- the walker refuses a folder row -- so the visited ids are the
    same folder row repeated.
    """
    host, _tree = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_tree(host, pilot)
        await _press_g(screen, pilot)
        rows = _list_rows(screen)
        expected = [row.id for row in rows]
        assert screen.focused is rows[0], (screen.focused, expected)
        assert any(row.has_class("library-notes-tree-pager") for row in rows), expected

        visited: list[str | None] = []
        for _ in range(len(expected) - 1):
            await pilot.press("down")
            await pilot.pause()
            visited.append(screen.focused.id if screen.focused else None)
        assert visited == expected[1:], f"Down visited {visited}; the tree is {expected}"
        assert screen.focused.has_class("library-notes-tree-pager")

        # The list end: Down is claimed but does not wrap or leave the list.
        await pilot.press("down")
        await pilot.pause()
        assert screen.focused is rows[-1]
        # Up climbs back onto the last note row.
        await pilot.press("up")
        await pilot.pause()
        assert screen.focused is rows[-2], screen.focused

        # Fix round 1 (M2): Enter on the pager loads the next page and hands
        # focus to the first row that page ADDED. It used to leave focus in
        # the filter, where the next printable key typed into it. Base: the
        # loading recompose pruned the focused pager, Textual's reset landed
        # on the grip, and the screen counted that reset as the user's own
        # focus move -- which vetoed the pager's hand-off.
        await pilot.press("down")
        await pilot.pause()
        assert screen.focused is rows[-1], screen.focused
        before_page = {
            getattr(row, "placement_id", "")
            for row in screen.query(".library-notes-tree-note-row")
        }
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: len(screen.query(".library-notes-tree-note-row")) > len(before_page),
            "Enter on the pager to load the next page",
        )
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused.has_class("library-notes-tree-note-row")
            and getattr(screen.focused, "placement_id", "") not in before_page,
            lambda: f"focus to land on the first added row (focused={screen.focused!r})",
        )
        added = screen.focused
        await pilot.pause(0.3)
        assert screen.focused is added, screen.focused
        assert not isinstance(screen.focused, (Input, TextArea))
        # A real key still works from there: Up reaches the row above it.
        await pilot.press("up")
        await pilot.pause()
        assert screen.focused is not added
        assert screen.focused.has_class("library-notes-tree-note-row"), screen.focused
        # Nothing was opened or changed by the walk: still the list, no note.
        assert screen._notes_state.view == "list"
        assert not screen.query("#library-note-body")


async def test_right_expands_and_left_collapses_then_climbs(tmp_path) -> None:
    """AC#3: Right expands a folder; Left collapses it or climbs to its parent.

    Base: Right and Left do nothing on a focused folder row (Textual's
    Button ignores them), so Projects never expands.
    """
    host, tree = _mounted(tmp_path)
    before_keys = _data_shape(tmp_path, tree.profile.note_id)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_tree(host, pilot)
        await _press_g(screen, pilot)
        while _folder_id(screen.focused) != tree.projects_id:
            before = screen.focused
            await pilot.press("down")
            await pilot.pause()
            assert screen.focused is not before, f"Down stalled on {before!r}"
        expanded = screen._notes_state.tree_expanded_ids
        assert tree.projects_id not in expanded
        # Fix round 1 (I-3): Right/Left only toggle. Selecting a folder is
        # Enter's or a click's job -- it retargets the folder toolbar's
        # Rename / Move / Remove onto that folder -- so navigation alone must
        # leave the selection and the toolbar exactly as they were. Base:
        # Right pressed the row, which selected Projects.
        selection = _tree_selection(screen)
        selection_paint = _painted(host)
        assert screen.focused.placement_id != selection[0], selection

        await pilot.press("right")
        await _wait(
            pilot,
            lambda: tree.projects_id in expanded,
            "Right to expand Projects",
        )
        await _wait(
            pilot,
            lambda: "▾ Projects" in _painted(host) and "▸ Thesis" in _painted(host),
            lambda: f"Projects to paint expanded with Thesis beneath:\n{_painted(host)}",
        )
        # Focus survives the recompose on the same folder.
        await _wait(
            pilot,
            lambda: _folder_id(screen.focused) == tree.projects_id,
            lambda: f"focus to stay on Projects (focused={screen.focused!r})",
        )
        assert _tree_selection(screen) == selection
        # Right on an already-expanded folder changes nothing.
        await pilot.press("right")
        await pilot.pause()
        assert tree.projects_id in expanded
        assert _folder_id(screen.focused) == tree.projects_id

        await pilot.press("down")
        await _wait(
            pilot,
            lambda: _folder_id(screen.focused) == tree.thesis_id,
            lambda: f"Down to land on Thesis (focused={screen.focused!r})",
        )
        # Left on a collapsed folder climbs to its parent and toggles nothing.
        await pilot.press("left")
        await _wait(
            pilot,
            lambda: _folder_id(screen.focused) == tree.projects_id,
            lambda: f"Left to climb to Projects (focused={screen.focused!r})",
        )
        assert tree.projects_id in expanded
        assert tree.thesis_id not in expanded
        assert "▸ Thesis" in _painted(host)
        assert _tree_selection(screen) == selection

        # Left on the expanded parent collapses it.
        await pilot.press("left")
        await _wait(
            pilot,
            lambda: tree.projects_id not in expanded,
            "Left to collapse Projects",
        )
        await _wait(
            pilot,
            lambda: "▸ Projects" in _painted(host),
            lambda: f"Projects to paint collapsed:\n{_painted(host)}",
        )
        assert "Thesis" not in _painted(host)
        await _wait(
            pilot,
            lambda: _folder_id(screen.focused) == tree.projects_id,
            lambda: f"focus to stay on Projects (focused={screen.focused!r})",
        )
        assert _tree_selection(screen) == selection
        # ...and what the user reads agrees: the folder toolbar did not grow
        # Rename / Move / Remove for a folder nobody selected.
        assert ("Rename" in _painted(host)) == ("Rename" in selection_paint)
        # A real key still works after the recomposes (lessons-textual: a
        # recompose can orphan ``app.focused``): Down reaches Unfiled.
        await pilot.press("down")
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.has_class("library-notes-unfiled-row"), screen.focused
        # Nothing in the database moved: same notes, folders and placements.
        assert _data_shape(tmp_path, tree.profile.note_id) == before_keys


async def test_sort_chooser_takes_focus_on_the_active_choice_and_returns_it(
    tmp_path, monkeypatch
) -> None:
    """AC#8: Enter on Sort focuses the ``✓`` choice; arrows and Enter choose.

    Base: the strip opens and the footer reads "enter choose sort", but focus
    stays on the opener (or the filter), so Enter never chooses a sort.

    Fix round 1: Left/Right walk the open strip without choosing (I-2), and
    the re-page a changed sort requests is HELD until focus is back on Sort,
    then released -- the live walk's race, deterministically (I-4).

    The same boot ends with the Right/Left negative control in the filter
    (one boot fewer in the PR lane).
    """
    host, tree = _mounted(tmp_path)
    gate, held = _hold_root_pages(tree, monkeypatch)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_tree(host, pilot)
        await _leave_text_fields(screen, pilot)
        # The review's route: "/" into the filter, then Tab to Sort.
        await pilot.press("/")
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused.id == "library-notes-filter",
            "/ to focus the filter",
        )
        visited: list[str | None] = []
        for _ in range(12):
            if screen.focused is not None and screen.focused.id == "library-notes-sort":
                break
            await pilot.press("tab")
            await pilot.pause()
            visited.append(screen.focused.id if screen.focused else None)
        assert screen.focused.id == "library-notes-sort", visited
        assert str(screen.focused.label) == "Sort: Newest"

        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: screen._notes_state.sort_choices_visible,
            "the sort strip to open",
        )
        await _wait(
            pilot,
            lambda: "choose sort" in _painted(host),
            lambda: f"the footer to advertise the chooser:\n{_painted(host)}",
        )
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused.id == "library-notes-sort-newest",
            lambda: f"focus to land on the active choice (focused={screen.focused!r})",
        )
        assert str(screen.focused.label).startswith("✓ Newest"), screen.focused.label

        # Fix round 1 (I-2): the open strip is one row of choices. Left and
        # Right move between them -- no wrap at either end, and moving never
        # chooses (Enter does). Base: every arrow left focus where it was.
        for key, expected in (
            ("left", "library-notes-sort-newest"),
            ("right", "library-notes-sort-oldest"),
            ("right", "library-notes-sort-title"),
            ("right", "library-notes-sort-title"),
            ("left", "library-notes-sort-oldest"),
        ):
            await pilot.press(key)
            await _wait(
                pilot,
                lambda: _focused_id(screen) == expected,
                lambda: f"{key} to focus {expected} (focused={screen.focused!r})",
            )
        assert screen._notes_state.sort == "newest"
        assert screen._notes_state.sort_choices_visible
        assert "✓ Newest" in _painted(host)

        # Fix round 1 (I-4): choosing re-pages the tree. Hold that root page
        # until the closed strip has handed focus back to Sort, then let it
        # land: the sync it triggers replays the focus it finds, and Sort has
        # to survive that. Base: no portable role for Sort, so the replay
        # fell back to the filter (the live walk's capture 07, first round).
        gate.clear()
        await pilot.press("enter")
        await _wait(pilot, lambda: bool(held), "the re-page to be held")
        await _wait(
            pilot,
            lambda: screen._notes_state.sort == "oldest"
            and not screen._notes_state.sort_choices_visible,
            "Enter to choose Oldest and close the strip",
        )
        await _wait(
            pilot,
            lambda: _focused_id(screen) == "library-notes-sort",
            lambda: f"focus to return to Sort before the page lands ({screen.focused!r})",
        )
        gate.set()
        await _wait(
            pilot,
            lambda: not any(
                branch.loading for branch in screen._notes_state.tree_branches.values()
            ),
            "the held pages to land",
        )
        await pilot.pause(0.3)
        assert _focused_id(screen) == "library-notes-sort", screen.focused
        await _wait(
            pilot,
            lambda: "Sort: Oldest" in _painted(host),
            lambda: f"the Sort control to read Oldest:\n{_painted(host)}",
        )
        assert "choose sort" not in _painted(host)
        # Escape from the reopened strip also returns focus to Sort.
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused.id == "library-notes-sort-oldest",
            lambda: f"the reopened strip to focus Oldest (focused={screen.focused!r})",
        )
        await pilot.press("escape")
        await _wait(
            pilot,
            lambda: not screen._notes_state.sort_choices_visible
            and screen.focused is not None
            and screen.focused.id == "library-notes-sort",
            lambda: f"Escape to return focus to Sort (focused={screen.focused!r})",
        )
        # The seam the live walk exposed (160x45, first round): a slice page
        # of the re-page landed while a choice still had focus, and the
        # identity it replayed -- ``sort-choice:oldest`` -- named a button
        # the closed strip no longer had, so the navigator fallback put focus
        # in the filter. A vanished choice resolves to its opener.
        sort_button = screen.focused
        stale_choice = dataclasses.replace(
            screen._capture_library_notes_focus_identity(),
            semantic_role="sort-choice:oldest",
        )
        assert screen._library_notes_role_target(stale_choice) is sort_button

        # Negative control (green before and after this task): the Right/Left
        # branch is gated to a focused FOLDER row, and the screen's ``on_key``
        # returns before it for any Input or TextArea -- so in the filter
        # Left and Right are caret moves and type normally.
        await pilot.press("/")
        await _wait(
            pilot,
            lambda: isinstance(screen.focused, Input)
            and screen.focused.id == "library-notes-filter",
            "/ to focus the filter",
        )
        filter_input = screen.focused
        await pilot.press("a", "b", "left", "c")
        await _wait(
            pilot,
            lambda: filter_input.value == "acb",
            lambda: f"the filter to read 'acb' (value={filter_input.value!r})",
        )
        await pilot.press("right", "d")
        await _wait(
            pilot,
            lambda: filter_input.value == "acbd",
            lambda: f"the filter to read 'acbd' (value={filter_input.value!r})",
        )
        assert screen.focused is filter_input


async def test_a_page_that_added_no_row_keeps_focus_on_its_pager(monkeypatch) -> None:
    """Fix round 1 (M2): nothing added -> focus the pager itself, by id.

    The other half of the pager hand-off. While the loading recompose has
    the pager pruned, Textual parks focus on a neighbour, so the identity the
    page sync captured names THAT neighbour; restoring it would leave focus
    where Textual put it (live: the filter). The hand-off therefore focuses
    the still-listed pager directly. No seeded tree reaches this by keys (a
    "more" page that adds nothing also ends the paging), so the branch is
    driven with the screen's own state shapes, like its sibling pins in
    ``test_library_notes_folder_navigator.py``. Base: the branch returned
    ``True`` (restore the captured identity) and focused nothing.
    """
    key = NotesBranchKey(None, "placements")
    pager_id = "library-notes-tree-pager-root-placements-more"
    focused: list[str] = []
    pager = SimpleNamespace(
        id=pager_id, focusable=True, focus=lambda: focused.append(pager_id)
    )
    row = SimpleNamespace(placement_id="a", focus=lambda: focused.append("a"))
    fake = SimpleNamespace(
        _notes_state=SimpleNamespace(
            tree_branches={key: SimpleNamespace(generation=2, item_ids=("a", "b"))},
            tree_lifecycle_generation=1,
            tree_topology_epoch=1,
            tree_request_generations={key: 2},
            focus_intent_generation=5,
        ),
        query=lambda selector: (pager,) if "pager" in selector else (row,),
    )
    monkeypatch.setattr(
        LibraryScreen,
        "_build_library_notes_tree_projection",
        lambda _self: SimpleNamespace(
            rows=(SimpleNamespace(kind="pager", focus_id=pager_id),)
        ),
    )

    result = LibraryScreen._focus_library_notes_tree_after_page(
        fake,
        key,
        pager_focus_id=pager_id,
        prior_item_ids=("a", "b"),
        request_generation=2,
        topology_epoch=1,
        lifecycle_generation=1,
        focus_generation=5,
    )

    assert focused == [pager_id]
    assert result is False  # handled: the captured identity is not replayed
