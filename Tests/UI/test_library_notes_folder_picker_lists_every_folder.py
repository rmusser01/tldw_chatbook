"""The Notes folder pickers list every folder; a moved-into folder lists its notes.

TASK-34000.28 (N-23, N-28), review notes-library-ux-review-2026-10-02,
verified on ``bc64fb2272``:

- N-23: on a fresh visit with every folder collapsed, Move note offered only
  the top-level folders -- "Projects / Thesis" and "Study / Japanese" were
  missing until their parents were expanded -- and the dialog was titled
  only "Move note". The picker was a ``Select`` over the branches the tree
  had loaded this visit.
- N-28: moving a folder into a collapsed folder opened the destination with
  the moved folder but none of the destination's own notes; collapsing and
  re-expanding it showed them. The locator that expanded it loaded only the
  folders slice it walked.

Two boots of the real ``LibraryScreen`` with the production stylesheet over
a real ChaChaNotes database whose folders are created through
``LocalNoteFolderRepository``; every folder is collapsed on arrival. Every
claim is a real key press or click and what the user then sees (painted
text), plus the database read back through a second connection.

N-24 (TASK-34000.28 AC#5, a later slice of this same task) is still open
here: a folder mutation fails intermittently with "That folder changed
elsewhere — refresh and try again." and writes nothing. Measured while
building this file, on this harness AND on a full ``TldwCli`` boot, and on
the base commit with its old ``Select`` picker (2 and 3 of 10 base moves):
the repository's write raises SQLITE_BUSY_SNAPSHOT (extended code 517) with
no other write to the database in the move's window. These arms pin the
picker and the tree after a move, not N-24, so a choice that ends in exactly
that notice is made again -- the notice's own "try again" -- at most
``ATTEMPTS`` times, every attempt recorded and asserted. Once N-24 is fixed
``ATTEMPTS`` drops to 1.
"""

from __future__ import annotations

import sqlite3

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import SIZE, _NotesProfile, _open_notes, _until
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteFolderTargetDialog,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: A behaviour wait: long enough for a slice load under CI contention, short
#: enough that a RED arm names its reason in seconds.
WAIT = 10.0

#: Every folder the seed creates, as the picker labels them, in path order.
ALL_FOLDERS = [
    "Projects",
    "Projects / App launch",
    "Projects / Thesis",
    "Study",
    "Study / Japanese",
]

#: Notes filed in Study before the folder move (N-28's "its own notes").
STUDY_NOTES = ("Kanji drills", "Grammar cheat sheet")

#: N-24's false conflict notice, and how many times a choice is made.
N24_NOTICE = "That folder changed elsewhere — refresh and try again."
ATTEMPTS = 3


class _Tree:
    """``_NotesProfile``'s two unfiled notes plus the review's nested folders.

    One more note lives in ``Projects / Thesis`` and two in ``Study``.
    """

    def __init__(self, tmp_path) -> None:
        self.profile = _NotesProfile(tmp_path)
        repository = LocalNoteFolderRepository(self.profile.db)
        projects = repository.create_folder(name="Projects", parent_id=None)
        thesis = repository.create_folder(name="Thesis", parent_id=projects.folder_id)
        launch = repository.create_folder(
            name="App launch", parent_id=projects.folder_id
        )
        study = repository.create_folder(name="Study", parent_id=None)
        japanese = repository.create_folder(name="Japanese", parent_id=study.folder_id)
        outline = self.profile.db.add_note("Thesis outline", "- chapter one")
        repository.attach_manual(folder_id=thesis.folder_id, note_id=outline)
        for title in STUDY_NOTES:
            note_id = self.profile.db.add_note(title, f"{title} body")
            repository.attach_manual(folder_id=study.folder_id, note_id=note_id)
        self.ids = {
            folder.path: folder.folder_id
            for folder in (projects, thesis, launch, study, japanese)
        }


def _mounted(tmp_path) -> tuple[LibraryProductionCSSHarness, _Tree]:
    """The real Library hierarchy over the seeded tree, production CSS."""
    app = _build_test_app()
    _seed_conversations(app, [])
    tree = _Tree(tmp_path)
    app.chachanotes_db = tree.profile.db
    app.notes_scope_service = tree.profile.scope_service
    app.notes_service = tree.profile.interop
    return LibraryProductionCSSHarness(app, screen=LibraryScreen(app)), tree


def _memberships(tmp_path) -> list[tuple[str, str]]:
    """Every live ``(note_id, folder_id)`` placement, read through a second
    connection to the test's own database file -- what a relaunch would see.
    """
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        return sorted(
            connection.execute(
                "SELECT note_id, folder_id FROM note_folder_memberships "
                "WHERE deleted = 0"
            ).fetchall()
        )


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


def _painted_list(host, screen) -> str:
    """The Notes list's painted rows only (the tree, not the editor or rail)."""
    strips = list(host.screen._compositor.render_strips())
    lists = list(screen.query("#library-notes-list"))
    if not lists:
        return ""
    region = lists[0].region
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, min(region.bottom, len(strips)))
    )


def _picker_rows(host) -> list[str]:
    return [
        str(row.label)
        for row in host.screen.query(".library-note-folder-target-row").results(Button)
    ]


async def _wait(pilot, predicate, what) -> None:
    """``_until`` with a message that may be a callable (read at raise time)."""
    try:
        await _until(pilot, predicate, "", timeout=WAIT)
    except AssertionError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


def _record_mutations(screen, monkeypatch) -> list[tuple[str, dict, bool, str]]:
    """Record every folder mutation the screen runs, passing each through.

    Not a fake: the real ``_execute_library_notes_tree_mutation`` runs
    (service, repository, reconcile, locator); the record is what it was
    asked for -- the picker's answer -- and whether it committed.
    """
    outcomes: list[tuple[str, dict, bool, str]] = []
    real = screen._execute_library_notes_tree_mutation

    async def recorded(operation, **payload):
        ok = await real(operation, **payload)
        asked = {key: value for key, value in payload.items() if key[0] != "_"}
        outcomes.append((operation, asked, ok, screen._notes_state.notice))
        return ok

    monkeypatch.setattr(screen, "_execute_library_notes_tree_mutation", recorded)
    return outcomes


async def _notes_collapsed(host, pilot) -> LibraryScreen:
    """Boot into Notes; wait until the root folders paint, every one collapsed."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _open_notes(screen, pilot)

    def settled() -> bool:
        folders = screen._notes_state.tree_branches.get(NotesBranchKey(None, "folders"))
        return (
            folders is not None
            and folders.freshness == "fresh"
            and not folders.loading
            and "▸ Projects" in _painted(host)
            and "▸ Study" in _painted(host)
        )

    await _wait(pilot, settled, "the collapsed root folder rows")
    assert not screen._notes_state.tree_expanded_ids
    return screen


async def _leave_text_fields(screen, pilot) -> None:
    """Tab until a non-text control holds focus (a printable key would type)."""
    for _ in range(12):
        if screen.focused is not None and not isinstance(
            screen.focused, (Input, TextArea)
        ):
            return
        await pilot.press("tab")
        await pilot.pause()
    raise AssertionError(f"no non-text control took focus ({screen.focused!r})")


async def _open_picker(host, pilot, button_id: str) -> LibraryNoteFolderTargetDialog:
    await pilot.click(button_id)
    await _wait(
        pilot,
        lambda: isinstance(host.screen, LibraryNoteFolderTargetDialog),
        f"{button_id} to open the folder picker",
    )
    picker = host.screen
    await _wait(
        pilot,
        lambda: bool(_picker_rows(host)),
        lambda: f"the picker's first page: {_painted(host)}",
    )
    assert picker.focused is picker.query_one(
        "#library-note-folder-target-filter", Input
    )
    return picker


async def _choose(
    host, pilot, screen, outcomes, *, button_id: str, query: str, rows, total: str
) -> tuple[str, dict, bool, str]:
    """Open the picker, type ``query``, wait for ``rows`` and ``total``, Down + Enter.

    Returns the mutation the choice ran; repeats only on N-24's notice.
    """
    for _attempt in range(ATTEMPTS):
        ran = len(outcomes)
        await _open_picker(host, pilot, button_id)
        await pilot.press(*query)
        await _wait(
            pilot,
            lambda: _picker_rows(host) == rows and total in _painted(host),
            lambda: f"the list narrowed to {rows} ({total}): {_picker_rows(host)} "
            f"{_painted(host)}",
        )
        await pilot.press("down", "enter")
        await _wait(pilot, lambda: host.screen is screen, "the picker to choose")
        await _wait(pilot, lambda: len(outcomes) > ran, "the chosen mutation to run")
        if outcomes[-1][2] or outcomes[-1][3] != N24_NOTICE:
            return outcomes[-1]
    raise AssertionError(f"N-24's false conflict {ATTEMPTS} times running: {outcomes}")


async def test_move_note_picker_offers_nested_folders_on_a_fresh_visit_and_narrows_by_typing(
    tmp_path, monkeypatch
) -> None:
    """AC#4: every active folder at any depth, a named title, type-to-filter."""
    host, tree = _mounted(tmp_path)
    note_id = tree.profile.note_id
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_collapsed(host, pilot)
        outcomes = _record_mutations(screen, monkeypatch)
        await _leave_text_fields(screen, pilot)

        # g, then Down to the unfiled "Ideas inbox"; Enter opens it, which
        # selects its placement for the list's Move note action.
        await pilot.press("g")
        for _ in range(8):
            await pilot.pause()
            if getattr(screen.focused, "note_id", "") == note_id:
                break
            await pilot.press("down")
        assert getattr(screen.focused, "note_id", "") == note_id, screen.focused
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: screen._library_note_session.snapshot is not None
            and screen._library_note_session.snapshot.note_id == note_id,
            "the note to open",
        )
        before = _memberships(tmp_path)

        await _open_picker(host, pilot, "#library-notes-placement-move")
        await _wait(
            pilot,
            lambda: _picker_rows(host) == ALL_FOLDERS
            and "5 folders" in _painted(host),
            lambda: f"every folder with its total: {_picker_rows(host)} "
            f"{_painted(host)}",
        )
        painted = _painted(host)
        assert 'Move "Ideas inbox" to…' in painted, painted
        # Nested under COLLAPSED parents the tree never loaded.
        for path in ("Projects / Thesis", "Projects / App launch", "Study / Japanese"):
            assert path in painted, f"{path!r} is not painted: {painted}"
        assert not screen._notes_state.tree_expanded_ids

        # Review focus: Escape asks for no mutation at all, and every
        # placement in the database is exactly as it was.
        await pilot.press("escape")
        await _wait(pilot, lambda: host.screen is screen, "the picker to close")
        await pilot.pause(0.3)
        assert outcomes == []
        assert _memberships(tmp_path) == before

        # Add to folder reads the same all-folders page and names the note
        # (fix round 1: the title and the nested rows had no pin).
        await _open_picker(host, pilot, "#library-notes-placement-add")
        await _wait(
            pilot,
            lambda: _picker_rows(host) == ALL_FOLDERS
            and "5 folders" in _painted(host),
            lambda: f"Add to folder's rows: {_picker_rows(host)} {_painted(host)}",
        )
        assert 'Add "Ideas inbox" to folder…' in _painted(host), _painted(host)
        await pilot.press("escape")
        await _wait(pilot, lambda: host.screen is screen, "Add to folder to close")
        await pilot.pause(0.3)
        assert outcomes == []
        assert _memberships(tmp_path) == before

        operation, asked, ok, notice = await _choose(
            host,
            pilot,
            screen,
            outcomes,
            button_id="#library-notes-placement-move",
            query="thes",
            rows=["Projects / Thesis"],
            total="1 folder",
        )
        assert (operation, asked["destination_folder_id"], ok) == (
            "move_placement",
            tree.ids["/Projects/Thesis"],
            True,
        ), (outcomes, notice)
        # Every attempt asked for Thesis: no attempt chose anything else.
        assert {attempt[1]["destination_folder_id"] for attempt in outcomes} == {
            tree.ids["/Projects/Thesis"]
        }
        # Exactly one placement added -- the note in Thesis -- and nothing else
        # written (the move's source is Unfiled, so nothing is detached).
        assert _memberships(tmp_path) == sorted(
            [*before, (note_id, tree.ids["/Projects/Thesis"])]
        )


async def test_moving_a_folder_into_a_collapsed_folder_shows_the_destinations_notes(
    tmp_path, monkeypatch
) -> None:
    """AC#7: the destination lists its own notes AND the moved folder, unasked."""
    host, tree = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_collapsed(host, pilot)
        outcomes = _record_mutations(screen, monkeypatch)

        projects = next(
            row
            for row in screen.query(".library-notes-folder-row").results(Button)
            if getattr(row, "folder_id", "") == tree.ids["/Projects"]
        )
        await pilot.click(projects)
        await _wait(
            pilot,
            lambda: bool(screen.query("#library-notes-folder-move")),
            "the folder toolbar's Move folder",
        )
        await _open_picker(host, pilot, "#library-notes-folder-move")
        # A folder cannot move into itself or beneath itself.
        await _wait(
            pilot,
            lambda: _picker_rows(host) == ["Top level", "Study", "Study / Japanese"]
            and "2 folders" in _painted(host),
            lambda: f"Projects' subtree left out: {_picker_rows(host)}",
        )
        assert 'Move folder "Projects" to…' in _painted(host)
        await pilot.press("escape")
        await _wait(pilot, lambda: host.screen is screen, "the picker to close")
        assert outcomes == []

        operation, asked, ok, notice = await _choose(
            host,
            pilot,
            screen,
            outcomes,
            button_id="#library-notes-folder-move",
            query="study",
            rows=["Study", "Study / Japanese"],
            total="2 folders",
        )
        assert (operation, asked["parent_id"], ok) == (
            "move_folder",
            tree.ids["/Study"],
            True,
        ), (outcomes, notice)
        # Every attempt asked for Study: no attempt chose anything else.
        assert {attempt[1]["parent_id"] for attempt in outcomes} == {tree.ids["/Study"]}

        def destination_lists_its_notes() -> bool:
            painted = _painted_list(host, screen)
            return "▾ Study" in painted and all(
                title in painted for title in STUDY_NOTES
            )

        await _wait(
            pilot,
            destination_lists_its_notes,
            lambda: "Study's own notes under ▾ Study after the move: "
            f"{_painted_list(host, screen)}",
        )
        painted = _painted_list(host, screen)
        study = painted.index("▾ Study")
        assert painted.index("Projects", study) > study, painted
        for title in STUDY_NOTES:
            assert painted.index(title) > study, painted
