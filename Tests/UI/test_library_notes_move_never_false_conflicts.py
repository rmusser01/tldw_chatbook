"""A genuine folder conflict names itself and offers Refresh; a delete
leaves nothing selected for Move note.

TASK-34000.28 (N-24), review notes-library-ux-review-2026-10-02: the verifier
saw 4 of 7 Move note operations fail with "That folder changed elsewhere —
refresh and try aga…" (truncated, no Refresh control) and nothing written,
and after a delete Move note and Add to folder stayed enabled.

The real ``LibraryScreen`` with the production stylesheet over a real
ChaChaNotes database (9 folders, 30 notes, 10 of them unfiled) whose folders
and placements are created through ``LocalNoteFolderRepository``. Every
claim is a real press, typed keys in the folder picker, and what the user
then sees (painted text), plus the database read back through a second
connection. The ten-move run (AC#5) is in ``..._extended.py`` beside this
file: about 3 s a move does not fit this lane.
"""

from __future__ import annotations

import sqlite3
import time

import pytest
from textual.widgets import Button

import tldw_chatbook.Notes.note_folder_repository as folder_repository_module
from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import SIZE, _NotesProfile, _open_notes, _until
from Tests.UI.test_library_shell import (
    _FakeSkillsScopeService,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_notes_state import LibraryNotesFocusIdentity
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

#: The success notice every move must end in.
UPDATED = "Folder organization updated."

#: The review's nine folders (path -> parent path), created in this order.
FOLDERS = (
    ("Journal", None),
    ("Meetings", None),
    ("Projects", None),
    ("App launch", "Projects"),
    ("Thesis", "Projects"),
    ("Recipes", None),
    ("Research", None),
    ("Study", None),
    ("Japanese", "Study"),
)

#: Two notes in each folder, three in Journal and in Meetings (20).
FILED = {
    "Journal": ("Journal day one", "Journal day two", "Journal day three"),
    "Meetings": ("Standup notes", "Retro notes", "Planning notes"),
    "Projects": ("Projects roadmap", "Projects budget"),
    "App launch": ("Launch checklist", "Launch press kit"),
    "Thesis": ("Thesis outline", "Thesis sources"),
    "Recipes": ("Pancakes", "Ramen"),
    "Research": ("Paper notes", "Reading queue"),
    "Study": ("Kanji drills", "Grammar sheet"),
    "Japanese": ("Hiragana", "Katakana"),
}

#: Eight more unfiled notes beside ``_NotesProfile``'s two (10 unfiled).
UNFILED = tuple(f"Loose note {index}" for index in range(1, 9))

#: Ten moves of ten different notes, Unfiled and foldered alternately:
#: ``(title, source folder or None, picker query, destination path)``. Each
#: foldered source is the previous move's destination, which the locator
#: has just expanded, so its rows are on screen without a further key.
MOVES = (
    ("Ideas inbox", None, "journ", "Journal"),
    ("Journal day one", "Journal", "meet", "Meetings"),
    ("Reading list", None, "recip", "Recipes"),
    ("Pancakes", "Recipes", "resea", "Research"),
    ("Loose note 1", None, "thes", "Projects / Thesis"),
    ("Thesis outline", "Thesis", "japan", "Study / Japanese"),
    ("Loose note 2", None, "launch", "Projects / App launch"),
    ("Launch checklist", "App launch", "journ", "Journal"),
    ("Loose note 3", None, "meet", "Meetings"),
    ("Standup notes", "Meetings", "recip", "Recipes"),
)


class _Profile:
    """The review's seeded power profile, cut to 9 folders and 30 notes."""

    def __init__(self, tmp_path) -> None:
        self.profile = _NotesProfile(tmp_path)
        db = self.profile.db
        repository = LocalNoteFolderRepository(db)
        self.folder_ids: dict[str, str] = {}
        for name, parent in FOLDERS:
            folder = repository.create_folder(
                name=name,
                parent_id=self.folder_ids[parent] if parent else None,
            )
            self.folder_ids[name] = folder.folder_id
        self.note_ids = {
            "Ideas inbox": self.profile.note_id,
            "Reading list": self.profile.other_id,
        }
        for title in UNFILED:
            self.note_ids[title] = db.add_note(title, f"{title} body")
        for folder, titles in FILED.items():
            for title in titles:
                note_id = db.add_note(title, f"{title} body")
                repository.attach_manual(
                    folder_id=self.folder_ids[folder], note_id=note_id
                )
                self.note_ids[title] = note_id
        assert len(self.note_ids) == 30


def _mounted(tmp_path) -> tuple[LibraryProductionCSSHarness, _Profile]:
    """The real Library hierarchy over the seeded profile, production CSS."""
    app = _build_test_app()
    # Notes owns this fixture; optional Skills filesystem startup is unrelated.
    app.skills_scope_service = _FakeSkillsScopeService(available=[])
    _seed_conversations(app, [])
    seeded = _Profile(tmp_path)
    app.chachanotes_db = seeded.profile.db
    app.notes_scope_service = seeded.profile.scope_service
    app.notes_service = seeded.profile.interop
    return LibraryProductionCSSHarness(app, screen=LibraryScreen(app)), seeded


def _placements(tmp_path) -> dict[str, set[str]]:
    """Every live note's folders, read through a second connection."""
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        rows = connection.execute(
            "SELECT note_id, folder_id FROM note_folder_memberships WHERE deleted = 0"
        ).fetchall()
    placements: dict[str, set[str]] = {}
    for note_id, folder_id in rows:
        placements.setdefault(note_id, set()).add(folder_id)
    return placements


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


def _picker_rows(host) -> list[str]:
    return [
        str(row.label)
        for row in host.screen.query(".library-note-folder-target-row").results(Button)
    ]


def _note_rows(screen) -> list[Button]:
    return list(screen.query(".library-notes-tree-note-row").results(Button))


async def _wait(pilot, predicate, what) -> None:
    """``_until`` with a message that may be a callable (read at raise time)."""
    try:
        await _until(pilot, predicate, "", timeout=WAIT)
    except AssertionError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


def _record_conflict_sites(monkeypatch) -> list[tuple[str, int | None, str]]:
    """Record every repository translation into ``FolderConflictError``.

    Wraps the three module-level translators the repository's mutations call
    (looked up by name at call time) and records the site plus the SQLite
    extended error code it saw, then re-raises exactly as before.
    """
    sites: list[tuple[str, int | None, str]] = []

    def sqlite_code(exc: BaseException) -> tuple[int | None, str]:
        pending: list[BaseException] = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, sqlite3.Error):
                return (
                    getattr(current, "sqlite_errorcode", None),
                    str(getattr(current, "sqlite_errorname", "")),
                )
            for link in (current.__cause__, current.__context__):
                if link is not None:
                    pending.append(link)
        return None, ""

    for name in (
        "_raise_mutation_operational_error",
        "_raise_wrapped_repository_error",
        "_raise_membership_integrity_error",
    ):
        real = getattr(folder_repository_module, name)

        def recorded(exc, *, _real=real, _name=name):
            code, code_name = sqlite_code(exc)
            sites.append((_name, code, code_name))
            return _real(exc)

        monkeypatch.setattr(folder_repository_module, name, recorded)
    return sites


def _record_mutations(screen, monkeypatch) -> list[tuple[str, dict, bool, str]]:
    """Record every folder mutation the screen runs, passing each through."""
    outcomes: list[tuple[str, dict, bool, str]] = []
    real = screen._execute_library_notes_tree_mutation

    async def recorded(operation, **payload):
        ok = await real(operation, **payload)
        asked = {key: value for key, value in payload.items() if key[0] != "_"}
        outcomes.append((operation, asked, ok, screen._notes_state.notice))
        return ok

    monkeypatch.setattr(screen, "_execute_library_notes_tree_mutation", recorded)
    return outcomes


async def _notes_ready(host, pilot) -> LibraryScreen:
    """Boot into Notes; wait until the root folders and Unfiled paint."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _open_notes(screen, pilot)

    def settled() -> bool:
        folders = screen._notes_state.tree_branches.get(NotesBranchKey(None, "folders"))
        return (
            folders is not None
            and folders.freshness == "fresh"
            and not folders.loading
            and "▸ Journal" in _painted(host)
        )

    await _wait(pilot, settled, "the root folder rows")
    return screen


async def _press(pilot, host, screen, find, done, what) -> None:
    """Press the button ``find()`` returns now, until ``done()`` holds.

    The previous move's tree reads land at no fixed moment, and the canvas
    recompose they cause can prune the pressed button before its ``Pressed``
    message is handled -- measured: the Move note instance read
    ``is_attached`` True at the press and False 2.5 s later with no picker,
    and the press was simply gone. So the CURRENT button is pressed again,
    but only once the one pressed is provably detached, nothing opened, and
    a further beat has passed for a press already on its way.
    """
    deadline = time.monotonic() + WAIT
    pressed: Button | None = None
    while not done():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what()}")
        if pressed is None or not pressed.is_attached:
            if pressed is not None:
                # Detached with nothing done: one beat for a press already
                # on its way before pressing the button that replaced it.
                await pilot.pause(0.3)
                if done():
                    return
                assert host.screen is screen, f"a dialog opened: {host.screen!r}"
            button = find()
            if button is not None and button.is_attached:
                button.press()
                pressed = button
        await pilot.pause(0.02)


def _row_for(screen, note_id: str, folder_id: str | None) -> Button | None:
    for row in _note_rows(screen):
        if row.note_id == note_id and (row.folder_id or None) == folder_id:
            return row
    return None


async def _move(host, pilot, screen, outcomes, seeded, move) -> tuple:
    """One Move note: press the row, Move note, type, Down, Enter.

    Starts as soon as the source row is in the tree -- for a foldered note
    that is the row the previous move's locator just revealed -- and returns
    as soon as the mutation itself returns, never waiting for the tree's
    follow-up reads to settle.
    """
    title, source, query, destination = move
    note_id = seeded.note_ids[title]
    source_id = seeded.folder_ids[source] if source else None
    await _wait(
        pilot,
        lambda: _row_for(screen, note_id, source_id) is not None,
        lambda: f"{title!r}'s row in {source or 'Unfiled'}",
    )
    await _press(
        pilot,
        host,
        screen,
        lambda: _row_for(screen, note_id, source_id),
        lambda: (selected := screen._selected_library_notes_tree_row()) is not None
        and selected.note_id == note_id
        and bool(screen.query("#library-notes-placement-move"))
        and not screen._library_notes_mutation_fenced(),
        lambda: f"{title!r} selected with Move note offered",
    )
    return await _pick(
        host,
        pilot,
        screen,
        outcomes,
        opener="#library-notes-placement-move",
        query=query,
        destination=destination,
        what=title,
    )


async def _pick(
    host, pilot, screen, outcomes, *, opener, query, destination, what
) -> tuple:
    """Open the folder picker with ``opener``, type, Down, Enter.

    Returns the mutation the choice ran (``_record_mutations``' record).
    """
    ran = len(outcomes)
    await _press(
        pilot,
        host,
        screen,
        lambda: next(iter(screen.query(opener)), None),
        lambda: isinstance(host.screen, LibraryNoteFolderTargetDialog),
        lambda: f"the folder picker for {what!r}",
    )
    await _wait(
        pilot,
        lambda: bool(_picker_rows(host)),
        f"the folder picker's rows for {what!r}",
    )
    await pilot.press(*query)
    await _wait(
        pilot,
        lambda: _picker_rows(host)[:1] == [destination]
        and host.screen._shown_query == query,
        lambda: f"the list narrowed to {destination!r}: {_picker_rows(host)}",
    )
    await pilot.press("down", "enter")
    await _wait(pilot, lambda: host.screen is screen, "the picker to choose")
    await _wait(pilot, lambda: len(outcomes) > ran, f"{what!r}'s mutation to run")
    return outcomes[-1]


def _painted_text_of(host, widget) -> str:
    """What ``widget``'s region paints, its lines joined by single spaces."""
    strips = list(host.screen._compositor.render_strips())
    region = widget.region
    lines = [
        strips[y].crop(region.x, region.right).text.strip()
        for y in range(region.y, min(region.bottom, len(strips)))
    ]
    return " ".join(line for line in lines if line)


def _status_line(host, screen) -> str:
    matches = list(screen.query("#library-notes-status-row"))
    return _painted_text_of(host, matches[0]) if matches else ""


def _laid_out(screen, selector: str) -> bool:
    """``selector`` is on screen now, not merely mounted behind a hidden pane."""
    return any(
        widget.display and widget.region.area > 0 for widget in screen.query(selector)
    )


def _folder_row(screen, folder_id: str) -> Button | None:
    for row in screen.query(".library-notes-folder-row").results(Button):
        if getattr(row, "folder_id", "") == folder_id:
            return row
    return None


async def test_after_deleting_the_selected_note_move_and_add_act_on_nothing(
    tmp_path, monkeypatch
) -> None:
    """AC#6 (owner ruling on question 3): once the selected note is deleted,
    nothing stays selected, so Move note and Add to folder act on nothing --
    not on the neighbour the tree would otherwise have selected."""
    host, seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        outcomes = _record_mutations(screen, monkeypatch)
        note_id = seeded.note_ids["Loose note 4"]
        await _press(
            pilot,
            host,
            screen,
            lambda: _row_for(screen, note_id, None),
            lambda: screen._library_note_session.snapshot is not None
            and screen._library_note_session.snapshot.note_id == note_id
            and bool(screen.query("#library-note-context")),
            lambda: "'Loose note 4' open in the editor",
        )
        selected = screen._selected_library_notes_tree_row()
        assert selected is not None and selected.note_id == note_id

        for control, then in (
            ("#library-note-context", "#library-note-context-delete"),
            ("#library-note-context-delete", "#library-note-delete-confirm"),
        ):
            await _press(
                pilot,
                host,
                screen,
                lambda control=control: next(iter(screen.query(control)), None),
                lambda then=then: _laid_out(screen, then),
                lambda then=then, control=control: f"{then} after {control}",
            )
        await _press(
            pilot,
            host,
            screen,
            lambda: next(iter(screen.query("#library-note-delete-confirm")), None),
            lambda: "✓ deleted · Loose note 4" in _painted(host),
            lambda: f"the delete receipt: {_painted(host)}",
        )
        await _wait(
            pilot,
            lambda: _row_for(screen, note_id, None) is None
            and "○ Move note" in _painted(host),
            lambda: f"the note gone and Move note withdrawn: {_painted(host)}",
        )

        assert screen._notes_state.tree_selected_placement_id == ""
        assert not list(screen.query(".library-notes-tree-note-row.is-selected"))
        painted = _painted(host)
        assert "○ Add to folder" in painted and "○ Move note" in painted, painted
        for control in ("#library-notes-placement-move", "#library-notes-placement-add"):
            button = screen.query_one(control, Button)
            assert button.disabled, control
            await pilot.click(control)
            await pilot.pause(0.3)
            assert host.screen is screen, f"{control} opened {host.screen!r}"
        # The seam both buttons call refuses too, whatever the buttons say.
        screen._choose_library_notes_placement_target(move=True)
        screen._choose_library_notes_placement_target(move=False)
        await pilot.pause(0.3)
        assert host.screen is screen, host.screen
        assert outcomes == []


async def test_a_genuine_conflict_offers_a_working_refresh(tmp_path, monkeypatch) -> None:
    """AC#6: a folder changed behind the tree -> the notice names it, in full,
    with a Refresh control; Refresh reloads the tree and clears the notice,
    and the same move then succeeds against the reloaded folder."""
    host, seeded = _mounted(tmp_path)
    recipes = seeded.folder_ids["Recipes"]
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        outcomes = _record_mutations(screen, monkeypatch)
        await _press(
            pilot,
            host,
            screen,
            lambda: _folder_row(screen, recipes),
            lambda: (selected := screen._selected_library_notes_tree_row()) is not None
            and selected.folder_id == recipes
            and bool(screen.query("#library-notes-folder-move")),
            lambda: "Recipes selected with Move folder offered",
        )
        loaded_version = _folder_row(screen, recipes).folder_version
        # Behind the tree: another session changes Recipes.
        with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
            connection.execute(
                "UPDATE note_folders SET version = version + 1 WHERE id = ?",
                (recipes,),
            )

        _operation, asked, ok, _notice = await _pick(
            host,
            pilot,
            screen,
            outcomes,
            opener="#library-notes-folder-move",
            query="resea",
            destination="Research",
            what="Recipes",
        )
        assert (asked["expected_version"], ok) == (loaded_version, False), outcomes

        conflict = "This folder was changed elsewhere since the tree loaded."
        await _wait(
            pilot,
            lambda: conflict in _status_line(host, screen)
            and "Refresh" in _status_line(host, screen),
            lambda: f"the whole conflict sentence and Refresh: "
            f"{_status_line(host, screen)!r}",
        )
        assert "…" not in _status_line(host, screen)
        assert screen._notes_state.notice_refresh_offered is True

        # M5 (fix round 1): Refresh has a portable focus role. A background
        # canvas sync while it holds focus -- the real Trash reload here --
        # recomposes it, and focus returns to the NEW Refresh, not to the
        # filter fallback (measured on e9ec76b170: the filter).
        refresh = screen.query_one("#library-notes-notice-refresh", Button)
        screen.set_focus(refresh)
        await _wait(pilot, lambda: screen.focused is refresh, "Refresh focused")
        assert screen._library_notes_semantic_role(refresh) == "notice-refresh"
        screen._refresh_library_notes_trash()
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused is not refresh
            and screen.focused.id == "library-notes-notice-refresh",
            lambda: f"focus kept on the recomposed Refresh: {screen.focused!r}",
        )
        assert (
            screen._library_notes_role_target(
                LibraryNotesFocusIdentity(
                    stage="notes",
                    region="navigator",
                    note_id=None,
                    semantic_role="notice-refresh",
                )
            )
            is screen.focused
        )

        # A real Enter on the focused Refresh.
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: not screen._notes_state.notice
            and (row := _folder_row(screen, recipes)) is not None
            and row.folder_version == loaded_version + 1,
            lambda: f"the tree reloaded: {_status_line(host, screen)!r}",
        )
        await _wait(
            pilot,
            lambda: conflict not in _painted(host)
            and not screen.query("#library-notes-notice-refresh"),
            lambda: f"the notice and Refresh gone: {_status_line(host, screen)!r}",
        )
        assert screen._notes_state.notice_refresh_offered is False

        # Where focus lands: Refresh starts a fresh visit, so the tree is
        # empty while it reloads, and the list-entry focus Refresh arms hands
        # an empty Notes list to the filter (task-31946's designed landing).
        # Waited on, never read straight after a mount.
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused.id == "library-notes-filter",
            lambda: f"focus on the filter after Refresh: {screen.focused!r}",
        )
        # And a real key still works on the recomposed screen: Tab moves on.
        await pilot.press("tab")
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen.focused.id != "library-notes-filter",
            lambda: f"Tab to move focus off the filter: {screen.focused!r}",
        )

        # The reloaded tree carries the new version: the same move now works.
        await _press(
            pilot,
            host,
            screen,
            lambda: _folder_row(screen, recipes),
            lambda: (selected := screen._selected_library_notes_tree_row()) is not None
            and selected.folder_id == recipes
            and bool(screen.query("#library-notes-folder-move")),
            lambda: "Recipes selected again",
        )
        _operation, asked, ok, notice = await _pick(
            host,
            pilot,
            screen,
            outcomes,
            opener="#library-notes-folder-move",
            query="resea",
            destination="Research",
            what="Recipes",
        )
        assert (asked["expected_version"], ok, notice) == (
            loaded_version + 1,
            True,
            UPDATED,
        ), outcomes


async def test_new_note_keeps_focus_after_a_targeted_notes_refresh(tmp_path) -> None:
    """A completed background refresh preserves the user's real Tab to New."""
    host, _seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        assert await pilot.click("#library-notes-filter")
        await _wait(
            pilot,
            lambda: screen.focused is screen.query_one("#library-notes-filter"),
            "filter focused",
        )
        await pilot.press("tab")
        await _wait(
            pilot,
            lambda: screen.focused is screen.query_one("#library-notes-new"),
            "Tab reaches New",
        )
        previous = screen.focused
        screen._refresh_library_notes_trash()
        await _wait(
            pilot,
            lambda: not previous.is_attached
            and bool(screen.query("#library-notes-new"))
            and screen.focused is screen.query_one("#library-notes-new"),
            lambda: f"New retains focus after its replacement: {screen.focused!r}",
        )
