"""Notes ▸ Recently deleted pages through every deleted note, with an exact total.

TASK-34000.29 (N-29), part 8a, review notes-library-ux-review-2026-10-02
(capture ``verify/nl-vb-2/n29-trash-160x45.txt``). The Trash view loaded the
newest 20 tombstones and said "Showing the 20 most recently deleted of N.
Restore one to see the rest." -- the 21st deletion was reachable only by
restoring one of the first 20. It pages now, the Media Trash way: "1–20 of 25
· Page 1 of 2" and Previous/Next.

Ruling T8-d, pinned here end to end: a Restore keeps focus on the trash list
(the row now at the restored index, clamped); a page past the shrunken total
clamps to the last page, never an empty one; after Next on the last page
focus is on Previous, never on the disabled Next or nowhere.

Delete forever (8b) waits for the owner's ruling; the file keeps its name so
that half can extend it.

One boot of the real ``LibraryScreen`` with the production stylesheet over a
real ChaChaNotes database (``_NotesProfile`` plus 25 soft-deleted notes).
Every gesture is a real click or key; every claim is painted text, a
focused control, or the database read back through a second connection.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from textual.widgets import Button
from textual.worker import WorkerState

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import _NotesProfile
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)

#: A behaviour wait: long enough for a page read under CI contention, short
#: enough that a RED arm names its reason in seconds.
WAIT = 10.0

#: Soft-deleted notes, newest deletion first: page 1 holds 01-20, page 2 21-25.
DELETED = 25
TITLES = tuple(f"Deleted {index:02d}" for index in range(1, DELETED + 1))


class _TrashSeed:
    """``_NotesProfile``'s two live notes plus ``DELETED`` tombstones."""

    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "notes.sqlite3"
        self.profile = _NotesProfile(tmp_path)
        db = self.profile.db
        self.ids: dict[str, str] = {}
        for title in TITLES:
            note_id = db.add_note(title, f"{title} body")
            db.soft_delete_note(note_id, db.get_note_by_id(note_id)["version"])
            self.ids[title] = note_id
        # Deletions inside one millisecond tie, so the page split was a coin
        # toss. One distinct minute each, newest deletion first.
        with sqlite3.connect(self.path) as connection:
            for minute, title in enumerate(TITLES):
                connection.execute(
                    "UPDATE notes SET last_modified = ? WHERE id = ?",
                    (f"2026-10-27T09:{59 - minute:02d}:00.000Z", self.ids[title]),
                )

    def deleted(self) -> set[str]:
        """Titles still soft-deleted, through a second connection."""
        with sqlite3.connect(self.path) as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT title FROM notes WHERE deleted = 1"
                ).fetchall()
            }


def _mounted(tmp_path: Path) -> tuple[LibraryProductionCSSHarness, _TrashSeed]:
    app = _build_test_app()
    _seed_conversations(app, [])
    seed = _TrashSeed(tmp_path)
    app.chachanotes_db = seed.profile.db
    app.notes_scope_service = seed.profile.scope_service
    app.notes_service = seed.profile.interop
    return LibraryProductionCSSHarness(app, screen=LibraryScreen(app)), seed


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


async def _until(pilot, predicate, what) -> None:
    try:
        async with asyncio.timeout(WAIT):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


def _focused_id(screen) -> str:
    return str(getattr(screen.focused, "id", "") or "")


#: The worker groups a Restore and a page read run in.
_TRASH_WORK = frozenset({"library_note_mutation", "library_notes_trash"})


async def _settled_on(pilot, host, screen, predicate, what) -> None:
    """Wait until ``predicate`` holds with no Restore or page read running.

    Every Notes canvas sync recomposes the Trash list, so for a moment focus
    sits on a pane grip (Textual's reset) until the replay puts it back; a
    key that arrives in that window is the grip's, not the row's (traced:
    1 run in 8 under 8-way load lost an ``r`` there). A Restore keeps
    syncing after its commit and its re-read page -- it holds the Notes
    interlock until the tree reconcile, then syncs once more -- so a gesture
    waits for both worker groups to finish and the focus to come back, and
    for that to hold across one more pause.
    """

    def idle() -> bool:
        return predicate() and not any(
            worker.group in _TRASH_WORK
            and worker.state in (WorkerState.PENDING, WorkerState.RUNNING)
            for worker in host.workers
        )

    for _ in range(2):
        await _until(pilot, idle, what)
        await pilot.pause(0.05)
    assert idle(), what() if callable(what) else what


async def _notes_tree_settled(host, pilot, screen) -> None:
    """Wait for the visit's reads to land before clicking into the tree.

    The opener row paints with the Trash read, which can land before the
    tree's own slices do; those then mount the two live rows ABOVE it, so a
    click aimed at the opener's first position hit an empty cell (1 run in
    5 before this wait).
    """

    def settled() -> bool:
        branches = screen._notes_state.tree_branches
        return (
            all(
                (branch := branches.get(NotesBranchKey(None, kind))) is not None
                and branch.freshness == "fresh"
                and not branch.loading
                for kind in ("folders", "placements")
            )
            and "Ideas inbox" in _painted(host)
            and f"Recently deleted ({DELETED})" in _painted(host)
        )

    await _until(pilot, settled, lambda: f"the settled tree: {_painted(host)}")
    await pilot.pause()


async def _restore_focused_row(host, pilot, screen, seed, title, *, then_range):
    """``r`` on the focused row restores exactly ``title``; the page re-reads.

    Settles first (``_settled_on``): a Restore pressed while the previous
    one still holds the Notes interlock is refused (``test_trash_restore_
    refuses_while_a_notes_mutation_is_in_flight``), which lost 3 of 16
    presses under 8-way load before this wait.
    """
    await _settled_on(
        pilot,
        host,
        screen,
        lambda: getattr(screen.focused, "note_title", "") == title,
        lambda: f"focus settled on {title} (focus: {_focused_id(screen)!r})",
    )
    await pilot.press("r")
    await _until(
        pilot,
        lambda: title not in seed.deleted() and then_range in _painted(host),
        lambda: f"{title} restored and {then_range!r} painted: {_painted(host)}",
    )


async def test_every_deleted_note_is_reachable_through_paging(tmp_path) -> None:
    """AC#10 and ruling T8-d's disabled-control and clamp arms, in one boot.

    Base (734e632f07): "1–20 of 25" never paints -- the view says "Showing
    the 20 most recently deleted of 25. Restore one to see the rest." and
    has no Next.
    """
    host, seed = _mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _notes_tree_settled(host, pilot, screen)

        await pilot.click("#library-notes-trash-open")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "1–20 of 25 · Page 1 of 2" in _painted(host)
            and "Deleted 20" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-restore-0",
            lambda: f"page 1 (focus: {_focused_id(screen)!r}): {_painted(host)}",
        )
        first_page = _painted(host)
        assert all(title in first_page for title in TITLES[:20])
        assert "Deleted 21" not in first_page

        # Next is the way on; the last page disables it, so focus is on
        # Previous -- not on the disabled Next, not on a pane grip.
        await pilot.click("#library-notes-trash-next")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "21–25 of 25 · Page 2 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-previous",
            lambda: f"page 2, focus on Previous (focus: {_focused_id(screen)!r}): "
            f"{_painted(host)}",
        )
        second_page = _painted(host)
        assert all(title in second_page for title in TITLES[20:])
        assert not any(title in second_page for title in TITLES[:20])
        assert screen.query_one("#library-notes-trash-next", Button).disabled

        # A real key from there: Enter on Previous reads page 1, where
        # Previous is disabled -- so focus is on Next; Enter again, page 2.
        await pilot.press("enter")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "1–20 of 25 · Page 1 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-next",
            lambda: f"page 1, focus on Next (focus: {_focused_id(screen)!r})",
        )
        await pilot.press("enter")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "21–25 of 25 · Page 2 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-previous",
            lambda: f"page 2 again (focus: {_focused_id(screen)!r})",
        )

        # Restore the last row: the page re-reads where it is, and focus
        # stays on the trash list -- on the row now at that index, clamped.
        await pilot.press("shift+tab")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: _focused_id(screen) == "library-notes-trash-restore-4",
            lambda: f"Shift+Tab to the last Restore (focus: {_focused_id(screen)!r})",
        )
        await _restore_focused_row(
            host, pilot, screen, seed, "Deleted 25", then_range="21–24 of 24"
        )
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: getattr(screen.focused, "note_title", "") == "Deleted 24",
            lambda: f"focus on the row now last (focus: {_focused_id(screen)!r})",
        )

        # Restore the other four: the page empties, so it clamps to page 1
        # -- never an empty page 2 -- and one page left means no controls.
        for title, after in (
            ("Deleted 24", "21–23 of 23"),
            ("Deleted 23", "21–22 of 22"),
            ("Deleted 22", "21–21 of 21"),
            ("Deleted 21", "1–20 of 20"),
        ):
            await _restore_focused_row(
                host, pilot, screen, seed, title, then_range=after
            )
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: getattr(screen.focused, "note_title", "") == "Deleted 01",
            lambda: f"focus on the first row of page 1 (focus: {_focused_id(screen)!r})",
        )
        final = _painted(host)
        assert "Page 1 of 1" not in final and "Page 2" not in final
        assert not screen.query("#library-notes-trash-previous")
        assert not screen.query("#library-notes-trash-next")
        assert seed.deleted() == set(TITLES[:20])

        # A real key still works: Escape leaves for the list, whose opener
        # names the exact total that is left.
        await pilot.press("escape")
        await _until(
            pilot,
            lambda: "Recently deleted (20)" in _painted(host),
            lambda: f"the list's opener after Escape: {_painted(host)}",
        )
