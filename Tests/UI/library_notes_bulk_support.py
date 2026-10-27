"""Shared fixtures for the Notes select-mode count, Space and bulk-action pins.

TASK-34000.29 (N-14) and TASK-34000.82. Two test files use these:
``test_library_notes_select_mode_counts_and_bulk.py`` (the PR-gated core) and
its ``_extended`` sibling. Not a test module, so pytest does not collect it.

The seed is a real ChaChaNotes database behind the real Notes scope service:
25 unfiled notes (the tree pages 20 of them on the first visit, so 20 are
rendered while the flat list holds every note), and three folders --
``Projects``, ``Projects / Thesis`` (one note) and ``Recipes`` (two notes).
The harness is ``LibraryProductionCSSHarness`` over a real ``LibraryScreen``:
a full ``TldwCli`` boot's deferred Notes wiring seeds its own
``Agent_Lessons`` root at no fixed moment after first paint, which would add
a folder row to the very tree whose rows these tests count
(lessons-testing-evidence, "A full `TldwCli` Pilot boot writes into the
test's own database after first paint").
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import _NotesProfile
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_notes_tree_paging import (
    LIBRARY_NOTES_TREE_PAGE_SIZE,
    NotesBranchKey,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

#: A behaviour wait: long enough for a slice load under CI contention, short
#: enough that a RED arm names its reason in seconds.
WAIT = 10.0

#: The three notes a filter on ``Trip`` finds, and nothing else.
TRIP_TITLES = ("Trip Kyoto", "Trip Lisbon", "Trip Oslo")

#: Unfiled notes beyond ``_NotesProfile``'s two and the three trips.
PLAIN_COUNT = 20

#: Every unfiled note: more than one tree page, so the flat list and the
#: rendered rows disagree (the review's "Select all 100 shown" over 20 rows).
UNFILED = 2 + PLAIN_COUNT + len(TRIP_TITLES)

#: What the first visit renders: one page of the unfiled placements.
RENDERED = LIBRARY_NOTES_TREE_PAGE_SIZE

#: The unfiled notes newest first -- the first page is the first 20 of these.
NEWEST_FIRST = (
    "Ideas inbox",
    "Reading list",
    *TRIP_TITLES,
    *(f"Note {index:02d}" for index in range(PLAIN_COUNT)),
)


class NotesSeed:
    """The seeded profile and the ids a test needs."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.profile = _NotesProfile(tmp_path)
        db = self.profile.db
        repository = LocalNoteFolderRepository(db)
        projects = repository.create_folder(name="Projects", parent_id=None)
        thesis = repository.create_folder(name="Thesis", parent_id=projects.folder_id)
        recipes = repository.create_folder(name="Recipes", parent_id=None)
        self.folder_ids = {
            "Projects": projects.folder_id,
            "Projects / Thesis": thesis.folder_id,
            "Recipes": recipes.folder_id,
        }
        self.note_ids: dict[str, str] = {
            "Ideas inbox": self.profile.note_id,
            "Reading list": self.profile.other_id,
        }
        for index in range(PLAIN_COUNT):
            title = f"Note {index:02d}"
            self.note_ids[title] = db.add_note(title, f"plain body {index}")
        for title in TRIP_TITLES:
            self.note_ids[title] = db.add_note(title, f"{title} itinerary")
        outline = db.add_note("Thesis outline", "- chapter one")
        repository.attach_manual(folder_id=thesis.folder_id, note_id=outline)
        self.note_ids["Thesis outline"] = outline
        for title in ("Soup", "Bread"):
            note_id = db.add_note(title, f"{title} method")
            repository.attach_manual(folder_id=recipes.folder_id, note_id=note_id)
            self.note_ids[title] = note_id
        # The tree pages newest first, and notes created in one second tie --
        # so which 20 the first page renders was a coin toss. One distinct
        # minute each, newest first in ``NEWEST_FIRST`` order.
        with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
            for minute, title in enumerate(NEWEST_FIRST):
                connection.execute(
                    "UPDATE notes SET last_modified = ? WHERE id = ?",
                    (f"2026-10-26T12:{59 - minute:02d}:00.000Z", self.note_ids[title]),
                )

    def memberships(self) -> list[tuple[str, str]]:
        """Every live ``(folder_id, note_id)`` placement, through a second
        connection to the test's own database file -- what a relaunch sees."""
        with sqlite3.connect(self.tmp_path / "notes.sqlite3") as connection:
            return sorted(
                connection.execute(
                    "SELECT folder_id, note_id FROM note_folder_memberships "
                    "WHERE deleted = 0"
                ).fetchall()
            )

    def placed_in(self, folder: str) -> set[str]:
        folder_id = self.folder_ids[folder]
        return {note for folder_id_, note in self.memberships() if folder_id_ == folder_id}

    def placement_rows(self, folder: str, title: str) -> list[tuple[int, int]]:
        """Every ``(deleted, version)`` row placing ``title`` in ``folder``,
        live or not, through a second connection."""
        with sqlite3.connect(self.tmp_path / "notes.sqlite3") as connection:
            return sorted(
                connection.execute(
                    "SELECT deleted, version FROM note_folder_memberships "
                    "WHERE folder_id = ? AND note_id = ?",
                    (self.folder_ids[folder], self.note_ids[title]),
                ).fetchall()
            )

    def keyword_state(self) -> tuple[list[tuple], list[tuple]]:
        """Every keyword row and every note-keyword link, live or not,
        through a second connection -- equal before and after means nothing
        was written."""
        with sqlite3.connect(self.tmp_path / "notes.sqlite3") as connection:
            return (
                sorted(connection.execute("SELECT * FROM keywords").fetchall()),
                sorted(connection.execute("SELECT * FROM note_keywords").fetchall()),
            )

    def keyword_links(self, keyword: str) -> set[str]:
        """Note ids linked to ``keyword`` (case-insensitive), second connection."""
        with sqlite3.connect(self.tmp_path / "notes.sqlite3") as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT nk.note_id FROM note_keywords AS nk "
                    "JOIN keywords AS k ON k.id = nk.keyword_id "
                    "WHERE k.keyword = ? COLLATE NOCASE AND k.deleted = 0",
                    (keyword,),
                ).fetchall()
            }


def mounted(tmp_path: Path) -> tuple[LibraryProductionCSSHarness, NotesSeed]:
    """The real Library hierarchy over the seed, production CSS."""
    app = _build_test_app()
    _seed_conversations(app, [])
    seed = NotesSeed(tmp_path)
    app.chachanotes_db = seed.profile.db
    app.notes_scope_service = seed.profile.scope_service
    app.notes_service = seed.profile.interop
    return LibraryProductionCSSHarness(app, screen=LibraryScreen(app)), seed


def painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


def painted_region(host, region) -> str:
    strips = list(host.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, min(region.bottom, len(strips)))
    )


async def until(pilot, predicate, what) -> None:
    """Pump until ``predicate()`` holds; ``what`` may be a callable read on failure."""
    try:
        async with asyncio.timeout(WAIT):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


def note_rows(screen) -> list[Button]:
    return list(screen.query(".library-notes-tree-note-row").results(Button))


def folder_row(screen, label: str) -> Button:
    for row in screen.query(".library-notes-folder-row").results(Button):
        if label in str(row.label):
            return row
    raise AssertionError(f"no folder row {label!r}")


def row_for(screen, title: str) -> Button:
    for row in note_rows(screen):
        if getattr(row, "note_id", "") and title in str(row.label):
            return row
    raise AssertionError(f"no rendered row for {title!r}: {[str(r.label) for r in note_rows(screen)]}")


async def notes_ready(host, pilot, *, rendered: int = RENDERED) -> LibraryScreen:
    """Boot into Notes; wait until the first page of rows is painted."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes", Button).press()

    def settled() -> bool:
        root = screen._notes_state.tree_branches.get(NotesBranchKey(None, "placements"))
        return (
            root is not None
            and root.freshness == "fresh"
            and not root.loading
            and len(note_rows(screen)) == rendered
            and "Recipes" in painted(host)
        )

    await until(
        pilot,
        settled,
        lambda: f"{rendered} painted note rows (have {len(note_rows(screen))})",
    )
    return screen


def strip_text(host, screen) -> str:
    """The painted select strip -- every row of it -- and nothing else."""
    strips = list(screen.query("#library-notes-selection-actions"))
    if not strips:
        return ""
    return painted_region(host, strips[0].region)


async def enter_select_mode(host, pilot, screen) -> None:
    await pilot.click("#library-notes-select-toggle")
    await until(
        pilot,
        lambda: screen._notes_state.select_mode
        and "Done" in strip_text(host, screen),
        lambda: f"the select strip to paint: {painted(host)}",
    )
