"""Recently deleted keeps every title readable at 120x36 (review R0, I-2).

TASK-34000.29 (N-29, part 8b). With Restore and Delete forever on every row,
the 120x36 Trash cut each journal title to "Daily log 2026-…" (live capture
``t8b-2-01``), so the list could no longer tell its notes apart. Delete
forever now shows on the FOCUSED row only -- exactly where ``x`` is live --
and a hidden one takes no width. A click on a row's title focuses that row's
Restore and never presses it, so the mouse reaches Delete forever in two
clicks (the title, then the button) without restoring anything.

One boot of the real ``LibraryScreen`` with the production stylesheet over a
real ChaChaNotes database (``_NotesProfile`` plus twelve dated tombstones).
Every claim is painted text, a focused control, or the database read back
through a second connection.

Base (780fe1da8b): every unfocused row read "Daily log 2026-…", and a title
click focused nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import _NotesProfile, _record_toasts
from Tests.UI.test_library_notes_trash_pages_and_purges import (
    _focused_id,
    _painted,
    _settled_on,
    _until,
)
from Tests.UI.test_library_notes_trash_pages_and_purges_extended import (
    _open_trash,
    _painted_in,
)
from Tests.UI.test_library_shell import LibraryProductionCSSHarness, _seed_conversations
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The size the review measured the regression at.
SIZE = (120, 36)
#: Journal titles that differ only in their last two characters.
TITLES = tuple(f"Daily log 2026-09-{day:02d}" for day in range(1, 13))


class _JournalSeed:
    """``_NotesProfile``'s two live notes plus twelve dated tombstones."""

    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "notes.sqlite3"
        self.profile = _NotesProfile(tmp_path)
        db = self.profile.db
        self.ids: dict[str, str] = {}
        for title in TITLES:
            note_id = db.add_note(title, f"{title} body")
            db.soft_delete_note(note_id, db.get_note_by_id(note_id)["version"])
            self.ids[title] = note_id
        # Newest deletion first: 01 at the top, one distinct minute each.
        with sqlite3.connect(self.path) as connection:
            for minute, title in enumerate(TITLES):
                connection.execute(
                    "UPDATE notes SET last_modified = ? WHERE id = ?",
                    (f"2026-10-27T09:{59 - minute:02d}:00.000Z", self.ids[title]),
                )

    def deleted(self) -> set[str]:
        with sqlite3.connect(self.path) as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT title FROM notes WHERE deleted = 1"
                ).fetchall()
            }


def _row_text(host, row) -> str:
    """One Trash row exactly as painted (its single line)."""
    region = row.region
    strips = host.screen._compositor.render_strips()
    return " ".join(
        strip.crop(region.x, region.right).text
        for strip in strips[region.y : region.bottom]
    )


def _rows(screen) -> list:
    return list(screen.query(".library-notes-trash-row"))


async def test_every_unfocused_title_reads_whole_and_a_title_click_restores_nothing(
    tmp_path, monkeypatch
) -> None:
    # Review R1 m-r1-2: the canvas constant is read here, where the busy
    # arm uses it, not at module level.
    from tldw_chatbook.Widgets.Library.library_notes_canvas import PURGE_BUSY_NOTICE

    app = _build_test_app()
    _seed_conversations(app, [])
    seed = _JournalSeed(tmp_path)
    app.chachanotes_db = seed.profile.db
    app.notes_scope_service = seed.profile.scope_service
    app.notes_service = seed.profile.interop
    host = LibraryProductionCSSHarness(app, screen=LibraryScreen(app))
    app.notify = host.notify
    toasts = _record_toasts(monkeypatch, app)

    async with host.run_test(size=SIZE) as pilot:
        screen = await _open_trash(host, pilot, count=len(TITLES))
        rows = _rows(screen)
        assert len(rows) == len(TITLES)
        painted = [_row_text(host, row) for row in rows]

        # The focused row offers Delete forever (x's own row) ...
        assert "Restore" in painted[0] and "Delete forever" in painted[0], painted[0]
        # ... and every other row paints its whole title and no Delete forever.
        for index, (title, text) in enumerate(zip(TITLES, painted)):
            if index == 0:
                continue
            assert title in text, f"row {index} cut its title: {text!r}"
            assert "Delete forever" not in text, f"row {index}: {text!r}"
        assert screen.check_action("library_notes_trash_delete", ()) is True
        assert "x delete forever" in _painted(host)

        # A click on a title focuses that row -- it never restores it.
        before = seed.deleted()
        copy = rows[3].query_one(".library-notes-trash-row-copy")
        await pilot.click(copy)
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: _focused_id(screen) == "library-notes-trash-restore-3",
            lambda: f"the clicked row's Restore (focus: {_focused_id(screen)!r})",
        )
        rows = _rows(screen)
        painted = [_row_text(host, row) for row in rows]
        assert len(rows) == len(TITLES)
        assert seed.deleted() == before
        assert toasts == []
        assert "Delete forever" in painted[3], painted[3]
        assert TITLES[0] in painted[0] and "Delete forever" not in painted[0]
        assert screen.check_action("library_notes_trash_delete", ()) is True

        # Review R0, m-5: while another Notes change holds the interlock, x
        # says why on the Trash -- no confirm, nothing written -- and the
        # reason goes once that change releases it.
        controller = screen._notes_controller
        controller._library_notes_mutation_in_flight = True
        assert screen.check_action("library_notes_trash_delete", ()) is True
        await pilot.press("x")
        await _until(
            pilot,
            lambda: _painted_in(host, "#library-notes-trash-notice")
            == PURGE_BUSY_NOTICE,
            lambda: f"the busy reason: {_painted(host)}",
        )
        assert host.screen is screen
        assert seed.deleted() == before
        controller._release_library_notes_mutation()
        await _until(
            pilot,
            lambda: not screen.query("#library-notes-trash-notice"),
            lambda: f"the reason cleared: {_painted(host)}",
        )
