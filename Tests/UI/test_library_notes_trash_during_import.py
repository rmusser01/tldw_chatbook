"""Recently deleted while an import runs: reads act, writes say why, one release.

TASK-34000.29 final review I-2, I-3 and m-3. After Task 10 Escape leaves a
running import, so the user can open Recently deleted while a 60-file import
still runs. At ``29d2272b95`` the import's fence (``_library_notes_mutation_
fenced``) swallowed every Trash READ: the opener returned silently, Previous,
Next and Retry moved ``trash_page`` and queued an owed reload that nothing
released (an import end only reloaded the tree), a second Next counted from
the page already asked for, and the stale owed role fired on the NEXT
unrelated release. ``x`` painted the busy reason and only a later re-read
cleared it.

Now the Trash's reads wait behind the WRITER interlock alone
(``_library_notes_writer_held``): under an import a page press reads at
once, ``x`` paints ``PURGE_BUSY_NOTICE`` and Restore (re-review NB-1, R-3)
``RESTORE_BUSY_NOTICE`` on the pressed row's behalf, and the import ending
releases through the ONE release body (``_release_library_notes_mutation``),
which clears the reason and runs any read queued under a Restore exactly
once. Leaving the Trash drops a queued read (m-3).

One boot of the real ``LibraryScreen`` on the production stylesheet over a
real ChaChaNotes database with 25 tombstones (two pages) and the real import
executor, the first note write held on a thread event so the import is
provably mid-flight; plus controller-shaped fakes for the release paths.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import _NotesProfile
from Tests.UI.test_library_notes_import_escape_keeps_running import (
    _Gate,
    _batch_of_one,
    _footer,
    _open_import_once_on_the_folder,
    _press,
    _review_the_folder,
    _until,
)
from Tests.UI.test_library_notes_riders_trash import (
    NOW,
    _ControllerShaped,
    _deleted_records,
    _pager_press_fake,
)
from Tests.UI.test_library_notes_trash_pages_and_purges import _focused_id
from Tests.UI.test_library_notes_trash_pages_and_purges_extended import (
    _pager,
    _painted_in,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_note_import_state import NoteImportPhase
from tldw_chatbook.Library.library_notes_state import build_library_notes_trash_state
from tldw_chatbook.Notes.note_import_executor import LocalNoteImportTarget
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_canvas import PURGE_BUSY_NOTICE

pytestmark = pytest.mark.bootstrap_profile

SIZE = (120, 36)
#: Two Trash pages: page 1 holds 01-20, page 2 holds 21-25.
DELETED = 25
#: Files in the folder the held import works through.
NOTES = 5
#: The words a fenced Restore paints (``RESTORE_BUSY_NOTICE``), spelled out
#: as the User Guide quotes them, so the pin reads what the user reads.
RESTORE_WAITS = "Restore waits: another change to your notes is still running."


class _Seed:
    """Two live notes, 25 dated tombstones and a folder of files to import."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = tmp_path / "notes.sqlite3"
        self.profile = _NotesProfile(tmp_path)
        db = self.profile.db
        titles = [f"Deleted {index:02d}" for index in range(1, DELETED + 1)]
        ids = []
        for title in titles:
            note_id = db.add_note(title, f"{title} body")
            db.soft_delete_note(note_id, db.get_note_by_id(note_id)["version"])
            ids.append(note_id)
        # One distinct minute each, newest deletion first (the page split
        # would otherwise be a coin toss inside one millisecond).
        with sqlite3.connect(self.path) as connection:
            for minute, note_id in enumerate(ids):
                connection.execute(
                    "UPDATE notes SET last_modified = ? WHERE id = ?",
                    (f"2026-10-29T09:{59 - minute:02d}:00.000Z", note_id),
                )
        monkeypatch.setattr(
            library_screen_module,
            "get_notes_sync_state_db_path",
            lambda: tmp_path / "import-receipts.sqlite",
        )
        self.folder = tmp_path / "Field notes"
        self.folder.mkdir()
        for index in range(1, NOTES + 1):
            (self.folder / f"note-{index}.md").write_text(
                f"# Field note {index}\n\nBody {index}.", encoding="utf-8"
            )

    def host(self) -> LibraryProductionCSSHarness:
        app = _build_test_app()
        _seed_conversations(app, [])
        app.chachanotes_db = self.profile.db
        app.notes_scope_service = self.profile.scope_service
        app.notes_service = self.profile.interop
        return LibraryProductionCSSHarness(app, screen=LibraryScreen(app))

    def deleted(self) -> int:
        with sqlite3.connect(self.path) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM notes WHERE deleted = 1"
                ).fetchone()[0]
            )


@pytest.mark.asyncio
async def test_next_in_the_trash_during_an_import_reads_the_page_and_x_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-2 at 120x36: a held import runs; the Trash opens, Next reads page 2
    and lands on Next, ``x`` paints the busy reason, and the import ending
    clears it with no press -- the page counter never drifts.

    RED at 29d2272b95: the opener's press is swallowed by the import fence,
    so the Trash never opens (and a Next in it would have queued silently).
    """
    seed = _Seed(tmp_path, monkeypatch)
    write = _Gate()
    monkeypatch.setattr(
        LocalNoteImportTarget,
        "create_note",
        write.wrap(LocalNoteImportTarget.create_note),
    )
    host = seed.host()
    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._library_note_import_controller
            _batch_of_one(controller)
            await _open_import_once_on_the_folder(host, screen, pilot, seed.folder)
            await _review_the_folder(host, screen, pilot)
            await _press(host, screen, pilot, "#note-import-import")
            await _until(pilot, write.entered.is_set, "the first note write to start")
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING
            assert screen._library_notes_mutation_fenced()

            await pilot.press("escape")
            await _until(
                pilot,
                lambda: screen._notes_state.view == "list"
                and f"Recently deleted ({DELETED})" in _painted_in(host, "#library-notes-trash-open"),
                lambda: (
                    "Escape to the Notes list with the Trash opener painted: "
                    f"view={screen._notes_state.view!r}"
                ),
            )
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING

            # The open is a read: the running import does not swallow it.
            await pilot.click("#library-notes-trash-open")
            await _until(
                pilot,
                lambda: screen._notes_state.view == "trash"
                and _focused_id(screen) == "library-notes-trash-restore-0"
                and _pager(screen) == f"1–20 of {DELETED} · Page 1 of 2",
                lambda: (
                    "the Trash to open on its first row during the import: "
                    f"view={screen._notes_state.view!r} "
                    f"focus={_focused_id(screen)!r} pager={_pager(screen)!r}"
                ),
            )

            # Next reads page 2 at once and lands on the pager -- on Previous,
            # since Next is disabled on the last page (Task 8a's landing rule):
            # no silent press.
            await pilot.click("#library-notes-trash-next")
            await _until(
                pilot,
                lambda: _pager(screen) == f"21–25 of {DELETED} · Page 2 of 2"
                and _focused_id(screen) == "library-notes-trash-previous",
                lambda: (
                    "Next to read page 2 during the import: "
                    f"pager={_pager(screen)!r} focus={_focused_id(screen)!r} "
                    f"owed={screen._notes_controller._trash_reload_owed!r}"
                ),
            )
            assert screen._notes_controller._library_notes_trash_page == 2
            assert screen._notes_controller._trash_reload_owed == ""
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING

            # x on a row (focused by a title click) says why, writes nothing.
            rows = list(screen.query(".library-notes-trash-row"))
            await pilot.click(rows[1].query_one(".library-notes-trash-row-copy"))
            await _until(
                pilot,
                lambda: _focused_id(screen) == "library-notes-trash-restore-1",
                lambda: f"the clicked row's Restore (focus: {_focused_id(screen)!r})",
            )
            before = seed.deleted()
            await pilot.press("x")
            await _until(
                pilot,
                lambda: _painted_in(host, "#library-notes-trash-notice")
                == PURGE_BUSY_NOTICE,
                lambda: "the busy reason on the Trash's own line, painted "
                f"{_painted_in(host, '#library-notes-trash-notice')!r}",
            )
            assert host.screen is screen, "a confirm opened under the fence"
            assert seed.deleted() == before

            # The import ends: the reason clears by itself, the page holds.
            write.release.set()
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.RECEIPT,
                lambda: f"the import to settle, phase={controller.snapshot.phase}",
            )
            await _until(
                pilot,
                lambda: not screen.query("#library-notes-trash-notice")
                and _pager(screen) == f"21–25 of {DELETED} · Page 2 of 2",
                lambda: (
                    "the busy reason to clear on the same page: "
                    f"notice={_painted_in(host, '#library-notes-trash-notice')!r} "
                    f"pager={_pager(screen)!r}"
                ),
            )
            # The settle's own tree reload re-reads the Trash page it is on
            # (page 2), so the page holds once that read lands too.
            def on_page_2_and_settled() -> bool:
                trash = screen._notes_state.trash
                return (
                    screen._notes_state.view == "trash"
                    and trash is not None
                    and not trash.loading
                    and trash.page == 2
                    and _pager(screen) == f"21–25 of {DELETED} · Page 2 of 2"
                    and not screen.query("#library-notes-trash-notice")
                )

            for _ in range(2):
                await _until(
                    pilot,
                    on_page_2_and_settled,
                    lambda: (
                        "the Trash to hold page 2 after the import settled: "
                        f"view={screen._notes_state.view!r} "
                        f"trash={screen._notes_state.trash!r} "
                        f"pager={_pager(screen)!r}"
                    ),
                )
                await pilot.pause(0.3)
            assert on_page_2_and_settled()
            assert screen._notes_controller._library_notes_trash_page == 2
            assert seed.deleted() == before
            assert seed.profile.db.count_notes() == 2 + NOTES
    finally:
        write.release.set()
        seed.profile.interop.close_all_user_connections()
        seed.profile.db.close_connection()


@pytest.mark.asyncio
async def test_restore_during_an_import_says_why_keeps_its_row_then_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-review NB-1 (residuals R-2, R-3): under a held import ``r`` on a
    Trash row paints why Restore waits, restores nothing, and keeps focus
    owed to its row; ``x`` on the same row then paints its own reason; the
    import ending clears the line through the one release, and ``r`` then
    restores the note.

    The owed row is proven as the ``x`` pin proves it: focus is parked on the
    grip (a programmatic move, as Textual's reset is) before the release.

    RED at 5d0b731bd1: the ``r`` binding is gated on the import's fence, so
    the key does nothing and the Trash's line stays empty (a Restore press
    returned silently), while the footer offers ``r restore note``. RED at
    ff2efa993b: the line after ``r`` names Delete forever.
    """
    seed = _Seed(tmp_path, monkeypatch)
    write = _Gate()
    monkeypatch.setattr(
        LocalNoteImportTarget,
        "create_note",
        write.wrap(LocalNoteImportTarget.create_note),
    )
    host = seed.host()

    def deleted_flag(note_id: str) -> int:
        with sqlite3.connect(seed.path) as connection:
            return int(
                connection.execute(
                    "SELECT deleted FROM notes WHERE id = ?", (note_id,)
                ).fetchone()[0]
            )

    try:
        async with host.run_test(size=SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            controller = screen._library_note_import_controller
            _batch_of_one(controller)
            await _open_import_once_on_the_folder(host, screen, pilot, seed.folder)
            await _review_the_folder(host, screen, pilot)
            await _press(host, screen, pilot, "#note-import-import")
            await _until(pilot, write.entered.is_set, "the first note write to start")
            await pilot.press("escape")
            await _until(
                pilot,
                lambda: screen._notes_state.view == "list",
                "Escape to the Notes list",
            )
            await pilot.click("#library-notes-trash-open")
            await _until(
                pilot,
                lambda: _focused_id(screen) == "library-notes-trash-restore-0"
                and _pager(screen) == f"1–20 of {DELETED} · Page 1 of 2",
                lambda: f"the Trash's first row (focus: {_focused_id(screen)!r})",
            )
            for _ in range(2):
                await _until(
                    pilot,
                    lambda: not screen._notes_state.trash.loading
                    and _focused_id(screen) == "library-notes-trash-restore-0",
                    lambda: f"the Trash to settle (focus: {_focused_id(screen)!r})",
                )
                await pilot.pause(0.2)
            # A row other than the first, reached by real keys (one Tab stop
            # per row), so the Trash's own first-row fallback cannot pass for
            # the owed row.
            for index in (1, 2):
                await pilot.press("tab")
                await _until(
                    pilot,
                    lambda: _focused_id(screen) == f"library-notes-trash-restore-{index}",
                    lambda: f"Tab to row {index} (focus: {_focused_id(screen)!r})",
                )
            note_id = str(screen.focused.note_id)
            before = seed.deleted()
            assert controller.snapshot.phase is NoteImportPhase.IMPORTING
            assert "restore note" in _footer(host, screen), _footer(host, screen)

            await pilot.press("r")
            await _until(
                pilot,
                lambda: _painted_in(host, "#library-notes-trash-notice")
                == RESTORE_WAITS,
                lambda: "Restore's busy reason on the Trash's own line, painted "
                f"{_painted_in(host, '#library-notes-trash-notice')!r}",
            )
            assert seed.deleted() == before
            assert deleted_flag(note_id) == 1, "the note was restored under the fence"
            assert _focused_id(screen) == "library-notes-trash-restore-2"
            assert note_id in {row.note_id for row in screen._notes_state.trash.rows}

            # Residual R-3: x on the same row, still fenced, names its own
            # action -- the line follows the last key pressed -- and opens
            # no confirm under the fence.
            await pilot.press("x")
            await _until(
                pilot,
                lambda: _painted_in(host, "#library-notes-trash-notice")
                == PURGE_BUSY_NOTICE,
                lambda: "x's busy reason replacing Restore's, painted "
                f"{_painted_in(host, '#library-notes-trash-notice')!r}",
            )
            assert host.screen is screen, "a confirm opened under the fence"
            assert seed.deleted() == before
            assert _focused_id(screen) == "library-notes-trash-restore-2"

            # Textual's reset landing, not yet replayed, as the release's
            # syncs capture it.
            grip = screen.query_one("#library-browse-items-grip")
            screen.set_focus(grip)
            await _until(pilot, lambda: screen.focused is grip, "focus on the grip")

            write.release.set()
            await _until(
                pilot,
                lambda: controller.snapshot.phase is NoteImportPhase.RECEIPT,
                lambda: f"the import to settle, phase={controller.snapshot.phase}",
            )
            for _ in range(2):
                await _until(
                    pilot,
                    lambda: not screen.query("#library-notes-trash-notice")
                    and _focused_id(screen) == "library-notes-trash-restore-2"
                    and not screen._notes_state.trash.loading,
                    lambda: (
                        "the reason cleared with focus still on the row: "
                        f"focus={_focused_id(screen)!r} "
                        f"notice={_painted_in(host, '#library-notes-trash-notice')!r}"
                    ),
                )
                await pilot.pause(0.3)
            assert deleted_flag(note_id) == 1

            # The fence is down: the same key restores the same note.
            await pilot.press("r")
            await _until(
                pilot,
                lambda: deleted_flag(note_id) == 0 and seed.deleted() == before - 1,
                lambda: f"r to restore the note (deleted now {seed.deleted()})",
            )
            assert not screen.query("#library-notes-trash-notice")
    finally:
        write.release.set()
        seed.profile.interop.close_all_user_connections()
        seed.profile.db.close_connection()


# -- the release paths, on controller-shaped fakes ----------------------------


def _settle_fake(*, owed: str, notice: str):
    """A controller after a Restore queued ``owed`` and ``x`` painted ``notice``."""
    refreshes: list[dict] = []
    trash = build_library_notes_trash_state(
        _deleted_records(1, 21), total=37, page=2, now=NOW
    )
    fake = _ControllerShaped(
        _notes_state=SimpleNamespace(mutation_in_flight=False),
        _library_notes_trash=dataclasses.replace(trash, notice=notice),
        _library_notes_trash_page=2,
        _library_notes_view="trash",
        _tree_reload_owed=False,
        _trash_reload_owed=owed,
        _trash_focus_role=owed,
        _trash_landing_settling=owed,
        _refresh_local_source_snapshot=lambda: None,
        _request_library_notes_tree_initial_load=lambda: None,
        _refresh_library_notes_trash=lambda **kwargs: refreshes.append(kwargs),
        _apply_library_notes_footer_context=lambda: None,
        _library_note_import_execution_active=lambda: False,
        is_mounted=False,
    )
    return fake, refreshes


def test_an_import_ending_releases_once_running_the_owed_read_and_clearing_busy():
    """I-2/I-3: the import's settle path is a release like any other.

    A page read queued under a Restore runs exactly once, and the busy
    reason clears; a second settle owes nothing. RED at 29d2272b95: the
    settle path only reloaded the tree.
    """
    fake, refreshes = _settle_fake(owed="trash-next", notice=PURGE_BUSY_NOTICE)

    LibraryNotesController._refresh_after_library_note_import(fake)

    assert refreshes == [{"focus_role": "trash-next"}]
    assert fake._trash_reload_owed == ""
    assert fake._library_notes_trash.notice == ""
    assert fake._notes_state.mutation_in_flight is False

    LibraryNotesController._refresh_after_library_note_import(fake)
    assert refreshes == [{"focus_role": "trash-next"}]


def test_an_import_ending_never_clears_a_writer_still_in_flight():
    """A mutation in flight at import end owns its own release."""
    fake, refreshes = _settle_fake(owed="trash-next", notice=PURGE_BUSY_NOTICE)
    fake._notes_state.mutation_in_flight = True

    LibraryNotesController._refresh_after_library_note_import(fake)

    assert fake._notes_state.mutation_in_flight is True
    assert refreshes == []
    assert fake._trash_reload_owed == "trash-next"


def test_leaving_the_trash_drops_a_queued_page_read_so_no_stale_role_fires():
    """m-3: Back or Escape with a read owed clears it and its settling mark;
    a later, unrelated release (the next Restore's) reads nothing for it.
    RED at 29d2272b95: ``_leave_library_notes_trash`` left the owed role."""
    fake, refreshes = _settle_fake(owed="trash-next", notice="")

    LibraryNotesController._leave_library_notes_trash(fake)

    assert fake._library_notes_view == "list"
    assert fake._trash_reload_owed == ""
    assert fake._trash_landing_settling == ""
    assert fake._trash_focus_role == ""
    assert fake._library_notes_trash_page == 1

    fake._notes_state.mutation_in_flight = True
    LibraryNotesController._release_library_notes_mutation(fake)
    assert refreshes == []
    assert fake._notes_state.mutation_in_flight is False


@pytest.mark.parametrize("press", ["next", "retry", "open"])
def test_a_trash_read_under_an_import_fence_acts_at_once(press: str) -> None:
    """I-2: with the writer free and only the import's fence up, Next, Retry
    and the opener read at once -- nothing is queued, nothing is silent.
    RED at 29d2272b95: all three asked ``_library_notes_mutation_fenced``.
    """
    fake, refreshes = _pager_press_fake(fenced=False)
    fake._library_notes_mutation_fenced = lambda: True  # an import runs
    fake._library_notes_view = "list" if press == "open" else "trash"
    fake._apply_library_notes_footer_context = lambda: None
    fake.is_mounted = False
    event = SimpleNamespace(stop=lambda: None)

    if press == "next":
        LibraryNotesController.handle_library_notes_trash_next(fake, event)
        assert refreshes == [{"focus_role": "trash-next"}]
        assert fake._library_notes_trash_page == 2
    elif press == "retry":
        LibraryNotesController.handle_library_notes_trash_retry(fake, event)
        assert refreshes == [{"focus_role": "trash-restore:0"}]
    else:
        LibraryNotesController.handle_library_notes_trash_open(fake, event)
        assert fake._library_notes_view == "trash"
        assert refreshes == [{}]
    assert getattr(fake, "_trash_reload_owed", "") == ""
    assert fake._library_notes_mutation_in_flight is False
