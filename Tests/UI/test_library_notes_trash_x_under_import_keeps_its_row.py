"""Recently deleted: ``x`` under an import keeps focus on its row (live I-2 find).

Polish-1b final fix wave, finishing pass (2026-10-29). The live I-2 check at
120x36 (``Docs/QA/notes-library-ux-review-2026-10-02/fixes/task-34000.29/
f1b-fin-run2-06``/``07a``) found focus on the FIRST row once a held import's
release cleared the busy reason that ``x`` on the last row had painted (2 of
10 live runs; ~1 in 12 in a Pilot of the same flow). Kept beside
``test_library_notes_trash_during_import.py``, whose seed and import helpers
it reuses, so each census file stays well under 30 s alone.

One boot of the real ``LibraryScreen`` on the production stylesheet over a
real ChaChaNotes database with 25 tombstones and the real import executor,
the first note write held on a thread event.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.UI.test_library_notes_import_escape_keeps_running import (
    _Gate,
    _batch_of_one,
    _open_import_once_on_the_folder,
    _press,
    _review_the_folder,
    _until,
)
from Tests.UI.test_library_notes_trash_during_import import DELETED, SIZE, _Seed
from Tests.UI.test_library_notes_trash_pages_and_purges import _focused_id
from Tests.UI.test_library_notes_trash_pages_and_purges_extended import (
    _pager,
    _painted_in,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_note_import_state import NoteImportPhase
from tldw_chatbook.Notes.note_import_executor import LocalNoteImportTarget
from tldw_chatbook.Widgets.Library.library_notes_canvas import PURGE_BUSY_NOTICE

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_x_under_an_import_keeps_focus_on_its_row_through_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``x`` under an import owes focus to its row until the import ends.

    Found live in the polish-1b finishing pass (2026-10-29, 120x36, 2 of 10
    runs, ``f1b-fin-run2-06``/``07a``): focus was on the last row's Restore
    when ``x`` painted the busy reason, and on the FIRST row's once the
    import's release cleared it. Painting and clearing that line are Notes
    syncs; one that captures while a recompose has focus parked on a pane
    grip (Textual's reset, before the replay) falls back to the Trash's owed
    role -- empty after the user's own key, so ``trash-restore:0``. The
    fenced ``x`` now owes the row it was pressed on, as the confirm's Cancel
    does (``_cancel_library_notes_trash_delete``).

    Deterministic: the window is entered on purpose -- focus parked on the
    grip, a programmatic move as Textual's reset is -- before the import is
    released. RED at ``498fd32a89``: focus lands on ``restore-0``.
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
            await pilot.click("#library-notes-trash-next")
            await _until(
                pilot,
                lambda: _pager(screen) == f"21–25 of {DELETED} · Page 2 of 2"
                and _focused_id(screen) == "library-notes-trash-previous",
                lambda: f"page 2 on Previous (focus: {_focused_id(screen)!r})",
            )
            # The live walk's keys: Shift+Tab to the last row, then x.
            await pilot.press("shift+tab")
            await _until(
                pilot,
                lambda: _focused_id(screen) == "library-notes-trash-restore-4",
                lambda: f"the last row's Restore (focus: {_focused_id(screen)!r})",
            )
            await pilot.press("x")
            await _until(
                pilot,
                lambda: _painted_in(host, "#library-notes-trash-notice")
                == PURGE_BUSY_NOTICE,
                "the busy reason painted",
            )
            assert host.screen is screen, "a confirm opened under the fence"
            assert _focused_id(screen) == "library-notes-trash-restore-4"

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
                    and _pager(screen) == f"21–25 of {DELETED} · Page 2 of 2"
                    and _focused_id(screen) == "library-notes-trash-restore-4",
                    lambda: (
                        "the reason cleared with focus still on x's row: "
                        f"focus={_focused_id(screen)!r} pager={_pager(screen)!r}"
                    ),
                )
                await pilot.pause(0.3)
            assert _focused_id(screen) == "library-notes-trash-restore-4"
    finally:
        write.release.set()
        seed.profile.interop.close_all_user_connections()
        seed.profile.db.close_connection()
