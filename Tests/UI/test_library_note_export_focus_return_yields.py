"""The export's focus return never takes focus back from the user (PR #3423 m-3).

TASK-34000.29 (N-12) returns focus to the control that started a Notes
transfer once it settles, so a finished export does not leave focus on
Delete. Dev's ``ba5fdc5d46`` moved the export's write into a worker
(``asyncio.to_thread``), so the finish now lands some time after the file
dialog closed -- and in Info nothing captures or replays a focus the user
moved meanwhile (the apply patches in place). The return took focus back
from wherever the user had gone: a user who clicked into Info's keywords
field while a slow export ran had focus jump to Export Markdown, so the next
Enter re-opened the picker.

The return now moves focus only while it is on nothing, on Textual's reset
landing, or still on the origin. One boot of the real ``LibraryScreen`` with
the production stylesheet over a real ChaChaNotes database. The export's
destination write is held on an event (never a sleep) while the user clicks
into Info's keywords field -- the one text field Info shows; the note body is
not on screen while Info is open -- and then released. The file is written,
the completion is painted, and focus is where the user put it. The returns
after a cancelled export, a written one and a Copy stay pinned in
``test_library_note_editor_feedback_is_honest.py``.

Base (``bd01c0e46b``): focus jumps to Export Markdown once the write ends.
"""

from __future__ import annotations

import threading

import pytest

from Tests.UI.library_quit_guard_support import SIZE, _open_note
from Tests.UI.test_library_note_editor_feedback_is_honest import (
    EXPORT_MD,
    IDEAS,
    _back_on_screen,
    _export_dialog,
    _focused_id,
    _info,
    _library,
    _mounted,
    _painted,
    _wait,
)
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput
from tldw_chatbook.UI.Library_Modules import library_file_export

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

KEYWORDS = "library-note-context-keywords"

#: The longest the held write waits for its release; the test releases it
#: as soon as the user's click has landed.
_HOLD_SECONDS = 15.0


async def test_a_focus_the_user_moved_during_an_export_stays_where_they_put_it(
    tmp_path, monkeypatch
) -> None:
    host, seeded, _toasts, _copies = _mounted(tmp_path, monkeypatch)
    destination = tmp_path / "exports" / "ideas.md"
    destination.parent.mkdir()
    entered = threading.Event()
    release = threading.Event()
    real_write = library_file_export.write_export_text

    def held_write(*args, **kwargs):
        entered.set()
        if not release.wait(_HOLD_SECONDS):
            raise TimeoutError("the held export write was never released")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(library_file_export, "write_export_text", held_write)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)
        await _info(host, screen, pilot)

        await pilot.click(f"#{EXPORT_MD}")
        dialog = await _export_dialog(host, pilot)
        await pilot.press("ctrl+a")
        dialog.query_one(FileNameInput).value = str(destination)
        await pilot.press("enter")
        await _back_on_screen(host, screen, pilot)
        await _wait(pilot, entered.is_set, "the export's write to start")

        # While the write runs: the user clicks into Info's keywords field.
        await pilot.click(f"#{KEYWORDS}")
        await _wait(
            pilot,
            lambda: _focused_id(screen) == KEYWORDS,
            lambda: f"focus in the keywords field (focus: #{_focused_id(screen)})",
        )
        assert not destination.exists()

        release.set()
        await _wait(
            pilot,
            lambda: destination.exists() and "Export complete" in _painted(host),
            "the export to write and report",
        )
        # The finish's return (and any retry it schedules) has run.
        for _ in range(3):
            await pilot.pause()
        assert _focused_id(screen) == KEYWORDS, (
            f"the export's return took focus from the keywords field to "
            f"#{_focused_id(screen)}"
        )
