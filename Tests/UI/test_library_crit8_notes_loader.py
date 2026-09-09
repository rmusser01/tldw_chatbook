"""Opening a stored Library note must finish (critique #8, task-32050).

The first two tests drive the production wiring -- a real ``CharactersRAGDB``,
a real ``NotesInteropService``/``NotesScopeService`` and therefore the real
``_LibraryDatabaseNoteSessionPort`` -- because the defect lived in the
screen's supersession guards, not in any service: a fake scope service
returns detail just as fast and hides it. The last two isolate the load
deadline itself against a stubbed session (see the comment above them).
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import TextArea

from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTE_LOAD_TIMEOUT_COPY,
)
from tldw_chatbook.UI.Library_Modules.screen_support_types import (
    LibraryEntryReconcileResult,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module

NOTE_TITLE = "Reading list"
NOTE_BODY = "- Attention Is All You Need\n"


def _real_notes_app(tmp_path, *, title: str = NOTE_TITLE, body: str = NOTE_BODY):
    """Build the harness app on the production Notes wiring plus one note."""
    db = CharactersRAGDB(tmp_path / "crit8-notes.db", client_id="crit8")
    repository = LocalNoteFolderRepository(db)
    note_id = db.add_note(title, body)
    assert note_id is not None

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.chachanotes_db = db
    notes_service = NotesInteropService(tmp_path, "crit8", global_db_to_use=db)
    app.notes_service = notes_service
    app.notes_scope_service = NotesScopeService(
        notes_service,
        None,
        folder_repository=repository,
    )
    return app, note_id


@pytest.fixture
def real_notes_app(tmp_path):
    """Own the real database handle so every run closes it deterministically."""
    app, note_id = _real_notes_app(tmp_path)
    try:
        yield app, note_id
    finally:
        app.chachanotes_db.close_connection()


def _note_rows(screen) -> dict[str, object]:
    """Map note id to its currently mounted list row."""
    return {
        str(getattr(row, "note_id", "") or ""): row
        for row in screen.query(".library-notes-row")
        if str(getattr(row, "note_id", "") or "")
    }


async def _open_notes_list(host, pilot):
    """Select the Notes browse row and wait for the stored note rows."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
    for _ in range(200):
        if _note_rows(screen):
            return screen
        await pilot.pause(0.02)
    raise AssertionError("the stored note never appeared in the Notes list")


async def _click_note_row(pilot, screen, note_id: str) -> None:
    """Click one note row the way a user does, once the list has settled."""
    for _ in range(200):
        rows = _note_rows(screen)
        if note_id in rows:
            # Let the canvas finish any pending recompose, then re-query:
            # clicking a widget the list has already replaced hits nothing.
            await pilot.pause()
            await pilot.pause()
            rows = _note_rows(screen)
            if note_id in rows:
                await pilot.click(f"#{rows[note_id].id}")
                return
        await pilot.pause(0.02)
    raise AssertionError(f"note row {note_id!r} never settled in the Notes list")


@pytest.mark.asyncio
async def test_existing_note_opens_from_the_list_through_the_real_port(real_notes_app):
    app, note_id = real_notes_app
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_notes_list(host, pilot)
        await _click_note_row(pilot, screen, note_id)

        for _ in range(150):  # 3 s at 20 ms
            await pilot.pause(0.02)
            editors = screen.query("#library-note-body")
            if editors and NOTE_BODY.strip() in editors.first(TextArea).text:
                break
        else:
            pytest.fail(
                "note editor never rendered the stored body; load state is "
                f"{screen._notes_state.load_state!r}"
            )
        assert screen._notes_state.selected_note_id == note_id
        assert screen._notes_state.load_state == "loaded"


@pytest.mark.asyncio
async def test_a_stuck_note_load_fails_with_retry_and_the_next_note_still_opens(
    real_notes_app, monkeypatch
):
    app, _note_id = real_notes_app
    second_body = "- Second note body\n"
    second_id = app.chachanotes_db.add_note("Second note", second_body)
    assert second_id is not None
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_notes_list(host, pilot)
        port = screen._library_note_session._port
        original_load_note = port.load_note

        async def stalled_load_note(note_id: str):
            if note_id == second_id:
                return await original_load_note(note_id)
            await asyncio.sleep(10)
            raise AssertionError("the stalled load should never complete")

        monkeypatch.setattr(port, "load_note", stalled_load_note)

        await _click_note_row(pilot, screen, _note_id)
        for _ in range(200):  # 4 s at 20 ms
            await pilot.pause(0.02)
            if screen._notes_state.load_state == "failed":
                break
        else:
            pytest.fail("a stuck note load never reached the failed state")
        assert screen._notes_state.load_message == (
            "Unable to load note — timed out after 3 s. Press Retry."
        )
        for _ in range(100):
            await pilot.pause(0.02)
            if screen.query("#library-note-load-retry"):
                break
        else:
            pytest.fail("the failed note load offered no Retry button")

        if second_id not in _note_rows(screen):
            await screen.action_library_note_editor_back()

        await _click_note_row(pilot, screen, second_id)
        for _ in range(150):
            await pilot.pause(0.02)
            editors = screen.query("#library-note-body")
            if editors and second_body.strip() in editors.first(TextArea).text:
                break
        else:
            pytest.fail("a later note did not open after a timed-out load")


# --- Focused cover for the deadline branch itself (Qodo review, PR #2519) ----
#
# The two tests above drive the real port and real database end to end, which
# is what pinned the supersession defect. These two isolate the deadline
# branch: the session is stubbed and the deadline shortened, so the failed
# state, its retryable copy, and the ownership guard are each provable
# without the database or the note list in the way.


async def _open_seeded_note_editor(host, pilot):
    """Open ``n-1`` from the seeded (non-database) Notes fixtures."""
    screen = await _open_notes_list(host, pilot)
    await _click_note_row(pilot, screen, "n-1")
    for _ in range(200):
        await pilot.pause(0.02)
        if (
            screen._notes_state.selected_note_id == "n-1"
            and screen._notes_state.view == "editor"
            and screen._notes_state.load_state == "loaded"
        ):
            return screen
    raise AssertionError(
        "the seeded note never opened; load state is "
        f"{screen._notes_state.load_state!r}"
    )


def _stall_note_session(monkeypatch, screen, *, deadline: float = 0.05) -> None:
    """Hang the next session open, with a deadline short enough to test."""
    monkeypatch.setattr(
        library_screen_module, "LIBRARY_NOTE_LOAD_DEADLINE_SECONDS", deadline
    )

    async def never_opens(note_id: str):
        await asyncio.sleep(3600)
        raise AssertionError("the stalled session should never open")

    monkeypatch.setattr(screen._library_note_session, "open_session", never_opens)


@pytest.mark.asyncio
async def test_a_timed_out_load_fails_the_note_the_user_is_still_on(monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_seeded_note_editor(host, pilot)
        _stall_note_session(monkeypatch, screen)

        await screen._refresh_library_note_detail("n-1")

        assert screen._notes_state.load_state == "failed"
        assert screen._notes_state.load_message == LIBRARY_NOTE_LOAD_TIMEOUT_COPY


@pytest.mark.asyncio
async def test_a_timed_out_load_leaves_the_destination_the_user_moved_to(monkeypatch):
    """A deadline that expires after the user routed away owns nothing.

    The identity guard alone does not catch this: moving the Library rail to
    Media leaves the note id, the editor view and the database source intact,
    so only the captured entry route key separates "still mine" from "the
    user is somewhere else now".
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_seeded_note_editor(host, pilot)
        settled_state = screen._notes_state.load_state
        _stall_note_session(monkeypatch, screen)

        load = asyncio.create_task(
            screen._refresh_library_note_detail("n-1", entry_origin=True)
        )
        await pilot.pause()
        # The user routes away while the load is still stuck on its deadline.
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA

        assert await load is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._notes_state.load_state == settled_state
        assert screen._notes_state.load_message != LIBRARY_NOTE_LOAD_TIMEOUT_COPY
