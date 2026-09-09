"""Opening a stored Library note must finish (critique #8, task-32050).

Both tests drive the production wiring -- a real ``CharactersRAGDB``, a real
``NotesInteropService``/``NotesScopeService`` and therefore the real
``_LibraryDatabaseNoteSessionPort`` -- because the defect lived in the
screen's supersession guards, not in any service: a fake scope service
returns detail just as fast and hides it.
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
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

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
async def test_existing_note_opens_from_the_list_through_the_real_port(tmp_path):
    app, note_id = _real_notes_app(tmp_path)
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
    tmp_path, monkeypatch
):
    app, _note_id = _real_notes_app(tmp_path)
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
