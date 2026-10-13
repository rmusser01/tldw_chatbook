"""A real guarded quit keeps the newest unsaved source through store reopening."""

import asyncio
import inspect
from datetime import UTC, datetime

import pytest
from textual.widgets import Input, Select, Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _TITLE,
    SIZE,
    _ctrl_q,
    _library,
    _library_app,
    _no_autosave,
    _open_note,
    _settings_without_splash,
    _type,
    _until,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_notes_session import DatabaseNoteSessionCoordinator
from tldw_chatbook.Notes.note_draft_recovery import NoteDraftRecovery
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Library_Modules.note_session_port import (
    _LibraryDatabaseNoteSessionPort,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize(
    "source,refusal",
    [("", "validation"), ('{"latest": true}\n', "write")],
    ids=["empty", "newest"],
)
async def test_quit_recovers_newest_draft_after_an_older_retention_is_pending(
    tmp_path, monkeypatch, source, refusal
):
    _no_autosave(monkeypatch)
    events = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    entered, release = asyncio.Event(), asyncio.Event()
    database_path = profile.db.db_path_str
    principal = app.notes_user_id
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                body = await _open_note(screen, pilot, _TITLE, profile.note_id)
                session = screen._library_note_session
                recovery, scope = session._recovery, session._recovery_scope
                assert recovery is not None and scope is not None
                state_path = recovery._store.db_path
                save = recovery.save

                async def pause_first(scope, draft, base_version):
                    if not entered.is_set():
                        entered.set()
                        await release.wait()
                    await save(scope, draft, base_version)

                monkeypatch.setattr(recovery, "save", pause_first)
                title = screen.query_one("#library-note-title", Input)
                title.focus()
                title.cursor_position = len(title.value)
                if refusal == "validation":
                    await _type(pilot, " ")
                else:
                    screen.query_one("#library-note-keywords", Input).value = "queued"

                    def write_failure(**kwargs):
                        raise OSError("Synthetic canonical write failure")

                    monkeypatch.setattr(
                        profile.scope_service, "save_note", write_failure
                    )
                draft_title = _TITLE + " " if refusal == "validation" else _TITLE
                await asyncio.wait_for(entered.wait(), 2)
                older_revision = session.snapshot.draft_revision
                body.focus()
                body.action_select_all()
                await pilot.press("backspace")
                await _type(pilot, source.rstrip("\n"))
                if source.endswith("\n"):
                    await pilot.press("enter")
                screen.query_one("#library-note-document-tools").query_one(
                    Select
                ).value = "json"
                screen.query_one("#library-note-keywords", Input).value = "new, tags"
                await _until(
                    pilot,
                    lambda: session.snapshot.body == source
                    and session.snapshot.draft.content_language == "json"
                    and session.snapshot.keywords_text == "new, tags",
                    "the latest canonical draft",
                )
                assert session.snapshot.draft_revision > older_revision
                release.set()
                assert await _ctrl_q(pilot, app, events) is None
                [(kind, canonical)] = events
                assert kind == "quit"
                assert (canonical["title"], canonical["content"]) == (_TITLE, _BODY)
                retained, base_version = await recovery.load(scope, profile.note_id)
                assert (
                    retained.title,
                    retained.body,
                    retained.content_language,
                    retained.keywords_text,
                ) == (draft_title, source, "json", "new, tags")
                assert retained.revision == session.snapshot.draft_revision
                assert base_version == canonical["version"]
                assert session.snapshot.dirty
                status = screen.query_one("#library-note-status", Static).renderable
                assert not str(status).strip().startswith("Saved"), status
    finally:
        release.set()
        profile.db.close_connection()

    # New native stores and the production port/coordinator take the relaunch path.
    db = CharactersRAGDB(database_path, client_id="reopened-retained-draft")
    notes = NotesInteropService(
        tmp_path, "reopened-retained-draft", global_db_to_use=db
    )
    state = NotesDeviceStateStore(state_path)

    async def call(method, *args, isolate_in_worker=False, **kwargs):
        result = method(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    port = _LibraryDatabaseNoteSessionPort(
        run_service_call=call,
        notes_scope_service=NotesScopeService(notes, None),
        notes_service=notes,
        user_id=principal,
        clock=lambda: datetime.now(UTC),
    )
    reopened = DatabaseNoteSessionCoordinator(
        port, recovery=NoteDraftRecovery(state), recovery_scope=scope
    )
    try:
        await reopened.open_session(profile.note_id)
        snapshot = reopened.snapshot
        assert snapshot is not None and snapshot.dirty
        assert (
            snapshot.title,
            snapshot.body,
            snapshot.draft.content_language,
            snapshot.keywords_text,
        ) == (draft_title, source, "json", "new, tags")
        assert (snapshot.baseline.title, snapshot.baseline.body, snapshot.version) == (
            _TITLE,
            _BODY,
            base_version,
        )
    finally:
        state.close()
        notes.close_all_user_connections()
        db.close_connection()
