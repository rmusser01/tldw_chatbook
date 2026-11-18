"""Hidden Home content work must finish its current read without issuing more."""

import asyncio
import inspect
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace

import pytest

from tldw_chatbook.Home.dashboard_state import HomeContentSnapshot
from tldw_chatbook.UI.Screens.home_screen import HomeScreen

pytestmark = pytest.mark.bootstrap_profile


def _owner(held=None):
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def read(name, result):
        def callback(**_kwargs):
            calls.append(name)
            if name == held:
                entered.set()
                try:
                    assert release.wait(5), "held read was not released"
                finally:
                    retired.set()
            return result

        return callback

    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(is_memory_db=False),
        notes_scope_service=SimpleNamespace(
            count_notes=read("count", 7), list_notes=read("notes", [])
        ),
        chat_conversation_scope_service=SimpleNamespace(
            list_conversations=read("conversations", {"pagination": {"total": 3}})
        ),
        media_reading_scope_service=SimpleNamespace(
            list_media_items=read("media", {"pagination": {"total_items": 2}})
        ),
    )
    previous = HomeContentSnapshot(note_count=11)
    owner = SimpleNamespace(
        is_mounted=True,
        is_attached=True,
        app=app,
        app_instance=app,
        _home_content_snapshot=previous,
        _home_console_model_state=read("readiness", (True, False)),
        _sync_home_triage_after_refresh=lambda: calls.append("paint"),
    )
    app.screen = owner
    for name in ("_build_home_content_snapshot", "_home_content_seam_call"):
        setattr(owner, name, MethodType(getattr(HomeScreen, name), owner))
    return owner, calls, entered, release, retired


async def _refresh(owner):
    await inspect.unwrap(HomeScreen._refresh_home_content_snapshot)(owner)


@pytest.mark.asyncio
async def test_hidden_home_does_not_start_content_work_or_replace_last_snapshot():
    owner, calls, *_ = _owner()
    previous = owner._home_content_snapshot
    owner.app.screen = object()
    await _refresh(owner)
    assert calls == []
    assert owner._home_content_snapshot is previous


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "held", ["readiness", "count", "notes", "conversations", "media"]
)
async def test_suspend_finishes_issued_read_then_skips_followups_and_revisit_refreshes(
    held,
):
    owner, calls, entered, release, retired = _owner(held)
    previous = owner._home_content_snapshot
    refresh = asyncio.create_task(_refresh(owner))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        owner.app.screen = object()
        await asyncio.sleep(0)
        assert not refresh.done()
        assert not retired.is_set()
        release.set()
        await asyncio.wait_for(refresh, 5)
        expected = ["readiness", "count", "notes", "conversations", "media"]
        assert calls == expected[: expected.index(held) + 1]
        assert retired.is_set()
        assert owner._home_content_snapshot is previous

        owner.app.screen = owner
        calls.clear()
        await _refresh(owner)
        assert calls == expected + ["paint"]
        assert owner._home_content_snapshot.note_count == 7
        assert owner._home_content_snapshot.conversation_count == 3
        assert owner._home_content_snapshot.media_count == 2
    finally:
        release.set()
        await refresh


@pytest.mark.asyncio
async def test_unmounted_snapshot_builder_retains_direct_caller_behavior():
    owner, calls, *_ = _owner()
    owner.is_mounted = False
    owner.app.screen = object()
    await _refresh(owner)
    assert calls == ["readiness", "count", "notes", "conversations", "media", "paint"]
    assert owner._home_content_snapshot.note_count == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_failed", [False, True])
async def test_native_home_notes_batch_preserves_count_and_retires_connection(
    tmp_path, monkeypatch, preview_failed
):
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.DB import ChaChaNotes_DB as notes_db
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

    database = notes_db.CharactersRAGDB(tmp_path / "home.db", "profile")
    assert database.add_note("Recent note", "Home preview")
    database.close_connection()
    local = NotesInteropService(tmp_path, "home", database)
    owner, *_ = _owner()
    owner.app_instance.chachanotes_db = database
    owner.app_instance.notes_scope_service = NotesScopeService(local, None)
    opened = []
    original_open = notes_db.connect_private_sqlite

    def observe_open(store, path, **kwargs):
        if Path(path) == database.db_path:
            opened.append(threading.get_ident())
        return original_open(store, path, **kwargs)

    monkeypatch.setattr(notes_db, "connect_private_sqlite", observe_open)
    if preview_failed:

        def unavailable_preview(*args, **kwargs):
            raise OSError("preview unavailable")

        monkeypatch.setattr(notes_db.CharactersRAGDB, "list_notes", unavailable_preview)

    try:
        await _refresh(owner)
        snapshot = owner._home_content_snapshot
        assert snapshot.note_count == 1
        assert snapshot.resume_title == ("" if preview_failed else "Recent note")
        assert opened and threading.get_ident() not in opened
        if not preview_failed:
            assert len(opened) == 1
        with storage_admission._lock:
            assert not any(
                lease.resource_path == database.db_path
                for lease in storage_admission._live_leases
            )
    finally:
        local.close_all_user_connections()
        database.close_connection()
