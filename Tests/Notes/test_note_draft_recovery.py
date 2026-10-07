"""Exact-source, private and revision-safe draft recovery."""

from dataclasses import replace

import pytest

from tldw_chatbook.Library.library_notes_state import DatabaseNoteDraft
from tldw_chatbook.Notes.note_draft_recovery import NoteDraftRecovery, NoteDraftScope
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore


@pytest.mark.asyncio
async def test_restart_exact_empty_draft_and_matching_acknowledgement(tmp_path):
    path = tmp_path / "notes.db"
    scope = NoteDraftScope("profile", "alice", None)
    store = NotesDeviceStateStore(path)
    recovery = NoteDraftRecovery(store)
    original = DatabaseNoteDraft("n", "Title", "<>&\n雪", "", 1, "future-mode")
    await recovery.save(scope, original, 3)
    empty = replace(original, body="", revision=2, content_language="yaml")
    await recovery.save(scope, empty, 3)
    await recovery.acknowledge(scope, "n", 1)
    store.close()
    reopened = NotesDeviceStateStore(path)
    recovery = NoteDraftRecovery(reopened)
    assert await recovery.load(scope, "n") == (empty, 3)
    assert path.stat().st_mode & 0o077 == 0
    for other in (
        NoteDraftScope("other", "alice", None),
        NoteDraftScope("profile", "bob", None),
        NoteDraftScope("profile", "alice", "workspace"),
    ):
        assert await recovery.load(other, "n") is None
    await recovery.acknowledge(scope, "n", 2)
    assert await recovery.load(scope, "n") is None
    reopened.close()


@pytest.mark.asyncio
async def test_actual_v2_device_schema_upgrade(tmp_path):
    import sqlite3

    from tldw_chatbook.Notes import notes_device_state_schema as schema

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        for statement in (
            *schema.HISTORICAL_V1_IMPORT_TABLE_STATEMENTS,
            *schema.HISTORICAL_V1_IMPORT_INDEX_STATEMENTS,
            *schema._LASTING_TABLE_STATEMENTS,
            *schema._LASTING_INDEX_STATEMENTS,
        ):
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 2")
    store = NotesDeviceStateStore(path)
    recovery = NoteDraftRecovery(store)
    scope = NoteDraftScope("profile", "alice", None)
    draft = DatabaseNoteDraft("n", "Title", "", "", 1, "json")
    await recovery.save(scope, draft, 1)
    assert await recovery.load(scope, "n") == (draft, 1)
    with store.transaction() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert (
            connection.execute("SELECT count(*) FROM notes_sync_roots").fetchone()[0]
            == 0
        )
    store.close()
