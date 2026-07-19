"""Tests for SyncEngine pull/push/reconcile behavior."""

import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from tldw_chatbook.Scheduling.services.sync_engine import SyncEngine


@pytest.mark.asyncio
async def test_pull_inserts_server_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1", "title": "Server"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()
    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["title"] == "Server"
    assert rows[0]["server_id"] == "srv-1"


@pytest.mark.asyncio
async def test_pull_updates_existing_reminder_with_mapping(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-2",
        title="Old",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-2", "reminder_task", "server:1")

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-2", "title": "Updated"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["title"] == "Updated"


@pytest.mark.asyncio
async def test_pull_skips_when_no_server_client(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    engine = SyncEngine(db, server_client=None, owner_id="local")
    await engine.pull()
    rows = db.list_reminder_tasks(owner_id="local")
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_pull_records_last_pull_at(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    state = db.get_sync_state("server:1")
    assert state is not None
    assert state["last_pull_at"] is not None


@pytest.mark.asyncio
async def test_pull_skips_server_item_missing_id(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"title": "No id"}, {"id": "srv-1", "title": "Has id"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["server_id"] == "srv-1"


@pytest.mark.asyncio
async def test_pull_defaults_title_when_missing(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["title"] == "Untitled reminder"


@pytest.mark.asyncio
async def test_pull_defaults_title_when_empty(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1", "title": ""}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["title"] == "Untitled reminder"


@pytest.mark.asyncio
async def test_pull_inserts_multiple_server_reminders(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [
            {"id": "srv-1", "title": "First"},
            {"id": "srv-2", "title": "Second"},
        ]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 2
    titles = {row["title"] for row in rows}
    assert titles == {"First", "Second"}


@pytest.mark.asyncio
async def test_pull_creates_sync_mapping(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1", "title": "Mapped"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    local_id = rows[0]["id"]
    mapping = db.get_sync_mapping_by_server_id("srv-1", "reminder_task", "server:1")
    assert mapping is not None
    assert mapping["local_id"] == local_id


@pytest.mark.asyncio
async def test_pull_records_sync_errors_on_server_unavailable(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerUnavailableError("offline")
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 0

    state = db.get_sync_state("server:1")
    assert state is not None
    assert state["sync_errors"] is not None
    assert any("offline" in err["message"] for err in state["sync_errors"])


@pytest.mark.asyncio
async def test_pull_records_sync_errors_on_generic_exception(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = RuntimeError("boom")
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 0

    state = db.get_sync_state("server:1")
    assert state is not None
    assert state["sync_errors"] is not None
    assert any("boom" in err["message"] for err in state["sync_errors"])


@pytest.mark.asyncio
async def test_pull_updates_existing_reminder_without_mapping(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-orphan",
        title="Orphan",
        schedule_kind="one_time",
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-orphan", "title": "Recovered"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["id"] == local_id
    assert rows[0]["title"] == "Recovered"

    mapping = db.get_sync_mapping_by_server_id(
        "srv-orphan", "reminder_task", "server:1"
    )
    assert mapping is not None
    assert mapping["local_id"] == local_id


@pytest.mark.asyncio
async def test_sync_pushes_local_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local",
        title="Local",
        schedule_kind="one_time",
    )
    db.record_pending_mutation(
        local_id=local_id,
        primitive="reminder_task",
        owner_id="local",
        payload={"action": "create", "fields": {"title": "Local", "schedule_kind": "one_time"}},
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-1", "title": "Local"}

    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now()

    server_client.create_reminder.assert_awaited_once()
    local_row = db.get_reminder_task(local_id)
    assert local_row["server_id"] == "srv-1"

    pending = db.get_pending_mutations("local", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_sync_records_conflict_when_server_newer(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="Local",
        schedule_kind="one_time",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        local_id=local_id,
        primitive="reminder_task",
        owner_id="server:1",
        payload={"action": "update", "fields": {"title": "Local Update"}},
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [
            {
                "id": "srv-1",
                "title": "Server Newer",
                "schedule_kind": "one_time",
                "updated_at": "2026-07-19T00:00:00+00:00",
            }
        ]
    }

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    local_row = db.get_reminder_task(local_id)
    assert local_row["title"] == "Server Newer"

    conflicts = db.get_conflicts("server:1", primitive="reminder_task")
    assert len(conflicts) == 1
    assert conflicts[0]["local_id"] == local_id

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_sync_pushes_tombstone(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="To Delete",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.delete_reminder_task(local_id)
    db.record_tombstone(local_id, "reminder_task", "server:1")

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.delete_reminder.return_value = {}

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    server_client.delete_reminder.assert_awaited_once_with("srv-1")

    tombstones = db.get_tombstones("server:1", primitive="reminder_task")
    assert len(tombstones) == 0
