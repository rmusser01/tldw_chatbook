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
