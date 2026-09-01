"""Tests for SyncEngine pull/push/reconcile behavior."""

import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientNotFoundError,
    ServerUnavailableError,
)
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
    server_client.list_reminders.return_value = {"items": [{"id": "srv-1"}]}
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
        payload={
            "action": "create",
            "fields": {"title": "Local", "schedule_kind": "one_time"},
        },
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
    server_client.update_reminder.return_value = {"id": "srv-1"}

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    # The pending local mutation is preserved as a conflict, so the server
    # state is not applied to the local row until the conflict is resolved.
    local_row = db.get_reminder_task(local_id)
    assert local_row["title"] == "Local"

    conflicts = db.get_conflicts("server:1", primitive="reminder_task")
    assert len(conflicts) == 1
    assert conflicts[0]["local_id"] == local_id
    assert conflicts[0]["server_state"]["title"] == "Server Newer"

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


@pytest.mark.asyncio
async def test_sync_pushes_update_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="Old",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.update_reminder_task(local_id, title="Updated")
    db.record_pending_mutation(
        local_id=local_id,
        primitive="reminder_task",
        owner_id="server:1",
        payload={"action": "update", "fields": {"title": "Updated"}},
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.update_reminder.return_value = {"id": "srv-1", "title": "Updated"}

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    server_client.update_reminder.assert_awaited_once_with(
        "srv-1",
        idempotency_key=server_client.update_reminder.call_args.kwargs[
            "idempotency_key"
        ],
        title="Updated",
    )

    local_row = db.get_reminder_task(local_id)
    assert local_row["title"] == "Updated"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_sync_pushes_delete_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="To Delete",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        local_id=local_id,
        primitive="reminder_task",
        owner_id="server:1",
        payload={"action": "delete"},
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.delete_reminder.return_value = {}

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    server_client.delete_reminder.assert_awaited_once_with("srv-1")
    assert db.get_reminder_task(local_id) is None
    assert (
        db.get_sync_mapping_by_local_id(local_id, "reminder_task", "server:1") is None
    )

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_resolve_conflict_server_wins(tmp_path):
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

    conflict = db.get_conflicts("server:1", primitive="reminder_task")[0]
    assert engine.resolve_conflict(conflict["id"], "server") is True

    resolved = db.get_conflict_by_id(conflict["id"])
    assert resolved["resolution"] == "server"
    assert resolved["resolved_at"] is not None

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_resolve_conflict_local_requeues_mutation(tmp_path):
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

    conflict = db.get_conflicts("server:1", primitive="reminder_task")[0]
    assert engine.resolve_conflict(conflict["id"], "local") is True

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "update"
    assert pending[0]["payload"]["fields"]["title"] == "Local Update"

    resolved = db.get_conflict_by_id(conflict["id"])
    assert resolved["resolution"] == "local"
    assert resolved["retry_count"] == 1


@pytest.mark.asyncio
async def test_pull_records_conflict_on_server_deletion(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="Local",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    conflicts = db.get_conflicts("server:1", primitive="reminder_task")
    assert len(conflicts) == 1
    assert conflicts[0]["local_id"] == local_id
    assert conflicts[0]["server_state"] == {}


@pytest.mark.asyncio
async def test_push_records_sync_error_on_server_unavailable(tmp_path):
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
        payload={
            "action": "create",
            "fields": {"title": "Local", "schedule_kind": "one_time"},
        },
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.side_effect = ServerUnavailableError("offline")

    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now()

    pending = db.get_pending_mutations("local", primitive="reminder_task")
    assert len(pending) == 1

    state = db.get_sync_state("local")
    assert any("offline" in err["message"] for err in state["sync_errors"])


@pytest.mark.asyncio
async def test_push_records_sync_error_on_generic_exception(tmp_path):
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
        payload={
            "action": "create",
            "fields": {"title": "Local", "schedule_kind": "one_time"},
        },
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.side_effect = RuntimeError("boom")

    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now()

    pending = db.get_pending_mutations("local", primitive="reminder_task")
    assert len(pending) == 1

    state = db.get_sync_state("local")
    assert any("boom" in err["message"] for err in state["sync_errors"])


@pytest.mark.asyncio
async def test_push_tombstones_records_sync_error_on_server_unavailable(tmp_path):
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
    server_client.delete_reminder.side_effect = ServerUnavailableError("offline")

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    tombstones = db.get_tombstones("server:1", primitive="reminder_task")
    assert len(tombstones) == 1

    state = db.get_sync_state("server:1")
    # _push_tombstone returns None on retryable server errors; the phase aborts
    # and records a single sync error without the original exception text.
    assert state is not None
    assert len(state["sync_errors"]) >= 1


@pytest.mark.asyncio
async def test_push_tombstones_records_sync_error_on_generic_exception(tmp_path):
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
    server_client.delete_reminder.side_effect = RuntimeError("boom")

    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    tombstones = db.get_tombstones("server:1", primitive="reminder_task")
    assert len(tombstones) == 1

    state = db.get_sync_state("server:1")
    assert any("boom" in err["message"] for err in state["sync_errors"])


@pytest.mark.asyncio
async def test_push_tombstones_clears_local_only_tombstone(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local",
        title="Local Only",
        schedule_kind="one_time",
    )
    db.delete_reminder_task(local_id)
    db.record_tombstone(local_id, "reminder_task", "local")

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}

    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now()

    server_client.delete_reminder.assert_not_awaited()
    tombstones = db.get_tombstones("local", primitive="reminder_task")
    assert len(tombstones) == 0


@pytest.mark.asyncio
async def test_idempotency_key_stable_across_push_retries(tmp_path):
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
        payload={
            "action": "create",
            "fields": {"title": "Local", "schedule_kind": "one_time"},
        },
    )

    failing_client = AsyncMock()
    failing_client.list_reminders.return_value = {"items": []}
    failing_client.create_reminder.side_effect = ServerUnavailableError("offline")

    engine = SyncEngine(db, failing_client, owner_id="local")
    await engine.sync_now()

    pending = db.get_pending_mutations("local", primitive="reminder_task")
    assert len(pending) == 1
    stable_key = pending[0]["payload"]["idempotency_key"]
    assert stable_key

    succeeding_client = AsyncMock()
    succeeding_client.list_reminders.return_value = {"items": []}
    succeeding_client.create_reminder.return_value = {"id": "srv-1", "title": "Local"}

    engine.server_client = succeeding_client
    await engine.sync_now()

    succeeding_client.create_reminder.assert_awaited_once_with(
        idempotency_key=stable_key, title="Local", schedule_kind="one_time"
    )


@pytest.mark.asyncio
async def test_sync_now_uses_passed_owner_not_self_owner(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now("server:1")
    server_client.list_reminders.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_sync_error_appends_and_caps(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    engine = SyncEngine(db, None, owner_id="server:1")
    for i in range(12):
        engine._record_sync_error(f"err {i}")
    state = db.get_sync_state("server:1")
    assert len(state["sync_errors"]) == 10


@pytest.mark.asyncio
async def test_pull_conflict_when_local_pending_update_exists(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="Local",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Updated"}, "idempotency_key": "ik"},
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1", "title": "Server", "schedule_kind": "one_time"}]
    }
    server_client.update_reminder.return_value = {"id": "srv-1"}
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    conflicts = db.get_conflicts("server:1", primitive="reminder_task")
    assert len(conflicts) == 1
    row = db.get_reminder_task(local_id)
    assert row["title"] == "Local"  # server state not applied
    pending = db.get_pending_mutations("server:1")
    assert len(pending) == 0  # update was pushed successfully


@pytest.mark.asyncio
async def test_push_404_records_conflict_and_removes_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="T",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Updated"}, "idempotency_key": "ik"},
    )

    server_client = AsyncMock()
    server_client.update_reminder.side_effect = ServerClientNotFoundError("gone")
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    conflicts = db.get_conflicts("server:1", primitive="reminder_task")
    assert len(conflicts) == 1
    pending = db.get_pending_mutations("server:1")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_use_local_on_server_deletion_clears_server_id_and_requeues_create(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="T",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    conflict_id = db.record_conflict(
        local_id, "reminder_task", "server:1", server_state={}, local_state={"record": db.get_reminder_task(local_id)}
    )

    engine = SyncEngine(db, None, owner_id="server:1")
    engine.resolve_conflict(conflict_id, "local")

    row = db.get_reminder_task(local_id)
    assert row["server_id"] is None
    pending = db.get_pending_mutations("server:1")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"


@pytest.mark.asyncio
async def test_pull_does_not_push_mutations_or_tombstones(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="Local",
        schedule_kind="one_time",
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Updated"}, "idempotency_key": "ik"},
    )
    db.record_tombstone("deleted-local-id", "reminder_task", "server:1")

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.pull()

    server_client.update_reminder.assert_not_awaited()
    server_client.delete_reminder.assert_not_awaited()
    pending = db.get_pending_mutations("server:1")
    assert len(pending) == 1
    tombstones = db.get_tombstones("server:1", primitive="reminder_task")
    assert len(tombstones) == 1


@pytest.mark.asyncio
async def test_policy_refusal_is_not_recorded_as_sync_error(tmp_path):
    """task-2722: a runtime-mode policy refusal ("requires server mode") is a
    not-applicable outcome, not a sync failure. The old path persisted it as a
    standing sync error badge on local-only profiles."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientPolicyError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerClientPolicyError(
        "notifications.reminders.list.server requires server mode."
    )
    engine = SyncEngine(db, server_client, owner_id="local")

    await engine.pull()
    await engine.sync_now()

    state = db.get_sync_state("local") or {}
    assert not (state.get("sync_errors") or []), (
        f"policy refusal was recorded as a sync error: {state.get('sync_errors')}"
    )


@pytest.mark.asyncio
async def test_real_server_failure_still_records_sync_error(tmp_path):
    """Guard for task-2722: only policy refusals are exempt — genuine server
    failures must keep surfacing as sync errors."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerUnavailableError("offline")
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.pull()

    state = db.get_sync_state("server:1") or {}
    assert state.get("sync_errors"), "real failure no longer recorded"


# --- task-23105 review F3: sync_now reports what it actually did ----------


@pytest.mark.asyncio
async def test_sync_now_returns_ok_outcome_with_counts(tmp_path):
    from tldw_chatbook.Scheduling.services.sync_engine import SyncOutcome

    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [
            {"id": "srv-1", "title": "A"},
            {"id": "srv-2", "title": "B"},
        ]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")
    outcome = await engine.sync_now()
    assert outcome == SyncOutcome("ok", pulled=2, pushed=0)


@pytest.mark.asyncio
async def test_sync_now_counts_pushed_mutations(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Local", schedule_kind="one_time"
    )
    db.record_pending_mutation(
        local_id=local_id,
        primitive="reminder_task",
        owner_id="local",
        payload={
            "action": "create",
            "fields": {"title": "Local", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-1", "title": "Local"}
    engine = SyncEngine(db, server_client, owner_id="local")
    outcome = await engine.sync_now()
    assert outcome.status == "ok"
    assert outcome.pushed == 1


@pytest.mark.asyncio
async def test_sync_now_returns_not_applicable_without_server_client(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    engine = SyncEngine(db, server_client=None, owner_id="local")
    outcome = await engine.sync_now()
    assert outcome.status == "not_applicable"
    assert (outcome.pulled, outcome.pushed) == (0, 0)


@pytest.mark.asyncio
async def test_sync_now_returns_not_applicable_on_policy_refusal(tmp_path):
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientPolicyError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerClientPolicyError(
        "requires server mode"
    )
    engine = SyncEngine(db, server_client, owner_id="local")
    outcome = await engine.sync_now()
    assert outcome.status == "not_applicable"
    # Policy refusals are never persisted as sync errors (task-2722).
    state = db.get_sync_state("local") or {}
    assert not (state.get("sync_errors") or [])


@pytest.mark.asyncio
async def test_sync_now_returns_error_outcome_on_server_error(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerUnavailableError("boom")
    engine = SyncEngine(db, server_client, owner_id="local")
    outcome = await engine.sync_now()
    assert outcome.status == "error"
    assert "boom" in (outcome.error or "")
    state = db.get_sync_state("local") or {}
    assert state.get("sync_errors"), "the failure must still be recorded"


# ----------------------------------------------------------------------
# Automation definitions/results sync mirrors + review pushback
# (schedules-handoff PR-3, task 4)
# ----------------------------------------------------------------------


@pytest.fixture
def captured_logs():
    """Collect loguru records emitted during the test."""
    from loguru import logger

    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="DEBUG",
    )
    yield records
    logger.remove(sink_id)


def _empty_reminders_client() -> AsyncMock:
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    return server_client


def _definition_page(items, has_more=False):
    return {"items": items, "total": len(items), "has_more": has_more}


def _result_page(items, has_more=False):
    return {"items": items, "total": len(items), "has_more": has_more}


def _result_items(n, prefix="res"):
    return [
        {
            "id": f"{prefix}-{i}",
            "definition_id": "def-1",
            "run_id": "run-1",
            "kind": "finding",
            "title": f"Result {i}",
            "summary": "S",
            "dedupe_key": f"key-{prefix}-{i}",
            "review_state": "unread",
            "created_at": "2026-08-30T09:00:00+00:00",
            "updated_at": "2026-08-30T09:00:00+00:00",
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_sync_now_replays_review_mutation_and_clears_on_success(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1"
    )
    db.record_pending_mutation(
        result_id,
        "automation_result_review",
        "server:1",
        {"server_result_id": "srv-res-1", "review_state": "dismissed"},
    )

    server_client = _empty_reminders_client()
    server_client.review_automation_result.return_value = {"id": "srv-res-1"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.review_automation_result.assert_awaited_once_with(
        "srv-res-1", "dismissed", review_note=None
    )
    assert db.get_pending_mutations("server:1", primitive="automation_result_review") == []


@pytest.mark.asyncio
async def test_sync_now_review_mutation_not_found_clears_it(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1"
    )
    db.record_pending_mutation(
        result_id,
        "automation_result_review",
        "server:1",
        {"server_result_id": "srv-res-1", "review_state": "read"},
    )

    server_client = _empty_reminders_client()
    server_client.review_automation_result.side_effect = ServerClientNotFoundError(
        "retired"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_pending_mutations("server:1", primitive="automation_result_review") == []
    state = db.get_sync_state("server:1") or {}
    assert not (state.get("sync_errors") or []), "a retired result is not a sync error"


@pytest.mark.asyncio
async def test_sync_now_review_mutation_other_error_retains_it(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1"
    )
    db.record_pending_mutation(
        result_id,
        "automation_result_review",
        "server:1",
        {"server_result_id": "srv-res-1", "review_state": "read"},
    )

    server_client = _empty_reminders_client()
    server_client.review_automation_result.side_effect = ServerUnavailableError("offline")
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # task-23105-review F2: the reminder phase itself succeeded, but a
    # pushback phase failed -- that must surface as an error outcome, not
    # a clean "ok" beside a fresh error badge.
    assert outcome.status == "error"
    assert "offline" in (outcome.error or "")
    pending = db.get_pending_mutations("server:1", primitive="automation_result_review")
    assert len(pending) == 1, "the mutation must be left queued for retry"
    state = db.get_sync_state("server:1") or {}
    assert state.get("sync_errors"), "a genuine server failure must be recorded"


@pytest.mark.asyncio
async def test_sync_now_skips_review_fields_for_just_pushed_result_this_cycle(tmp_path):
    """Task 5 same-cycle echo (Qodo finding): `_replay_review_mutations`'
    pushed `server_result_id`s must thread into `_pull_results` as
    `skip_review_server_ids`. By the time this SAME sync's results pull
    runs, the pending mutation the pushback phase just cleared is already
    gone, so the pending-mutation guard alone can no longer protect this
    row. Without the pushed-this-cycle skip set, a results page that
    still echoes the pre-review server state (server write/read-path lag)
    would revert the review that was just pushed.
    """
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()

    # First sync: seed a server-mirrored, unread result locally.
    stale_item = _result_items(1)[0]
    server_client.list_automation_results.return_value = _result_page([stale_item])
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()
    local_id = db.list_automation_results("server:1")[0]["id"]

    # Review it locally with a pending mutation queued, mirroring what
    # SchedulingService.review_automation_result does.
    db.update_result_review(
        local_id, "dismissed", "handled", "user:1",
        pending_mutation={
            "local_id": local_id,
            "primitive": "automation_result_review",
            "owner_id": "server:1",
            "payload": {
                "server_result_id": stale_item["id"],
                "review_state": "dismissed",
                "review_note": "handled",
            },
        },
    )

    # Second sync: pushback succeeds (mutation cleared), but the results
    # page mock is left UNCHANGED -- still reporting "unread" -- to
    # simulate the server's own read path lagging its just-committed write
    # within this same round trip.
    server_client.review_automation_result.return_value = {"id": stale_item["id"]}
    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.review_automation_result.assert_awaited_once_with(
        stale_item["id"], "dismissed", review_note="handled"
    )
    assert db.get_pending_mutations("server:1", primitive="automation_result_review") == []
    refreshed = db.get_automation_result(local_id)
    assert refreshed["review_state"] == "dismissed", (
        "the same-cycle stale echo must not revert the review just pushed"
    )
    assert refreshed["review_note"] == "handled"


@pytest.mark.asyncio
async def test_sync_now_pulls_and_upserts_definitions(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_definitions.return_value = _definition_page(
        [{"id": "srv-def-1", "family": "recurring_question", "name": "Daily"}]
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    rows = db.list_automation_definitions(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["server_id"] == "srv-def-1"
    assert rows[0]["name"] == "Daily"


@pytest.mark.asyncio
async def test_sync_now_pages_definitions_until_has_more_false(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_definitions.side_effect = [
        _definition_page(
            [{"id": f"srv-def-{i}", "family": "recurring_question", "name": f"D{i}"}
             for i in range(50)],
            has_more=True,
        ),
        _definition_page(
            [{"id": "srv-def-50", "family": "recurring_question", "name": "D50"}],
            has_more=False,
        ),
    ]
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert server_client.list_automation_definitions.await_count == 2
    assert len(db.list_automation_definitions(owner_id="server:1")) == 51


@pytest.mark.asyncio
async def test_sync_now_definitions_pull_caps_at_max_pages_and_logs(
    tmp_path, captured_logs
):
    """F4: the definitions pull was an unbounded `while True` -- a server
    that always claims `has_more=True` must not spin forever."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_definitions.side_effect = [
        _definition_page(
            [{"id": f"srv-def-p{page}-{i}", "family": "recurring_question",
              "name": f"D{page}-{i}"} for i in range(50)],
            has_more=True,
        )
        for page in range(10)
    ]
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.sync_now()

    assert server_client.list_automation_definitions.await_count == 4  # _SYNC_MAX_PAGES
    assert len(db.list_automation_definitions(owner_id="server:1")) == 200
    assert any(
        "cap" in message.lower() and level == "INFO" for level, message in captured_logs
    )


@pytest.mark.asyncio
async def test_sync_now_pulls_and_upserts_results(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_results.return_value = _result_page(
        _result_items(2)
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    rows = db.list_automation_results("server:1")
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_sync_now_results_pull_stops_early_on_short_page(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_results.return_value = _result_page(
        _result_items(3), has_more=True  # fewer than the 50-page size wins
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.sync_now()

    assert server_client.list_automation_results.await_count == 1


@pytest.mark.asyncio
async def test_sync_now_results_pull_caps_at_max_pages_and_logs(tmp_path, captured_logs):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_results.side_effect = [
        _result_page(_result_items(50, prefix=f"p{page}"), has_more=True)
        for page in range(4)
    ]
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.sync_now()

    assert server_client.list_automation_results.await_count == 4  # _SYNC_MAX_PAGES
    assert len(db.list_automation_results("server:1", limit=1000)) == 200
    assert any(
        "cap" in message.lower() and level == "INFO" for level, message in captured_logs
    )


@pytest.mark.asyncio
async def test_sync_now_definitions_phase_failure_does_not_abort_results_phase(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_definitions.side_effect = ServerUnavailableError("down")
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # task-23105-review F2: a failed automation phase must surface as an
    # error outcome even though the reminder phase (and the later results
    # phase) still ran to completion -- results are still pulled, but the
    # workbench must not toast "Sync completed" next to a fresh error
    # badge (the controller ruled the prior "ok" pin a plan artifact).
    assert outcome.status == "error"
    assert "down" in (outcome.error or "")
    assert len(db.list_automation_results("server:1")) == 1
    state = db.get_sync_state("server:1") or {}
    assert state.get("sync_errors"), "the definitions-phase failure must be recorded"


@pytest.mark.asyncio
async def test_sync_now_definitions_policy_refusal_is_not_recorded_as_error(tmp_path):
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientPolicyError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.list_automation_definitions.side_effect = ServerClientPolicyError(
        "requires server mode"
    )
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert len(db.list_automation_results("server:1")) == 1  # later phase still ran
    state = db.get_sync_state("server:1") or {}
    assert not (state.get("sync_errors") or [])


@pytest.mark.asyncio
async def test_pull_gains_definitions_and_results_without_pushback(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1"
    )
    db.record_pending_mutation(
        result_id,
        "automation_result_review",
        "server:1",
        {"server_result_id": "srv-res-1", "review_state": "dismissed"},
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.list_automation_definitions.return_value = _definition_page(
        [{"id": "srv-def-1", "family": "recurring_question", "name": "Daily"}]
    )
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.pull()

    assert len(db.list_automation_definitions(owner_id="server:1")) == 1
    # 2 rows: the pre-existing local-only result plus the server-pulled one.
    rows = db.list_automation_results("server:1", limit=10)
    assert len(rows) == 2
    assert any(row["server_id"] == "res-0" for row in rows)
    server_client.review_automation_result.assert_not_awaited()
    # The pending review mutation is untouched -- pull() never pushes.
    assert len(
        db.get_pending_mutations("server:1", primitive="automation_result_review")
    ) == 1


# --- review round 1 #1: reminder-phase failures must not short-circuit the
# automation phases in either entry point --------------------------------


def _break_apply_pulled_reminders(db: ScheduledTasksDB, message: str = "boom") -> None:
    """Force the reminder DB-transaction phase to fail (test-only)."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError(message)

    db._apply_pulled_reminders = _raise  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_pull_reminder_network_failure_still_pulls_automation_results(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerUnavailableError("down")
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.pull()

    assert len(db.list_automation_results("server:1")) == 1
    state = db.get_sync_state("server:1") or {}
    assert state.get("sync_errors"), "the reminder network failure must still be recorded"


@pytest.mark.asyncio
async def test_pull_reminder_transaction_failure_still_pulls_automation_results(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    _break_apply_pulled_reminders(db)
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1", "title": "A"}]
    }
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.pull()

    assert len(db.list_automation_results("server:1")) == 1
    state = db.get_sync_state("server:1") or {}
    assert state.get(
        "sync_errors"
    ), "the reminder transaction failure must still be recorded"


@pytest.mark.asyncio
async def test_sync_now_reminder_network_failure_still_pulls_automation_results(
    tmp_path,
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerUnavailableError("down")
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # The reminder phase's own outcome/status semantics are unchanged.
    assert outcome.status == "error"
    assert "down" in (outcome.error or "")
    assert len(db.list_automation_results("server:1")) == 1


@pytest.mark.asyncio
async def test_sync_now_reminder_transaction_failure_still_pulls_automation_results(
    tmp_path,
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    _break_apply_pulled_reminders(db)
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-1", "title": "A"}]
    }
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "error"
    assert "boom" in (outcome.error or "")
    assert len(db.list_automation_results("server:1")) == 1


# --- review round 2: "not_applicable" reminder phase must not mask a ------
# --- genuinely-failed automation phase -------------------------------------


@pytest.mark.asyncio
async def test_sync_now_reminder_policy_refusal_does_not_mask_automation_error(
    tmp_path,
):
    """The reminder phase's own policy action can be refused
    (`not_applicable`) while a DIFFERENT policy action -- an automation
    phase -- genuinely fails. The old `status == "ok"` guard only caught
    this when the reminder phase was "ok"; "not_applicable" let the
    automation failure through unreported."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientPolicyError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerClientPolicyError(
        "requires server mode"
    )
    server_client.list_automation_definitions.side_effect = ServerUnavailableError(
        "defs down"
    )
    server_client.list_automation_results.return_value = _result_page(_result_items(1))
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "error"
    assert "defs down" in (outcome.error or "")
    assert len(db.list_automation_results("server:1")) == 1  # later phase still ran
    state = db.get_sync_state("server:1") or {}
    assert state.get("sync_errors"), "the definitions-phase failure must be recorded"


@pytest.mark.asyncio
async def test_sync_now_all_policy_refusal_stays_not_applicable(tmp_path):
    """A pure all-refusal round (every phase's policy action refused) is
    still `not_applicable`, not `error` -- refusals are never errors."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientPolicyError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.side_effect = ServerClientPolicyError(
        "requires server mode"
    )
    server_client.list_automation_definitions.side_effect = ServerClientPolicyError(
        "requires server mode"
    )
    server_client.list_automation_results.side_effect = ServerClientPolicyError(
        "requires server mode"
    )
    engine = SyncEngine(db, server_client, owner_id="local")

    outcome = await engine.sync_now()

    assert outcome.status == "not_applicable"
    state = db.get_sync_state("local") or {}
    assert not (state.get("sync_errors") or [])
