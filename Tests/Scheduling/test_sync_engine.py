"""Tests for SyncEngine pull/push/reconcile behavior."""

import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientNotFoundError,
    ServerClientValidationError,
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

    # UAT finding 3c (supersedes task-23105-review F2's "must surface as
    # an error outcome" ruling): the reminder phase itself succeeded, and
    # that must be reported honestly -- collapsing it to `status="error"`
    # is exactly the "Sync failed" toast the UAT caught lying over a
    # phase that had nothing to do with the failure. The pushback
    # failure still reaches the caller, as its own labeled entry.
    assert outcome.status == "ok"
    assert outcome.error is None
    assert len(outcome.phase_errors) == 1
    assert "Automation review pushback" in outcome.phase_errors[0]
    assert "offline" in outcome.phase_errors[0]
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


# ----------------------------------------------------------------------
# Definition create/update push replay (schedules-handoff PR-4, task 3)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_now_replays_definition_create_and_dedupes_same_cycle_pull(tmp_path):
    """Task 3 pull-ordering note: a successful create's server_id must land
    before the same-cycle definitions pull runs, so the pull matches the
    existing local row by (owner_id, server_id) instead of inserting a
    duplicate mirror."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "create",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Daily digest",
                # Client vocabulary (schedule_compute.py): `every_seconds`,
                # not the server's `seconds` -- the push must translate
                # this onto the wire (schedule_vocabulary.py, task 3).
                "schedule": {"kind": "interval", "every_seconds": 3600},
            },
            "server_definition_id": None,
        },
    )

    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "configured",
    }
    # Same-cycle pull returns the exact row that was just created.
    server_client.list_automation_definitions.return_value = _definition_page(
        [{"id": "srv-def-1", "family": "recurring_question", "name": "Daily digest"}]
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["mode"] == "create"
    assert "definition_id" not in request
    assert request["schedule"] == {"kind": "interval", "seconds": 3600}, (
        "schedule must be translated to server vocabulary on the wire"
    )
    server_client.create_automation_definition.assert_awaited_once_with(
        "prev-1", initial_lifecycle="configured"
    )
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )
    rows = db.list_automation_definitions(owner_id="server:1")
    assert len(rows) == 1, "create replay + same-cycle pull must yield exactly one row"
    assert rows[0]["id"] == local_id
    assert rows[0]["server_id"] == "srv-def-1"


@pytest.mark.asyncio
async def test_sync_now_definition_push_stamps_last_push_at(tmp_path):
    """UAT finding 3c: `_sync_reminders` used to be the ONLY writer of
    `last_push_at`, and only for its own reminder pushes -- a cycle that
    pushed nothing on the reminder side but DID push a definition create
    left the sync bar showing "Last push: -" forever, even though a push
    genuinely happened this cycle."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "create",
            "definition_payload": {"family": "recurring_question", "name": "Draft"},
            "server_definition_id": None,
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Draft",
        "lifecycle": "configured",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    assert not (db.get_sync_state("server:1") or {}).get("last_push_at")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert outcome.pushed == 0, "nothing was pushed on the REMINDER side this cycle"
    state = db.get_sync_state("server:1") or {}
    assert state.get("last_push_at"), (
        "a genuine definition push must stamp last_push_at even when the "
        "reminder phase itself pushed nothing"
    )


@pytest.mark.asyncio
async def test_sync_now_definition_noop_cycle_does_not_stamp_last_push_at(tmp_path):
    """The other half of the same fix: a cycle with no pending definition
    mutations at all must not fabricate a push timestamp."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    state = db.get_sync_state("server:1") or {}
    assert not state.get("last_push_at")


def test_definition_push_success_outcomes_cover_every_return_value():
    """Review round 1 finding 2: `_DEFINITION_PUSH_SUCCESS_OUTCOMES` is a
    hand-maintained set with no automatic tie to the string vocabulary
    `_push_definition_mutation` and its five helpers can actually return.
    A new outcome added to any of those six functions (or to
    `_replay_definition_mutations`'s own inline `"transfer_skipped"`)
    without a matching classification here used to be silent: nothing
    would fail, `last_push_at` would just never move for it. This walks
    the AST of every one of those function bodies and asserts every
    outcome string it can produce -- literal `return "..."` values, PLUS
    `_push_definition_lifecycle`'s two dynamic shapes (`return action`
    and `return f"{action}_not_found"`, both keyed on the three lifecycle
    actions) -- is classified in EXACTLY ONE of `_DEFINITION_PUSH_
    SUCCESS_OUTCOMES` / `_DEFINITION_PUSH_NON_SUCCESS_OUTCOMES`. An
    unclassified new outcome fails this test instead of silently
    freezing `last_push_at`.
    """
    import ast
    import inspect

    from tldw_chatbook.Scheduling.services import sync_engine as sync_engine_module
    from tldw_chatbook.Scheduling.services.sync_engine import (
        _DEFINITION_LIFECYCLE_ACTIONS,
        _DEFINITION_PUSH_NON_SUCCESS_OUTCOMES,
        _DEFINITION_PUSH_SUCCESS_OUTCOMES,
    )

    push_helper_names = {
        "_push_definition_mutation",
        "_push_definition_create",
        "_push_definition_update",
        "_push_definition_lifecycle",
        "_push_definition_release",
        "_push_definition_transfer",
    }

    source = inspect.getsource(sync_engine_module)
    tree = ast.parse(source)

    discovered: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AsyncFunctionDef) and node.name in push_helper_names
        ):
            continue
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Return) and sub.value is not None):
                continue
            value = sub.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                discovered.add(value.value)
            elif node.name == "_push_definition_lifecycle" and isinstance(
                value, ast.Name
            ):
                # `return action` -- dynamic, but only ever one of the
                # three lifecycle actions this handler dispatches on.
                discovered.update(_DEFINITION_LIFECYCLE_ACTIONS)
            elif node.name == "_push_definition_lifecycle" and isinstance(
                value, ast.JoinedStr
            ):
                # `return f"{action}_not_found"` -- same dynamic source.
                discovered.update(
                    f"{action}_not_found" for action in _DEFINITION_LIFECYCLE_ACTIONS
                )
            # A delegating `return await self._push_definition_X(...)`
            # (from `_push_definition_mutation`'s own dispatch) contributes
            # nothing itself -- the callee's own literal returns, walked
            # separately above, already cover it.

    # `_replay_definition_mutations`'s own pre-dispatch short-circuit --
    # not a `return`, a `counts["transfer_skipped"] = ...` dict write --
    # is a seventh real outcome this vocabulary must include.
    discovered.add("transfer_skipped")

    assert discovered, "the AST walk found nothing -- the helper names above drifted"

    known = _DEFINITION_PUSH_SUCCESS_OUTCOMES | _DEFINITION_PUSH_NON_SUCCESS_OUTCOMES
    unclassified = discovered - known
    assert not unclassified, (
        f"new/unclassified push outcome(s) {sorted(unclassified)} -- add each "
        "to _DEFINITION_PUSH_SUCCESS_OUTCOMES (if it means a mutation reached "
        "the server) or _DEFINITION_PUSH_NON_SUCCESS_OUTCOMES (if not) in "
        "sync_engine.py"
    )
    stale = _DEFINITION_PUSH_SUCCESS_OUTCOMES - discovered
    assert not stale, (
        f"_DEFINITION_PUSH_SUCCESS_OUTCOMES claims outcome(s) {sorted(stale)} "
        "that no longer exist in source -- prune it"
    )


@pytest.mark.asyncio
async def test_sync_now_definition_create_orphan_clears_mutation_and_reports_both_ids(
    tmp_path,
):
    """Qodo MEDIUM: the local row is deleted between queueing the create and
    replaying it, so `adopt_server_definition_identity` finds nothing to
    write and the fresh server definition has no local home.

    The mutation is still cleared -- replaying it would create a SECOND
    server definition, not recover the first -- and a sync error naming BOTH
    ids makes the orphan discoverable. (Deleting it server-side is a
    lifecycle action, out of scope here.)"""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "create",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Daily digest",
                # Client vocabulary (schedule_compute.py): `time_of_day`,
                # not the server's `at` -- the push must translate this
                # onto the wire (schedule_vocabulary.py, task 3).
                "schedule": {"kind": "daily", "time_of_day": "09:00"},
            },
            "server_definition_id": None,
        },
    )

    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.return_value = {
        "id": "srv-def-9",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "configured",
    }
    server_client.list_automation_definitions.return_value = _definition_page([])
    engine = SyncEngine(db, server_client, owner_id="server:1")

    # The row vanishes after the mutation is queued but before the replay.
    assert db.delete_automation_definition(local_id) is True

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["schedule"] == {"kind": "daily", "at": "09:00"}, (
        "schedule must be translated to server vocabulary on the wire"
    )
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    ), "replaying would create a duplicate server definition"
    state = db.get_sync_state("server:1")
    messages = [err["message"] for err in (state["sync_errors"] or [])]
    assert any(
        local_id in message and "srv-def-9" in message for message in messages
    ), f"the orphan error must name both ids; got {messages}"


@pytest.mark.asyncio
async def test_sync_now_definition_create_invalid_preview_clears_and_records_error(
    tmp_path,
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "create",
            "definition_payload": {"family": "recurring_question"},
            "server_definition_id": None,
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": None,
        "status": "invalid",
        "validation_errors": [
            {"field": "schedule.kind", "code": "required", "message": "Schedule kind is required."}
        ],
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.create_automation_definition.assert_not_awaited()
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    ), "a rejected payload will never succeed by retrying -- clear it"
    state = db.get_sync_state("server:1") or {}
    errors = state.get("sync_errors") or []
    assert errors
    assert "schedule.kind:required" in errors[-1]["message"]
    row = db.get_automation_definition(local_id)
    assert row["server_id"] is None


@pytest.mark.asyncio
async def test_sync_now_replays_definition_update(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Old name", server_id="srv-def-2"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "update",
            "definition_payload": {"family": "recurring_question", "name": "New name"},
            "server_definition_id": "srv-def-2",
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-2",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.update_automation_definition.return_value = {
        "id": "srv-def-2",
        "name": "New name",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["mode"] == "update"
    assert request["definition_id"] == "srv-def-2"
    server_client.update_automation_definition.assert_awaited_once_with(
        "srv-def-2", "prev-2"
    )
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )
    row = db.get_automation_definition(local_id)
    assert row["name"] == "New name"


@pytest.mark.parametrize("pause_queued_first", [False, True])
@pytest.mark.asyncio
async def test_sync_now_replays_a_queued_edit_and_pause_together(
    tmp_path, pause_queued_first
):
    """Qodo findings 5+6: an offline edit and an offline pause on the SAME
    row occupy different `pending_mutations` slots, so both survive being
    queued and both replay in one cycle -- whichever was queued first.

    The convergence hinge is the echo. `_replay_definition_mutations`
    settles the definition slot BEFORE the lifecycle slot, so when
    `adopt_server_definition_identity` applies the PATCH response (which
    here still reports the pre-pause `lifecycle` -- the server write/read
    lag this whole guard family exists for), the pause is still queued,
    and the adopt-identity belt withholds `lifecycle` on exactly that
    basis. Without that belt the row would end the cycle `configured`
    with nothing left to correct it."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Old name", server_id="srv-def-2"
    )

    def _queue_pause():
        db.record_pending_mutation(
            local_id,
            "automation_lifecycle",
            "server:1",
            {"action": "pause", "server_definition_id": "srv-def-2"},
        )

    def _queue_edit():
        db.record_pending_mutation(
            local_id,
            "automation_definition",
            "server:1",
            {
                "action": "update",
                "definition_payload": {
                    "family": "recurring_question",
                    "name": "New name",
                },
                "server_definition_id": "srv-def-2",
            },
        )

    for step in (
        (_queue_pause, _queue_edit)
        if pause_queued_first
        else (_queue_edit, _queue_pause)
    ):
        step()

    server_client = _empty_reminders_client()
    server_client.list_automation_definitions.return_value = _definition_page([])
    server_client.preview_automation_definition.return_value = {
        "id": "prev-2",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.update_automation_definition.return_value = {
        "id": "srv-def-2",
        "name": "New name",
        # Stale: the pause has not been pushed yet at echo time.
        "lifecycle": "configured",
    }
    server_client.pause_automation_definition.return_value = {
        "id": "srv-def-2",
        "family": "recurring_question",
        "name": "New name",
        "lifecycle": "paused",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.update_automation_definition.assert_awaited_once_with(
        "srv-def-2", "prev-2"
    )
    server_client.pause_automation_definition.assert_awaited_once_with("srv-def-2")
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []
    assert db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    row = db.get_automation_definition(local_id)
    assert row["name"] == "New name"
    assert row["lifecycle"] == "paused"


@pytest.mark.asyncio
async def test_sync_now_definition_update_without_server_id_converts_to_create(
    tmp_path,
):
    """Authored offline, never synced -- mirrors `_push_mutation`'s reminder
    offline-create-conversion precedent."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Offline draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "update",
            "definition_payload": {
                "family": "recurring_question", "name": "Offline draft"
            },
            "server_definition_id": None,
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-3",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.return_value = {"id": "srv-def-3"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.update_automation_definition.assert_not_awaited()
    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["mode"] == "create"
    server_client.create_automation_definition.assert_awaited_once_with(
        "prev-3", initial_lifecycle="configured"
    )
    row = db.get_automation_definition(local_id)
    assert row["server_id"] == "srv-def-3"


@pytest.mark.asyncio
async def test_sync_now_definition_update_not_found_converts_to_create(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Stale", server_id="srv-def-deleted"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "update",
            "definition_payload": {"family": "recurring_question", "name": "Stale"},
            "server_definition_id": "srv-def-deleted",
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.side_effect = [
        {"id": "prev-4", "status": "valid", "validation_errors": []},  # update preview
        {"id": "prev-5", "status": "valid", "validation_errors": []},  # create preview
    ]
    server_client.update_automation_definition.side_effect = ServerClientNotFoundError(
        "gone"
    )
    server_client.create_automation_definition.return_value = {"id": "srv-def-new"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.create_automation_definition.assert_awaited_once_with(
        "prev-5", initial_lifecycle="configured"
    )
    row = db.get_automation_definition(local_id)
    assert row["server_id"] == "srv-def-new"
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )


@pytest.mark.asyncio
async def test_sync_now_definition_create_retryable_error_retains_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "create",
            "definition_payload": {"family": "recurring_question"},
            "server_definition_id": None,
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # UAT finding 3c: the reminder phase's own "ok" is no longer masked
    # by this unrelated definition-push failure; the failure still
    # reaches the caller as its own labeled `phase_errors` entry.
    assert outcome.status == "ok"
    assert outcome.error is None
    assert len(outcome.phase_errors) == 1
    assert "Automation definition push" in outcome.phase_errors[0]
    assert "offline" in outcome.phase_errors[0]
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1, "the mutation must be left queued for retry"
    row = db.get_automation_definition(local_id)
    assert row["server_id"] is None


@pytest.mark.asyncio
async def test_sync_now_definition_poisoned_mutation_does_not_block_the_rest(tmp_path):
    """Final review I3: a non-retryable 4xx on ONE definition mutation used
    to raise out of the replay loop and abort the whole push phase, so one
    permanently-rejected payload stopped every other definition mutation
    for that owner forever. It must settle that mutation (cleared, error
    recorded) and carry on with the queue."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientValidationError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    poisoned_id = db.create_automation_definition(
        "server:1", "recurring_question", "Poisoned", server_id="srv-def-old"
    )
    db.record_pending_mutation(
        poisoned_id,
        "automation_definition",
        "server:1",
        {
            "action": "update",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Poisoned",
                "definition_version": 2,
            },
            "server_definition_id": "srv-def-old",
        },
    )
    healthy_id = db.create_automation_definition(
        "server:1", "recurring_question", "Healthy"
    )
    db.record_pending_mutation(
        healthy_id,
        "automation_definition",
        "server:1",
        {
            "action": "create",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Healthy",
            },
            "server_definition_id": None,
        },
    )

    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.update_automation_definition.side_effect = ServerClientValidationError(
        "scheduled_task_definition_version_conflict"
    )
    server_client.create_automation_definition.return_value = {
        "id": "srv-def-new",
        "family": "recurring_question",
        "name": "Healthy",
        "lifecycle": "configured",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    # The healthy create went through despite the poisoned update ahead of it.
    assert db.get_automation_definition(healthy_id)["server_id"] == "srv-def-new"
    # And the poisoned one is settled, not left to jam the queue forever.
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []
    errors = (db.get_sync_state("server:1") or {}).get("sync_errors") or []
    assert any("version_conflict" in error["message"] for error in errors)



# ----------------------------------------------------------------------
# Definition lifecycle mutation replay (pause/resume/archive) --
# schedules-handoff PR-5, task 2. Direct endpoint calls (no preview):
# success mirrors the response echo via `upsert_automation_definitions_
# from_server` and clears the mutation; `ServerClientNotFoundError` clears
# it with an info log (nothing left server-side to transition, no local
# edit to preserve by converting to a create like the update leg does);
# `ServerClientValidationError` settles via the existing per-mutation
# rejection path (final review I3 -- one poisoned mutation never blocks
# the rest); a retryable error leaves the mutation queued.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_now_replays_definition_pause_and_mirrors_echo(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Daily digest", server_id="srv-def-1"
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "pause", "server_definition_id": "srv-def-1"},
    )
    server_client = _empty_reminders_client()
    server_client.pause_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "paused",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.pause_automation_definition.assert_awaited_once_with("srv-def-1")
    assert (
        db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    )
    row = db.get_automation_definition(local_id)
    assert row["lifecycle"] == "paused"


@pytest.mark.asyncio
async def test_sync_now_replays_definition_resume(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1",
        "recurring_question",
        "Daily digest",
        server_id="srv-def-1",
        lifecycle="paused",
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "resume", "server_definition_id": "srv-def-1"},
    )
    server_client = _empty_reminders_client()
    server_client.resume_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "configured",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.resume_automation_definition.assert_awaited_once_with("srv-def-1")
    assert (
        db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    )
    assert db.get_automation_definition(local_id)["lifecycle"] == "configured"


@pytest.mark.asyncio
async def test_sync_now_replays_definition_archive(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Daily digest", server_id="srv-def-1"
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "archive", "server_definition_id": "srv-def-1"},
    )
    server_client = _empty_reminders_client()
    server_client.archive_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "archived",
        "archived_at": "2026-09-01T00:00:00+00:00",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.archive_automation_definition.assert_awaited_once_with("srv-def-1")
    assert (
        db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    )
    row = db.get_automation_definition(local_id)
    assert row["lifecycle"] == "archived"
    assert row["archived_at"] is not None


@pytest.mark.asyncio
async def test_sync_now_pull_skips_stale_lifecycle_echo_pushed_same_cycle(tmp_path):
    """PR-3 task 2, guard layer 2 end-to-end: the definitions push phase
    (which replays the pending `pause`) runs, then deletes the mutation,
    BEFORE the definitions pull phase runs in the same `sync_now` call.
    If the pull's page still echoes the pre-pause lifecycle (server
    write/read-path lag), guard 1 alone can't see it -- the mutation is
    already gone. `skip_lifecycle_server_ids`, threaded from the push
    phase's return value, is what stops the pull from reverting the
    pause that was just pushed THIS cycle."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Daily digest", server_id="srv-def-1"
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "pause", "server_definition_id": "srv-def-1"},
    )
    server_client = _empty_reminders_client()
    server_client.pause_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "paused",
    }
    # The same cycle's definitions-list page still echoes the PRE-pause
    # state.
    server_client.list_automation_definitions.return_value = _definition_page(
        [
            {
                "id": "srv-def-1",
                "family": "recurring_question",
                "name": "Daily digest (renamed)",
                "lifecycle": "configured",
            }
        ]
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    row = db.get_automation_definition(local_id)
    assert row["lifecycle"] == "paused"  # not reverted by the same-cycle stale echo
    assert row["name"] == "Daily digest (renamed)"  # every other field still server-wins


@pytest.mark.asyncio
async def test_sync_now_definition_lifecycle_not_found_clears_without_sync_error(
    tmp_path, captured_logs
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Gone", server_id="srv-def-gone"
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "archive", "server_definition_id": "srv-def-gone"},
    )
    server_client = _empty_reminders_client()
    server_client.archive_automation_definition.side_effect = ServerClientNotFoundError(
        "gone"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert (
        db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    ), "nothing left server-side to transition -- clear rather than retry forever"
    state = db.get_sync_state("server:1") or {}
    assert not (state.get("sync_errors") or []), "a 404 here is not a user-facing error"
    assert any(
        level == "INFO" and "srv-def-gone" in message for level, message in captured_logs
    )


@pytest.mark.asyncio
async def test_sync_now_definition_lifecycle_validation_error_does_not_block_the_rest(
    tmp_path,
):
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientValidationError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    poisoned_id = db.create_automation_definition(
        "server:1", "recurring_question", "Archived already", server_id="srv-def-old"
    )
    db.record_pending_mutation(
        poisoned_id,
        "automation_lifecycle",
        "server:1",
        {"action": "pause", "server_definition_id": "srv-def-old"},
    )
    healthy_id = db.create_automation_definition(
        "server:1", "recurring_question", "Healthy", server_id="srv-def-healthy"
    )
    db.record_pending_mutation(
        healthy_id,
        "automation_lifecycle",
        "server:1",
        {"action": "resume", "server_definition_id": "srv-def-healthy"},
    )
    server_client = _empty_reminders_client()
    server_client.pause_automation_definition.side_effect = ServerClientValidationError(
        "scheduled_task_lifecycle_transition_invalid"
    )
    server_client.resume_automation_definition.return_value = {
        "id": "srv-def-healthy",
        "family": "recurring_question",
        "name": "Healthy",
        "lifecycle": "configured",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    # The healthy resume went through despite the poisoned pause ahead of it.
    assert db.get_automation_definition(healthy_id)["lifecycle"] == "configured"
    # And the poisoned one is settled, not left to jam the queue forever.
    assert db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    errors = (db.get_sync_state("server:1") or {}).get("sync_errors") or []
    assert any("lifecycle_transition_invalid" in error["message"] for error in errors)


@pytest.mark.asyncio
async def test_sync_now_definition_lifecycle_retryable_error_retains_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Daily digest", server_id="srv-def-1"
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "pause", "server_definition_id": "srv-def-1"},
    )
    server_client = _empty_reminders_client()
    server_client.pause_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # UAT finding 3c: the reminder phase's own "ok" is no longer masked
    # by this unrelated definition-push (lifecycle) failure.
    assert outcome.status == "ok"
    assert outcome.error is None
    assert len(outcome.phase_errors) == 1
    assert "Automation definition push" in outcome.phase_errors[0]
    assert "offline" in outcome.phase_errors[0]
    pending = db.get_pending_mutations("server:1", primitive="automation_lifecycle")
    assert len(pending) == 1, "the mutation must be left queued for retry"
    assert db.get_automation_definition(local_id)["lifecycle"] == "configured"


@pytest.mark.asyncio
async def test_sync_now_definition_lifecycle_without_server_id_drops_mutation(tmp_path):
    """A lifecycle action can't apply to a definition the server has never
    seen -- unlike `update`, there is no create to convert this into, so an
    (unreachable via the current queuing path, but defensively guarded)
    missing `server_definition_id` just drops the mutation rather than
    retrying forever."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "server:1", "recurring_question", "Offline draft"
    )
    db.record_pending_mutation(
        local_id,
        "automation_lifecycle",
        "server:1",
        {"action": "pause", "server_definition_id": None},
    )
    server_client = _empty_reminders_client()
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.pause_automation_definition.assert_not_awaited()
    assert (
        db.get_pending_mutations("server:1", primitive="automation_lifecycle") == []
    )


# ----------------------------------------------------------------------
# Local -> server transfer replay (schedules-handoff PR-5, task 4) --
# spec §6.1. Both primitives share the same disarm-before-send / convert-
# or-merge / definitive-failure shape; see `_push_definition_transfer`/
# `_push_reminder_transfer`'s docstrings for the full reasoning.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_disarms_before_send_then_converts(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "local", "recurring_question", "Daily digest", lifecycle="paused"
    )
    db.set_transfer_state(
        "automation_definition", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Daily digest",
                # Client vocabulary -- the push must translate this onto
                # the wire exactly once (schedule_vocabulary.py, task 3).
                "schedule": {"kind": "interval", "every_seconds": 3600},
            },
        },
    )

    observed_state_at_send: list[object] = []
    server_client = _empty_reminders_client()

    async def _preview(request):
        observed_state_at_send.append(
            db.get_automation_definition(local_id)["transfer_state"]
        )
        return {"id": "prev-1", "status": "valid", "validation_errors": []}

    server_client.preview_automation_definition.side_effect = _preview
    server_client.create_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "paused",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert observed_state_at_send == ["to_server_sent"], (
        "the row must be disarmed (to_server_sent) BEFORE the network "
        "request fires"
    )
    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["mode"] == "create"
    assert request["schedule"] == {"kind": "interval", "seconds": 3600}
    server_client.create_automation_definition.assert_awaited_once_with(
        "prev-1", initial_lifecycle="paused"
    ), "initial_lifecycle must match the source row's own lifecycle"
    row = db.get_automation_definition(local_id)
    assert row["server_id"] == "srv-def-1"
    assert row["owner_id"] == "server:1"
    assert row["transfer_state"] is None
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_cas_skip_when_not_pending(tmp_path):
    """A concurrent cancel (or a prior attempt still awaiting Task 6's
    startup recovery) leaves the row NOT in to_server_pending -- the CAS
    fails and this replay must not touch the server or the mutation."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "local", "recurring_question", "Daily digest"
    )
    # No transfer_state set at all (simulates a cancelled/never-armed row)
    # despite a queued transfer mutation.
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {"family": "recurring_question", "name": "X"},
        },
    )
    server_client = _empty_reminders_client()
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.preview_automation_definition.assert_not_awaited()
    server_client.create_automation_definition.assert_not_awaited()
    assert (
        len(db.get_pending_mutations("server:1", primitive="automation_definition"))
        == 1
    ), "the mutation is left for whichever process owns the state change to clean up"


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_timeout_retains_sent_and_mutation(
    tmp_path,
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "local", "recurring_question", "Daily digest"
    )
    db.set_transfer_state(
        "automation_definition", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {"family": "recurring_question", "name": "X"},
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # UAT finding 3c: the reminder phase's own "ok" is no longer masked
    # by this unrelated definition-push (transfer) failure.
    assert outcome.status == "ok"
    assert len(outcome.phase_errors) == 1
    assert "Automation definition push" in outcome.phase_errors[0]
    row = db.get_automation_definition(local_id)
    assert row["transfer_state"] == "to_server_sent", (
        "disarm precedes the request -- an ambiguous failure leaves the "
        "row dormant, not re-armed"
    )
    assert row["server_id"] is None
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1, "the mutation must be left queued for recovery"


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_merges_with_existing_pulled_mirror(
    tmp_path,
):
    """§4 UNIQUE(owner_id, server_id) race: a pull lands the same server
    row between this transfer's send and its ack."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_automation_definition(
        "server:1",
        "recurring_question",
        "Daily digest",
        server_id="srv-def-1",
        created_at="2026-08-01T00:00:00+00:00",
    )
    local_id = db.create_automation_definition(
        "local",
        "recurring_question",
        "Daily digest (local)",
        created_at="2026-01-01T00:00:00+00:00",
    )
    db.set_transfer_state(
        "automation_definition", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Daily digest (local)",
            },
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.return_value = {"id": "srv-def-1"}
    server_client.list_automation_definitions.return_value = _definition_page([])
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_automation_definition(local_id) is None, (
        "the local transferring row must be deleted on merge"
    )
    mirror = db.get_automation_definition(mirror_id)
    assert mirror is not None
    assert mirror["created_at"] == "2026-01-01T00:00:00+00:00"
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_definitive_failure_no_auto_retry(
    tmp_path,
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "local", "recurring_question", "Daily digest"
    )
    db.set_transfer_state(
        "automation_definition", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {"family": "recurring_question", "name": "X"},
        },
    )
    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": None,
        "status": "invalid",
        "validation_errors": [
            {"field": "schedule.kind", "code": "required", "message": "required"}
        ],
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.create_automation_definition.assert_not_awaited()
    row = db.get_automation_definition(local_id)
    assert row["transfer_state"] == "to_server_failed", (
        "a definitive failure re-arms the row locally (Task 1: not a "
        "dormant state)"
    )
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1, "the mutation is RETAINED, not cleared"
    assert pending[0]["payload"]["transfer_errors"] == ["schedule.kind:required"]
    state = db.get_sync_state("server:1") or {}
    errors = state.get("sync_errors") or []
    assert any("schedule.kind:required" in error["message"] for error in errors)

    # A second sync cycle must NOT auto-retry the failed transfer.
    server_client.preview_automation_definition.reset_mock()
    outcome_2 = await engine.sync_now()
    assert outcome_2.status == "ok"
    server_client.preview_automation_definition.assert_not_awaited()
    assert (
        len(db.get_pending_mutations("server:1", primitive="automation_definition"))
        == 1
    )


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_errors_skip_check_is_not_just_the_cas(
    tmp_path,
):
    """`_replay_definition_mutations`'s own `transfer_errors` skip check
    (ruling 3: "encode the skip") must stop a replay independently of the
    CAS guard inside `_push_definition_transfer` -- not merely happen to
    be covered by it. Force the inconsistent state a plain CAS check alone
    would NOT catch (row back at `to_server_pending`, but the mutation
    still carries `transfer_errors` from a prior failure) and confirm the
    server is still never called."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_automation_definition(
        "local", "recurring_question", "Daily digest"
    )
    db.set_transfer_state(
        "automation_definition", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {"family": "recurring_question", "name": "X"},
            "transfer_errors": ["schedule.kind:required"],
        },
    )
    server_client = _empty_reminders_client()
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.preview_automation_definition.assert_not_awaited()
    server_client.create_automation_definition.assert_not_awaited()
    assert (
        len(db.get_pending_mutations("server:1", primitive="automation_definition"))
        == 1
    )


@pytest.mark.asyncio
async def test_sync_now_definition_transfer_poisoned_mutation_does_not_block_the_rest(
    tmp_path,
):
    """Phase discipline (PR-4 Qodo lesson): one poisoned transfer must not
    block a healthy mutation queued alongside it."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientValidationError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    poisoned_id = db.create_automation_definition(
        "local", "recurring_question", "Poisoned"
    )
    db.set_transfer_state(
        "automation_definition", poisoned_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        poisoned_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {"family": "recurring_question", "name": "Poisoned"},
        },
    )
    healthy_id = db.create_automation_definition(
        "local", "recurring_question", "Healthy"
    )
    db.set_transfer_state(
        "automation_definition", healthy_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        healthy_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {"family": "recurring_question", "name": "Healthy"},
        },
    )

    server_client = _empty_reminders_client()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.side_effect = [
        ServerClientValidationError("scheduled_task_definition_conflict"),
        {"id": "srv-def-healthy"},
    ]
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_automation_definition(healthy_id)["server_id"] == "srv-def-healthy"
    poisoned_row = db.get_automation_definition(poisoned_id)
    assert poisoned_row["transfer_state"] == "to_server_failed"
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["local_id"] == poisoned_id
    assert "transfer_errors" in pending[0]["payload"]


# --- Reminder-side transfer (same shape, no preview step) -------------


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_disarms_before_send_then_converts(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )

    observed_state_at_send: list[object] = []
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}

    async def _create_reminder(**kwargs):
        observed_state_at_send.append(
            db.get_reminder_task(local_id)["transfer_state"]
        )
        return {"id": "srv-rem-1", "title": kwargs.get("title")}

    server_client.create_reminder.side_effect = _create_reminder
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert observed_state_at_send == ["to_server_sent"], (
        "the row must be disarmed BEFORE the network request fires"
    )
    call_kwargs = server_client.create_reminder.await_args.kwargs
    assert call_kwargs["link_type"] == "chatbook_transfer"
    assert call_kwargs["link_id"] == local_id
    row = db.get_reminder_task(local_id)
    assert row["server_id"] == "srv-rem-1"
    assert row["owner_id"] == "server:1"
    assert row["transfer_state"] is None
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_cas_skip_when_not_pending(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.create_reminder.assert_not_awaited()
    assert (
        len(db.get_pending_mutations("server:1", primitive="reminder_task")) == 1
    )


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_timeout_retains_sent_and_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.side_effect = ServerUnavailableError("offline")
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "error"
    row = db.get_reminder_task(local_id)
    assert row["transfer_state"] == "to_server_sent"
    assert row["server_id"] is None
    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_merges_with_existing_pulled_mirror(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-1",
        created_at="2026-08-01T00:00:00+00:00",
    )
    local_id = db.create_reminder_task(
        owner_id="local",
        title="Standup (local)",
        schedule_kind="one_time",
        created_at="2026-01-01T00:00:00+00:00",
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup (local)", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-rem-1", "title": "Standup"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_reminder_task(local_id) is None
    mirror = db.get_reminder_task(mirror_id)
    assert mirror is not None
    assert mirror["created_at"] == "2026-01-01T00:00:00+00:00"
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_definitive_failure_no_auto_retry(tmp_path):
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientValidationError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.side_effect = ServerClientValidationError(
        "field_required"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    row = db.get_reminder_task(local_id)
    assert row["transfer_state"] == "to_server_failed"
    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["transfer_errors"] == ["field_required"]

    server_client.create_reminder.reset_mock(side_effect=True)
    outcome_2 = await engine.sync_now()
    assert outcome_2.status == "ok"
    server_client.create_reminder.assert_not_awaited()
    assert len(db.get_pending_mutations("server:1", primitive="reminder_task")) == 1


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_errors_skip_check_is_not_just_the_cas(
    tmp_path,
):
    """Reminder-side equivalent of the definitions-side isolation test:
    `_network_phase`'s own `transfer_errors` skip check must stop a replay
    independently of the CAS guard inside `_push_reminder_transfer`. Force
    the row back to `to_server_pending` while the mutation still carries
    `transfer_errors` from a prior failure and confirm the server is still
    never called."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
            "transfer_errors": ["field_required"],
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.create_reminder.assert_not_awaited()
    assert len(db.get_pending_mutations("server:1", primitive="reminder_task")) == 1


@pytest.mark.asyncio
async def test_sync_now_reminder_transfer_poisoned_mutation_does_not_block_the_rest(
    tmp_path,
):
    """Reminder-side equivalent of the definitions-side phase-discipline
    test (PR-4 Qodo lesson): one poisoned transfer must not block a
    healthy mutation queued alongside it."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientValidationError,
    )

    db = ScheduledTasksDB(tmp_path / "db.db")
    poisoned_id = db.create_reminder_task(
        owner_id="local", title="Poisoned", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", poisoned_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        poisoned_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Poisoned", "schedule_kind": "one_time"},
        },
    )
    healthy_id = db.create_reminder_task(
        owner_id="local", title="Healthy", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", healthy_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        healthy_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Healthy", "schedule_kind": "one_time"},
        },
    )

    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.side_effect = [
        ServerClientValidationError("field_required"),
        {"id": "srv-rem-healthy"},
    ]
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_reminder_task(healthy_id)["server_id"] == "srv-rem-healthy"
    poisoned_row = db.get_reminder_task(poisoned_id)
    assert poisoned_row["transfer_state"] == "to_server_failed"
    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["local_id"] == poisoned_id
    assert "transfer_errors" in pending[0]["payload"]


# ----------------------------------------------------------------------
# Server -> local release replay (schedules-handoff PR-5, task 5) --
# spec §6.2. `local_id` on the pending mutation is the SERVER-OWNED
# MIRROR row the release targets; `payload["local_copy_id"]` is the
# dormant local-owner copy `create_local_copy_from_mirror` already
# created, which is what actually arms once the release acks. See
# `_push_definition_release`/`_push_reminder_release`'s docstrings for
# the full reasoning.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_now_definition_release_archives_and_arms_local_copy(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_automation_definition(
        "server:1", "recurring_question", "Daily digest", server_id="srv-def-1"
    )
    copy_id = db.create_local_copy_from_mirror("automation_definition", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "automation_definition",
        "server:1",
        {
            "action": "release_from_server",
            "server_definition_id": "srv-def-1",
            "local_copy_id": copy_id,
        },
    )
    server_client = _empty_reminders_client()
    server_client.archive_automation_definition.return_value = {
        "id": "srv-def-1",
        "family": "recurring_question",
        "name": "Daily digest",
        "lifecycle": "archived",
        "archived_at": "2026-09-01T00:00:00+00:00",
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.archive_automation_definition.assert_awaited_once_with("srv-def-1")
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )
    mirror_row = db.get_automation_definition(mirror_id)
    assert mirror_row["lifecycle"] == "archived", (
        "the archive echo mirrors onto the server-mirror row"
    )
    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["transfer_state"] is None, "ack arms the local copy"


@pytest.mark.asyncio
async def test_sync_now_definition_release_not_found_treated_as_ack(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_automation_definition(
        "server:1", "recurring_question", "Gone", server_id="srv-def-gone"
    )
    copy_id = db.create_local_copy_from_mirror("automation_definition", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "automation_definition",
        "server:1",
        {
            "action": "release_from_server",
            "server_definition_id": "srv-def-gone",
            "local_copy_id": copy_id,
        },
    )
    server_client = _empty_reminders_client()
    server_client.archive_automation_definition.side_effect = ServerClientNotFoundError(
        "gone"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    ), "the server row is already gone -- settle rather than retry forever"
    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["transfer_state"] is None, "a 404 release is treated exactly as an ack"
    state = db.get_sync_state("server:1") or {}
    assert not (state.get("sync_errors") or []), "a 404 here is not a user-facing error"


@pytest.mark.asyncio
async def test_sync_now_definition_release_retryable_error_keeps_copy_dormant(
    tmp_path,
):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_automation_definition(
        "server:1", "recurring_question", "Daily digest", server_id="srv-def-1"
    )
    copy_id = db.create_local_copy_from_mirror("automation_definition", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "automation_definition",
        "server:1",
        {
            "action": "release_from_server",
            "server_definition_id": "srv-def-1",
            "local_copy_id": copy_id,
        },
    )
    server_client = _empty_reminders_client()
    server_client.archive_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    # UAT finding 3c: the reminder phase's own "ok" is no longer masked
    # by this unrelated definition-push (release) failure.
    assert outcome.status == "ok"
    assert len(outcome.phase_errors) == 1
    assert "Automation definition push" in outcome.phase_errors[0]
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1, "the mutation must be left queued for retry"
    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["transfer_state"] == "from_server_pending", (
        "dormant until an actual ack -- a failed release attempt must not arm it"
    )
    mirror_row = db.get_automation_definition(mirror_id)
    assert mirror_row["lifecycle"] == "configured", "the mirror is untouched too"


@pytest.mark.asyncio
async def test_sync_now_reminder_release_deletes_and_arms_local_copy(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-1",
    )
    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "reminder_task",
        "server:1",
        {
            "action": "release_from_server",
            "server_task_id": "srv-rem-1",
            "local_copy_id": copy_id,
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.delete_reminder.assert_awaited_once_with("srv-rem-1")
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []
    copy_row = db.get_reminder_task(copy_id)
    assert copy_row["transfer_state"] is None, "ack arms the local copy"
    # Final review I4: the mirror is torn down HERE, on the ack. Deferring
    # it to the pull's full-set reconciliation did not delete it -- that
    # scan only deletes rows carrying a local tombstone, so a released
    # mirror instead became a permanent bogus "the server deleted this"
    # conflict beside the armed local copy.
    assert db.get_reminder_task(mirror_id) is None
    assert (
        db.get_sync_mapping_by_local_id(mirror_id, "reminder_task", "server:1") is None
    )
    assert db.get_conflicts("server:1") == []


@pytest.mark.asyncio
async def test_sync_now_reminder_release_not_found_treated_as_ack(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-gone",
    )
    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "reminder_task",
        "server:1",
        {
            "action": "release_from_server",
            "server_task_id": "srv-rem-gone",
            "local_copy_id": copy_id,
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.delete_reminder.side_effect = ServerClientNotFoundError("gone")
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []
    copy_row = db.get_reminder_task(copy_id)
    assert copy_row["transfer_state"] is None, "a 404 release is treated exactly as an ack"
    state = db.get_sync_state("server:1") or {}
    assert not (state.get("sync_errors") or [])


@pytest.mark.asyncio
async def test_sync_now_reminder_release_retryable_error_keeps_copy_dormant(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-1",
    )
    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "reminder_task",
        "server:1",
        {
            "action": "release_from_server",
            "server_task_id": "srv-rem-1",
            "local_copy_id": copy_id,
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.delete_reminder.side_effect = ServerUnavailableError("offline")
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "error"
    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1, "the mutation must be left queued for retry"
    copy_row = db.get_reminder_task(copy_id)
    assert copy_row["transfer_state"] == "from_server_pending", (
        "dormant until an actual ack -- a failed release attempt must not arm it"
    )
    assert db.get_reminder_task(mirror_id) is not None


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

    # UAT finding 3c (supersedes task-23105-review F2's "must surface as
    # an error outcome" ruling): the reminder phase (and the later
    # results phase) ran to completion -- that success must be reported
    # honestly rather than collapsed into `status="error"` next to a
    # fresh error badge, which is exactly the "Sync failed" toast the
    # UAT caught misreporting an unrelated phase's failure. The
    # definitions-phase failure still reaches the caller, labeled.
    assert outcome.status == "ok"
    assert outcome.error is None
    assert len(outcome.phase_errors) == 1
    assert "Automation definitions pull" in outcome.phase_errors[0]
    assert "down" in outcome.phase_errors[0]
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
    phase -- genuinely fails. UAT finding 3c: `status`/`error` now
    describe ONLY the reminder phase (unconditionally, not just when it
    was "ok") -- the automation failure must still reach the caller, via
    `phase_errors`, never silently dropped."""
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

    assert outcome.status == "not_applicable"
    assert outcome.error is None
    assert len(outcome.phase_errors) == 1
    assert "Automation definitions pull" in outcome.phase_errors[0]
    assert "defs down" in outcome.phase_errors[0]
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


# ---------------------------------------------------------------------------
# Final whole-branch review fixes (2026-09-02)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reminder_transfer_does_not_fabricate_a_deletion_conflict(tmp_path):
    """C1: `_network_phase` pulls BEFORE it pushes, so the transfer's
    brand-new server_id is absent from the pull snapshot -- and
    `convert_row_to_server_mirror` flips owner_id into the scan's own
    (owner_id, server_id) window. A SUCCESSFUL move used to light up the
    Conflicts tab claiming the server had deleted the row."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Ping", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {"title": "Ping"}},
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-rem-1", "title": "Ping"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    row = db.get_reminder_task(local_id)
    assert row["owner_id"] == "server:1" and row["server_id"] == "srv-rem-1"
    assert db.get_conflicts("server:1") == [], (
        "a successful move must not report itself as a server-side deletion"
    )


@pytest.mark.asyncio
async def test_reminder_transfer_keeps_an_existing_link(tmp_path):
    """L12: a reminder that already carries a link (a watchlist run, say)
    keeps it -- the transfer marker is only stamped when there is none."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Ping", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {
                "title": "Ping",
                "link_type": "watchlist_run",
                "link_id": "wl-9",
            },
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-rem-1"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    await engine.sync_now()

    sent = server_client.create_reminder.await_args.kwargs
    assert sent["link_type"] == "watchlist_run"
    assert sent["link_id"] == "wl-9"


@pytest.mark.asyncio
async def test_reminder_release_raises_no_conflict_while_the_mirror_is_listed(tmp_path):
    """I4: the release mutation is keyed on the MIRROR, so the pull that
    still lists that (not yet released) server row used to hit the
    unconditional "local mutation pending => conflict" rule -- every
    server -> local move raised a bogus conflict on its own cycle."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-9",
    )
    db.set_sync_mapping(mirror_id, "srv-rem-9", "reminder_task", "server:1")
    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "reminder_task",
        "server:1",
        {
            "action": "release_from_server",
            "server_task_id": "srv-rem-9",
            "local_copy_id": copy_id,
        },
    )
    server_client = AsyncMock()
    # The pull still lists the server row: the release has not run yet.
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-rem-9", "title": "Standup"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    assert db.get_conflicts("server:1") == []
    assert db.get_reminder_task(copy_id)["transfer_state"] is None, "copy armed"
    assert db.get_reminder_task(mirror_id) is None, "mirror torn down on the ack"


@pytest.mark.asyncio
async def test_reminder_release_same_cycle_stale_pull_does_not_ghost_reinsert(
    tmp_path,
):
    """root-causes.md #4 (Major 6, task-3 ghost row): `_network_phase`
    pulls BEFORE it pushes, so the pull's `pulled_items` still lists the
    mirror server-side row `_push_reminder_release` is about to
    hard-delete in this SAME cycle. Applying that stale payload used to
    re-insert the mirror under a brand-new local id no tombstone could
    ever remove -- a permanent "deleted on server" conflict that survives
    navigation, re-sync, and conflict resolution.

    Fix: `_push_reminder_release` now returns `released_server_id`, and
    `_sync_reminders` filters `pulled_items` by it before applying --
    the exact twin of the `adopted_server_id` seen-set guard the opposite
    direction (transfer_to_server) already had.
    """
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-9",
    )
    db.set_sync_mapping(mirror_id, "srv-rem-9", "reminder_task", "server:1")
    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "reminder_task",
        "server:1",
        {
            "action": "release_from_server",
            "server_task_id": "srv-rem-9",
            "local_copy_id": copy_id,
        },
    )
    server_client = AsyncMock()
    # The pull still lists the server row: the release hasn't run yet
    # THIS cycle -- the exact same-cycle stale-payload window.
    server_client.list_reminders.return_value = {
        "items": [{"id": "srv-rem-9", "title": "Standup"}]
    }
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    all_rows = db.list_reminder_tasks(owner_id=None)
    assert len(all_rows) == 1, (
        "the stale pull must not re-insert the mirror the release just "
        f"tore down; rows: {all_rows!r}"
    )
    assert all_rows[0]["id"] == copy_id
    assert db.get_conflicts("server:1") == [], (
        "a re-inserted mirror with no tombstone would forever conflict "
        "as 'deleted on server'"
    )


@pytest.mark.asyncio
async def test_rejected_reminder_release_settles_per_mutation(tmp_path):
    """L15: a definitively rejected release used to re-raise through
    `_push_mutation`'s blanket `except ServerClientError: raise`, aborting
    the whole reminder push phase every cycle, forever."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-9",
    )
    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)
    db.record_pending_mutation(
        mirror_id,
        "reminder_task",
        "server:1",
        {
            "action": "release_from_server",
            "server_task_id": "srv-rem-9",
            "local_copy_id": copy_id,
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.delete_reminder.side_effect = ServerClientValidationError("refused")
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok", "one poisoned release must not abort the phase"
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []
    assert db.get_reminder_task(copy_id)["transfer_state"] == "from_server_pending", (
        "nothing was released, so nothing may arm -- cancel is its recovery"
    )
    state = db.get_sync_state("server:1") or {}
    assert state.get("sync_errors"), "the refusal must be reported, not swallowed"


# ----------------------------------------------------------------------
# Orphaned transfer_to_server mutation settlement (task-3, root-causes.md
# #5 / Major 9 / plan ruling 4). `_network_phase`'s mutation query is
# scoped to the CURRENT target_owner, so a `transfer_to_server` mutation
# recorded under a PRIOR server scope (the configured server's address
# changed underneath it) is never selected, never attempted, and used to
# hang `to_server_pending` forever -- no route to Retry/Cancel. `_settle_
# orphaned_transfer_mutations` is the sweep that finds and settles one.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphaned_reminder_transfer_mutation_settles_to_failed(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    # Queued while "server:old-host" was the active server; the config
    # has since moved to "server:1" (this test's SyncEngine owner).
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:old-host",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    # An orphaned mutation settles locally -- it must never be replayed
    # against whatever server happens to be active now.
    server_client.create_reminder.assert_not_awaited()
    row = db.get_reminder_task(local_id)
    assert row["transfer_state"] == "to_server_failed", (
        "settled, not left hanging to_server_pending forever"
    )
    pending = db.get_pending_mutations("server:old-host", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["transfer_errors"], (
        "Retry/Cancel + 'Last transfer error:' both key off a truthy "
        "transfer_errors entry"
    )


@pytest.mark.asyncio
async def test_orphaned_definition_transfer_mutation_settles_to_failed(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    definition_id = db.create_automation_definition(
        "local", "recurring_question", "Daily digest"
    )
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        definition_id,
        "automation_definition",
        "server:old-host",
        {
            "action": "transfer_to_server",
            "definition_payload": {
                "family": "recurring_question",
                "name": "Daily digest",
            },
        },
    )
    server_client = _empty_reminders_client()
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.preview_automation_definition.assert_not_awaited()
    row = db.get_automation_definition(definition_id)
    assert row["transfer_state"] == "to_server_failed"
    pending = db.get_pending_mutations(
        "server:old-host", primitive="automation_definition"
    )
    assert len(pending) == 1
    assert pending[0]["payload"]["transfer_errors"]


@pytest.mark.asyncio
async def test_transfer_mutation_still_scoped_to_active_server_is_not_orphaned(
    tmp_path,
):
    """The non-regression twin: a mutation whose scope MATCHES the
    current active server must replay normally, not get swept as
    orphaned."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "Standup", "schedule_kind": "one_time"},
        },
    )
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-rem-1"}
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok"
    server_client.create_reminder.assert_awaited_once()
    row = db.get_reminder_task(local_id)
    assert row["transfer_state"] is None, "converted to a server mirror, not failed"
    assert row["server_id"] == "srv-rem-1"


# ----------------------------------------------------------------------
# Capabilities handshake (task-3, root-causes.md #7, plan ruling 5).
# `get_capabilities` returning `None` means the server predates
# Scheduled Tasks automation entirely -- `_pull_definitions`/`_pull_
# results` must skip outright rather than page into a guaranteed 404.
# `ServerClientNotFoundError` from the ACTUAL results call despite a
# successful capabilities probe is the narrower, real UAT repro (a
# mid-rollout server): that must degrade to the SAME honest copy family,
# never a raw scheduled_task_not_found.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_results_skips_outright_when_capabilities_absent(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.get_capabilities = AsyncMock(return_value=None)
    engine = SyncEngine(db, server_client, owner_id="server:1")

    result = await engine._pull_results("server:1")

    assert result == {}
    server_client.list_automation_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_pull_definitions_skips_outright_when_capabilities_absent(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.get_capabilities = AsyncMock(return_value=None)
    engine = SyncEngine(db, server_client, owner_id="server:1")

    result = await engine._pull_definitions("server:1")

    assert result == {}
    server_client.list_automation_definitions.assert_not_awaited()


@pytest.mark.asyncio
async def test_pull_results_proceeds_when_capabilities_present(tmp_path):
    """The non-regression twin: capabilities present -> the pull still
    runs normally (the gate must not become a blanket skip)."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.get_capabilities = AsyncMock(return_value={"items": []})
    engine = SyncEngine(db, server_client, owner_id="server:1")

    result = await engine._pull_results("server:1")

    assert result == {"inserted": 0, "updated": 0, "skipped_dedupe": 0}
    server_client.list_automation_results.assert_awaited_once()


@pytest.mark.asyncio
async def test_pull_results_reports_honest_copy_when_route_missing_despite_capabilities(
    tmp_path,
):
    """The actual UAT repro (root-causes.md #7): capabilities ARE present
    (a mid-rollout server), but `/results` specifically 404s -- a probe
    alone cannot predict this, so the per-call catch is what turns it
    into the same honest family of copy instead of a raw
    scheduled_task_not_found poisoning the sync verdict (Minor 24 /
    Major 7)."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = _empty_reminders_client()
    server_client.get_capabilities = AsyncMock(return_value={"items": []})
    server_client.list_automation_results = AsyncMock(
        side_effect=ServerClientNotFoundError("scheduled_task_not_found")
    )
    engine = SyncEngine(db, server_client, owner_id="server:1")

    outcome = await engine.sync_now()

    assert outcome.status == "ok", "the reminder phase is unaffected"
    assert len(outcome.phase_errors) == 1
    assert "does not provide the results inbox" in outcome.phase_errors[0]
    assert "scheduled_task_not_found" not in outcome.phase_errors[0], (
        "the raw server error code must not leak into the user-facing copy"
    )
