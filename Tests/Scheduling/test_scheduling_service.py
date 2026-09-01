"""Tests for SchedulingService local/server routing and offline behavior."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services import SchedulingServerClient, SchedulingService
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService


@pytest.fixture
def db(tmp_path):
    """Return a fresh in-file ScheduledTasksDB for each test."""
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


def _reminder_payload(title, **kwargs):
    """Build a valid reminder payload for tests."""
    payload = {
        "title": title,
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }
    payload.update(kwargs)
    return payload


@pytest.mark.asyncio
async def test_create_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Test"))

    assert isinstance(task, ReminderTask)
    assert task.title == "Test"
    assert task.owner_id == "local"
    assert task.schedule_kind.value == "one_time"


@pytest.mark.asyncio
async def test_create_reminder_server_happy_path(db):
    server_client = AsyncMock()
    server_client.create_reminder.return_value = {
        "id": "srv-1",
        "title": "Server Task",
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    task = await svc.create_reminder(_reminder_payload("Server Task"))

    assert task.title == "Server Task"
    assert task.server_id == "srv-1"
    assert task.owner_id == "server:1"
    server_client.create_reminder.assert_awaited_once()

    mapping = db.get_sync_mapping_by_server_id("srv-1", "reminder_task", "server:1")
    assert mapping is not None
    assert mapping["local_id"] == task.id


@pytest.mark.asyncio
async def test_create_reminder_server_falls_back_local_on_unavailable(db):
    server_client = AsyncMock()
    server_client.create_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    task = await svc.create_reminder(_reminder_payload("Fallback"))

    assert task.title == "Fallback"
    assert task.owner_id == "server:1"
    server_client.create_reminder.assert_awaited_once()

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"
    assert pending[0]["payload"]["fields"]["title"] == "Fallback"


@pytest.mark.asyncio
async def test_create_reminder_server_falls_back_on_generic_error(db):
    server_client = AsyncMock()
    server_client.create_reminder.side_effect = RuntimeError("boom")

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    task = await svc.create_reminder(_reminder_payload("Fallback"))

    assert task.title == "Fallback"
    assert task.owner_id == "server:1"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"


@pytest.mark.asyncio
async def test_list_reminders_filtered_by_owner(db):
    db.create_reminder_task(
        owner_id="local",
        title="Local Task",
        schedule_kind="one_time",
        run_at="2026-07-20T14:00:00+00:00",
    )
    db.create_reminder_task(
        owner_id="server:1",
        title="Server Task",
        schedule_kind="one_time",
        run_at="2026-07-20T15:00:00+00:00",
    )

    svc = SchedulingService(db=db, runtime_source="server:1")
    tasks = await svc.list_reminders()

    assert len(tasks) == 1
    assert tasks[0].title == "Server Task"
    assert tasks[0].owner_id == "server:1"


@pytest.mark.asyncio
async def test_get_reminder_returns_none_for_missing_id(db):
    svc = SchedulingService(db=db, runtime_source="local")
    result = await svc.get_reminder("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_get_reminder_returns_task(db):
    svc = SchedulingService(db=db, runtime_source="local")
    created = await svc.create_reminder(_reminder_payload("Fetch me"))

    fetched = await svc.get_reminder(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "Fetch me"


@pytest.mark.asyncio
async def test_update_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    refreshed = await svc.get_reminder(task.id)
    assert refreshed is not None
    assert refreshed.title == "Updated"


@pytest.mark.asyncio
async def test_update_reminder_server_with_server_id_happy_path(db):
    server_client = AsyncMock()
    server_client.update_reminder.return_value = {
        "id": "srv-1",
        "title": "Updated",
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        task.id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Stale"}},
    )

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"
    assert updated.server_id == "srv-1"
    server_client.update_reminder.assert_awaited_once_with("srv-1", title="Updated")

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_update_reminder_server_without_server_id_creates_on_server(db):
    server_client = AsyncMock()
    server_client.create_reminder.return_value = {
        "id": "srv-new",
        "title": "Updated",
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"
    assert updated.server_id == "srv-new"
    server_client.create_reminder.assert_awaited_once()
    call_kwargs = server_client.create_reminder.call_args.kwargs
    assert call_kwargs["title"] == "Updated"
    assert call_kwargs["schedule_kind"] == "one_time"

    mapping = db.get_sync_mapping_by_server_id("srv-new", "reminder_task", "server:1")
    assert mapping is not None
    assert mapping["local_id"] == task.id


@pytest.mark.asyncio
async def test_update_reminder_server_falls_back_local_on_unavailable(db):
    server_client = AsyncMock()
    server_client.update_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "update"
    assert pending[0]["payload"]["fields"]["title"] == "Updated"


@pytest.mark.asyncio
async def test_update_reminder_server_falls_back_on_generic_error(db):
    server_client = AsyncMock()
    server_client.update_reminder.side_effect = RuntimeError("boom")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_delete_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))

    result = await svc.delete_reminder(task.id)

    assert result is True
    assert await svc.get_reminder(task.id) is None


@pytest.mark.asyncio
async def test_delete_reminder_server_with_server_id_happy_path(db):
    server_client = AsyncMock()
    server_client.delete_reminder.return_value = {"deleted": True}

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        task.id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Stale"}},
    )

    result = await svc.delete_reminder(task.id)

    assert result is True
    server_client.delete_reminder.assert_awaited_once_with("srv-1")
    assert await svc.get_reminder(task.id) is None
    assert db.get_sync_mapping_by_local_id(task.id, "reminder_task", "server:1") is None

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_delete_reminder_server_without_server_id_clears_pending_and_deletes_local(
    db,
):
    server_client = AsyncMock()
    server_client.delete_reminder.return_value = {"deleted": True}

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.record_pending_mutation(
        task.id,
        "reminder_task",
        "server:1",
        {"action": "create", "fields": {"title": "To delete"}},
    )

    result = await svc.delete_reminder(task.id)

    assert result is True
    server_client.delete_reminder.assert_not_awaited()
    assert await svc.get_reminder(task.id) is None

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_delete_reminder_server_falls_back_to_tombstone_on_unavailable(db):
    server_client = AsyncMock()
    server_client.delete_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")

    result = await svc.delete_reminder(task.id)

    assert result is True
    server_client.delete_reminder.assert_awaited_once_with("srv-1")
    assert await svc.get_reminder(task.id) is None

    tombstone = db.get_tombstone(task.id, "reminder_task", "server:1")
    assert tombstone is not None


@pytest.mark.asyncio
async def test_delete_reminder_server_falls_back_to_tombstone_on_generic_error(db):
    server_client = AsyncMock()
    server_client.delete_reminder.side_effect = RuntimeError("boom")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")

    result = await svc.delete_reminder(task.id)

    assert result is True
    assert await svc.get_reminder(task.id) is None

    tombstone = db.get_tombstone(task.id, "reminder_task", "server:1")
    assert tombstone is not None


@pytest.mark.asyncio
async def test_delete_reminder_returns_false_for_missing_id(db):
    svc = SchedulingService(db=db, runtime_source="local")
    result = await svc.delete_reminder("does-not-exist")
    assert result is False


@pytest.mark.asyncio
async def test_sync_now_delegates_to_sync_engine(db):
    svc = SchedulingService(db=db, runtime_source="local")
    svc.sync_engine.sync_now = AsyncMock()

    await svc.sync_now()

    svc.sync_engine.sync_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_owner_propagates_to_sync_engine(db):
    svc = SchedulingService(db=db, runtime_source="local")
    svc.set_owner("server:42")

    assert svc.owner_id == "server:42"
    assert svc.sync_engine.owner_id == "server:42"


@pytest.mark.asyncio
async def test_list_tasks_includes_watchlist_projection(db):
    """list_tasks merges reminders with watchlist projections and sorts by next_run_at."""
    svc = SchedulingService(db=db, runtime_source="local")
    await svc.create_reminder(_reminder_payload("Reminder"))

    projection = MagicMock(spec=WatchlistProjection)
    projection.list_jobs.return_value = [
        ScheduledTask(
            id="watchlist:1",
            title="Watchlist Job",
            type="watchlist_job",
            status=TaskStatus.WAITING,
            next_run_at=datetime(2026, 7, 20, 13, 0, tzinfo=timezone.utc),
            owner_id="local",
        )
    ]
    svc.watchlist_projection = projection

    tasks = await svc.list_tasks()

    assert len(tasks) == 2
    assert tasks[0].title == "Watchlist Job"
    assert tasks[1].title == "Reminder"
    projection.list_jobs.assert_called_once_with(owner_id="local")


@pytest.mark.asyncio
async def test_list_tasks_without_projection_returns_only_reminders(db):
    """list_tasks returns only local reminders when no projection is configured."""
    svc = SchedulingService(db=db, runtime_source="local")
    reminder = await svc.create_reminder(_reminder_payload("Reminder"))

    tasks = await svc.list_tasks()

    assert len(tasks) == 1
    assert isinstance(tasks[0], ReminderTask)
    assert tasks[0].id == reminder.id


@pytest.mark.asyncio
async def test_list_tasks_filters_watchlist_by_owner(db):
    """list_tasks passes the current owner_id to the watchlist projection."""
    svc = SchedulingService(db=db, runtime_source="server:1")
    projection = MagicMock(spec=WatchlistProjection)
    projection.list_jobs.return_value = []
    svc.watchlist_projection = projection

    tasks = await svc.list_tasks()

    assert tasks == []
    projection.list_jobs.assert_called_once_with(owner_id="server:1")


# --- briefing projection (task-1810): scheduled briefings on the unified list ---
#
# Mirrors the watchlist-projection tests immediately above -- same shape,
# same sort/merge behavior, one extra source. `_cadenced_watchlist` and
# `_force_briefing_created_at` mirror the equivalent helpers in
# `test_briefing_projection.py`.


def _cadenced_watchlist(subs_db, name="Watch", cadence_seconds=3600):
    """Create a watchlist with a non-NULL briefing cadence (opted in)."""
    watchlist_id = WatchlistBundleService(subs_db).create(name=name)["id"]
    subs_db.set_watchlist_briefing_settings(
        watchlist_id, briefing_cadence_seconds=cadence_seconds
    )
    return watchlist_id


def _force_briefing_created_at(subs_db, briefing_id, timestamp):
    """Overwrite a `briefings` row's `created_at` directly (second resolution)."""
    subs_db.conn.execute(
        "UPDATE briefings SET created_at = ? WHERE id = ?", (timestamp, briefing_id)
    )
    subs_db.conn.commit()


@pytest.mark.asyncio
async def test_list_tasks_includes_briefing_projection(db):
    """list_tasks merges reminders with briefing projections and sorts by
    next_run_at -- the mock-based structural mirror of
    test_list_tasks_includes_watchlist_projection above."""
    svc = SchedulingService(db=db, runtime_source="local")
    await svc.create_reminder(_reminder_payload("Reminder"))

    projection = MagicMock(spec=BriefingProjection)
    projection.list_jobs.return_value = [
        ScheduledTask(
            id="briefing:1",
            title="Briefing Job",
            type="briefing_job",
            status=TaskStatus.WAITING,
            next_run_at=datetime(2026, 7, 20, 13, 0, tzinfo=timezone.utc),
            owner_id="local",
        )
    ]
    svc.briefing_projection = projection

    tasks = await svc.list_tasks()

    assert len(tasks) == 2
    assert tasks[0].title == "Briefing Job"
    assert tasks[1].title == "Reminder"
    projection.list_jobs.assert_called_once_with(owner_id="local")


@pytest.mark.asyncio
async def test_list_tasks_filters_briefing_by_owner(db):
    """list_tasks passes the current owner_id to the briefing projection too."""
    svc = SchedulingService(db=db, runtime_source="server:1")
    projection = MagicMock(spec=BriefingProjection)
    projection.list_jobs.return_value = []
    svc.briefing_projection = projection

    tasks = await svc.list_tasks()

    assert tasks == []
    projection.list_jobs.assert_called_once_with(owner_id="server:1")


@pytest.mark.asyncio
async def test_list_tasks_includes_both_watchlist_and_briefing_projections(db):
    """The two projections are additive, not mutually exclusive -- both
    branches extend the same unified list."""
    svc = SchedulingService(db=db, runtime_source="local")

    watchlist_projection = MagicMock(spec=WatchlistProjection)
    watchlist_projection.list_jobs.return_value = [
        ScheduledTask(
            id="watchlist:1",
            title="Watchlist Job",
            type="watchlist_job",
            status=TaskStatus.WAITING,
            next_run_at=None,
            owner_id="local",
        )
    ]
    briefing_projection = MagicMock(spec=BriefingProjection)
    briefing_projection.list_jobs.return_value = [
        ScheduledTask(
            id="briefing:1",
            title="Briefing Job",
            type="briefing_job",
            status=TaskStatus.WAITING,
            next_run_at=None,
            owner_id="local",
        )
    ]
    svc.watchlist_projection = watchlist_projection
    svc.briefing_projection = briefing_projection

    tasks = await svc.list_tasks()

    types = {task.type for task in tasks if isinstance(task, ScheduledTask)}
    assert types == {"watchlist_job", "briefing_job"}


@pytest.mark.asyncio
async def test_list_tasks_includes_a_cadenced_briefing_schedule_ac1(db, tmp_path):
    """AC #1: a watchlist with a non-NULL briefing_cadence_seconds shows up
    as a scheduled task on the unified list, alongside reminders and
    watchlist checks."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = _cadenced_watchlist(subs_db, name="Acme Watch", cadence_seconds=3600)
    complete_id = subs_db.insert_briefing(watchlist_id, status="complete")
    _force_briefing_created_at(subs_db, complete_id, "2026-01-01 00:00:00")

    projection = BriefingProjection(subs_db)
    svc = SchedulingService(db=db, runtime_source="local", briefing_projection=projection)
    await svc.create_reminder(_reminder_payload("Reminder"))

    tasks = await svc.list_tasks()

    briefing_tasks = [t for t in tasks if getattr(t, "type", None) == "briefing_job"]
    assert len(briefing_tasks) == 1
    assert briefing_tasks[0].id == f"briefing:{watchlist_id}"
    assert briefing_tasks[0].title == "Acme Watch"
    reminder_tasks = [t for t in tasks if isinstance(t, ReminderTask)]
    assert len(reminder_tasks) == 1


@pytest.mark.asyncio
async def test_briefing_schedule_next_run_at_matches_the_projection_ac2(db, tmp_path):
    """AC #2: the projected briefing task's next-run time on the unified
    list matches BriefingProjection.list_jobs' own calculation for the same
    data -- asserted against the projection directly, never re-derived
    here (a re-derivation could independently agree with a broken
    passthrough by coincidence)."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = _cadenced_watchlist(subs_db, name="Acme Watch", cadence_seconds=7200)
    complete_id = subs_db.insert_briefing(watchlist_id, status="complete")
    _force_briefing_created_at(subs_db, complete_id, "2026-01-01 00:00:00")

    projection = BriefingProjection(subs_db)
    svc = SchedulingService(db=db, runtime_source="local", briefing_projection=projection)

    tasks = await svc.list_tasks()
    [briefing_task] = [t for t in tasks if getattr(t, "type", None) == "briefing_job"]

    # The projection's OWN calculation for the same data, called
    # independently -- not a re-derivation of the expected value by hand.
    # Deterministic (not `now`-dependent) because the watchlist has a
    # `complete` briefing on record, so `next_run_at` is
    # `last_completed_at + cadence`, unaffected by wall-clock timing.
    [expected_task] = projection.list_jobs(owner_id="local")
    assert briefing_task.next_run_at == expected_task.next_run_at
    assert briefing_task.next_run_at == datetime(
        2026, 1, 1, 2, 0, 0, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_null_cadence_watchlist_absent_from_scheduling_screen_ac4(db, tmp_path):
    """AC #4: a watchlist with a NULL cadence (scheduling off) does not
    appear on the unified list at all."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db", "test")
    WatchlistBundleService(subs_db).create(name="Never Scheduled")  # no cadence set

    projection = BriefingProjection(subs_db)
    svc = SchedulingService(db=db, runtime_source="local", briefing_projection=projection)

    tasks = await svc.list_tasks()

    assert all(getattr(t, "type", None) != "briefing_job" for t in tasks)


@pytest.mark.asyncio
async def test_created_reminder_is_picked_up_by_priority_queue(db):
    """Locally created reminders must have next_run_at set so the queue loads them."""
    svc = SchedulingService(db=db, runtime_source="local")
    await svc.create_reminder(_reminder_payload("Queue me"))

    queue = PriorityQueue(db=db)
    queue.load(now=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert len(queue._items) == 1
    assert queue._items[0]["title"] == "Queue me"


@pytest.mark.asyncio
async def test_updated_reminder_is_picked_up_by_priority_queue(db):
    """Updating schedule fields recomputes next_run_at so the queue loads the reminder."""
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))

    updated = await svc.update_reminder(
        task.id,
        {
            "schedule_kind": "recurring",
            "run_at": None,
            "cron": "0 9 * * *",
            "timezone": "UTC",
        },
    )

    assert updated is not None
    assert updated.next_run_at is not None

    queue = PriorityQueue(db=db)
    queue.load(now=datetime(2099, 1, 1, tzinfo=timezone.utc))

    assert any(item["id"] == task.id for item in queue._items)


@pytest.mark.asyncio
async def test_sync_now_passes_owner_id_to_engine(db):
    engine = MagicMock()
    engine.sync_now = AsyncMock()
    svc = SchedulingService(db=db, runtime_source="local")
    svc.sync_engine = engine

    await svc.sync_now("server:example.com")

    engine.sync_now.assert_awaited_once_with("server:example.com")


@pytest.mark.asyncio
async def test_sync_now_defaults_to_current_owner(db):
    engine = MagicMock()
    engine.sync_now = AsyncMock()
    svc = SchedulingService(db=db, runtime_source="local")
    svc.sync_engine = engine

    await svc.sync_now()

    engine.sync_now.assert_awaited_once_with("local")


def test_server_client_is_always_present(db):
    svc = SchedulingService(db=db, runtime_source="local", server_client=None)
    assert isinstance(svc.server_client, SchedulingServerClient)
    assert svc.server_client.notifications_service is None


def test_app_wiring_briefing_projection_is_live_not_a_frozen_none():
    """Seam-level liveness proof for the app.py construction-order fix
    (task-1810): `_wire_watchlists_and_notifications_services` must pass a
    REAL `BriefingProjection` into `SchedulingService` at construction time,
    not `None` frozen in because the projection used to be built AFTER the
    service. That is the exact bug class task-1810's own dispatch brief
    flags as already having shipped once elsewhere in this same method (the
    kept-briefings branch's construction-order bug) -- this test reds
    against the naive fix (pass `briefing_projection=briefing_projection`
    at the OLD call site, before `briefing_projection` is assigned) just as
    it would red against never wiring the parameter at all.

    `_build_test_app`'s fake `get_cli_setting` passes through whatever
    default the caller supplies for any key other than
    `general.default_tab` (see its own docstring) -- `briefing_schedules_enabled`
    defaults to `True` in `app.py`, so this exercises the real, enabled-by-
    default production path, not a special-cased test config.
    """
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()

    assert app.scheduling_service.briefing_projection is not None
    assert isinstance(app.scheduling_service.briefing_projection, BriefingProjection)
    # And it is the SAME instance `SchedulerLoop`'s queue was wired with --
    # not two independently constructed projections that happen to both be
    # real, which would hide a subtler drift between the two consumers.
    assert (
        app.scheduling_service.briefing_projection
        is app.scheduler_loop.queue.briefing_projection
    )


# ---------------------------------------------------------------------------
# review_automation_result (schedules-handoff PR-3, task 5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_automation_result_local_only_updates_without_mutation(db):
    result_id = db.create_automation_result(
        "local", "def-1", "run-1", "finding", "T", "S", "key-1"
    )
    svc = SchedulingService(db=db, runtime_source="local")

    ok = await svc.review_automation_result(result_id, "dismissed", "not relevant")

    assert ok is True
    row = db.get_automation_result(result_id)
    assert row["review_state"] == "dismissed"
    assert row["review_note"] == "not relevant"
    assert db.get_pending_mutations("local", primitive="automation_result_review") == []


@pytest.mark.asyncio
async def test_review_automation_result_server_mirrored_records_pending_mutation(db):
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1",
        server_id="srv-res-1",
    )
    svc = SchedulingService(db=db, runtime_source="server:1")

    ok = await svc.review_automation_result(result_id, "dismissed", "noise")

    assert ok is True
    row = db.get_automation_result(result_id)
    assert row["review_state"] == "dismissed"

    pending = db.get_pending_mutations("server:1", primitive="automation_result_review")
    assert len(pending) == 1
    assert pending[0]["local_id"] == result_id
    assert pending[0]["payload"] == {
        "server_result_id": "srv-res-1",
        "review_state": "dismissed",
        "review_note": "noise",
        "idempotency_key": pending[0]["payload"]["idempotency_key"],
    }


@pytest.mark.asyncio
async def test_review_automation_result_rejects_unknown_review_state(db):
    result_id = db.create_automation_result(
        "local", "def-1", "run-1", "finding", "T", "S", "key-1"
    )
    svc = SchedulingService(db=db, runtime_source="local")

    ok = await svc.review_automation_result(result_id, "bogus")

    assert ok is False
    row = db.get_automation_result(result_id)
    assert row["review_state"] == "unread"  # untouched


@pytest.mark.asyncio
async def test_review_automation_result_unknown_id_returns_false(db):
    svc = SchedulingService(db=db, runtime_source="local")

    ok = await svc.review_automation_result("no-such-id", "dismissed")

    assert ok is False
