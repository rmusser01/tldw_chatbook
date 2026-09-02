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
async def test_review_automation_result_records_mutation_under_row_owner(db):
    """F1 (task-23105-review): the row's own owner_id is authoritative.

    The workbench can toggle the service's active owner independently of
    which owner a given result row belongs to -- recording the pending
    mutation under ``self.owner_id`` would strand it where the row's real
    owner never sees it via ``get_pending_mutations``.
    """
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1",
        server_id="srv-res-1",
    )
    svc = SchedulingService(db=db, runtime_source="local")

    ok = await svc.review_automation_result(result_id, "dismissed", "noise")

    assert ok is True
    pending = db.get_pending_mutations("server:1", primitive="automation_result_review")
    assert len(pending) == 1
    assert pending[0]["local_id"] == result_id
    assert db.get_pending_mutations("local", primitive="automation_result_review") == []


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


@pytest.mark.asyncio
async def test_review_automation_result_server_mirrored_makes_a_single_db_call(db):
    """review round 1 finding: the review write and its outbox mutation
    must be ONE ``update_result_review(..., pending_mutation=...)`` call --
    not a separate ``record_pending_mutation`` call after -- so the two
    can never land in different transactions.
    """
    result_id = db.create_automation_result(
        "server:1", "def-1", "run-1", "finding", "T", "S", "key-1",
        server_id="srv-res-1",
    )
    svc = SchedulingService(db=db, runtime_source="server:1")

    real_update = db.update_result_review
    calls: list[dict] = []

    def _spy_update(*args, **kwargs):
        calls.append(kwargs)
        return real_update(*args, **kwargs)

    db.update_result_review = _spy_update  # type: ignore[method-assign]
    db.record_pending_mutation = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError(
            "record_pending_mutation must not be called separately from "
            "review_automation_result -- it must go through "
            "update_result_review(pending_mutation=...)"
        )
    )

    ok = await svc.review_automation_result(result_id, "dismissed", "noise")

    assert ok is True
    assert len(calls) == 1
    mutation = calls[0]["pending_mutation"]
    assert mutation["local_id"] == result_id
    assert mutation["owner_id"] == "server:1"
    assert mutation["payload"]["server_result_id"] == "srv-res-1"

    pending = db.get_pending_mutations("server:1", primitive="automation_result_review")
    assert len(pending) == 1


# ----------------------------------------------------------------------
# preview_definition / save_definition (schedules-handoff PR-4, task 4)
# ----------------------------------------------------------------------


def _definition_payload(**overrides):
    """Build a valid recurring_question authoring payload for tests."""
    payload = {
        "family": "recurring_question",
        "name": "Daily standup question",
        "description": "Asks a daily question",
        "config": {},
        "input": {"question": "What did you work on today?"},
        "schedule": {"kind": "interval", "every_seconds": 3600},
        "visibility_policy": "findings_only",
        "notification_policy": {},
        "approval_policy": {},
    }
    payload.update(overrides)
    return payload


def _server_definition_echo(**overrides):
    """A `ScheduledTaskDefinitionResponse`-shaped create/update echo."""
    item = {
        "id": "srv-def-1",
        "owner_id": "server:1",
        "family": "recurring_question",
        "name": "Daily standup question",
        "lifecycle": "configured",
        "health": "execution_unavailable",
        "schedule": {"kind": "interval", "every_seconds": 3600},
        "input": {"question": "What did you work on today?"},
        "config": {},
        "visibility_policy": {"mode": "findings_only"},
        "notification_policy": {},
        "approval_policy": {},
        "version": 1,
        "created_at": "2026-09-01T09:00:00+00:00",
        "updated_at": "2026-09-01T09:00:00+00:00",
    }
    item.update(overrides)
    return item


@pytest.mark.asyncio
async def test_preview_definition_local_owner_valid(db):
    svc = SchedulingService(db=db, runtime_source="local")
    preview = await svc.preview_definition(_definition_payload(), "local")

    assert preview.status == "valid"
    assert preview.normalized_config["name"] == "Daily standup question"


@pytest.mark.asyncio
async def test_preview_definition_local_owner_invalid(db):
    svc = SchedulingService(db=db, runtime_source="local")
    preview = await svc.preview_definition(_definition_payload(name=""), "local")

    assert preview.status == "invalid"
    assert any(error["field"] == "name" for error in preview.validation_errors)


@pytest.mark.asyncio
async def test_preview_definition_rejects_unsupported_family(db):
    """v1 scope guard: `agent_task` is rejected before Task 1's pure preview
    runs, so its fabricated `family: unsupported` scope-cut error (a
    documented gap, not real server parity) never reaches a caller through
    this facade."""
    svc = SchedulingService(db=db, runtime_source="local")
    preview = await svc.preview_definition(
        _definition_payload(family="agent_task"), "local"
    )

    assert preview.status == "invalid"
    assert preview.validation_errors == [
        {
            "field": "family",
            "code": "unsupported",
            "message": (
                "Only recurring_question automations can be authored here "
                "(agent_task authoring is not yet available)."
            ),
        }
    ]


@pytest.mark.asyncio
async def test_preview_definition_server_owner_online(db):
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "mode": "create",
        "family": "recurring_question",
        "status": "valid",
        "normalized_config": {"name": "Daily standup question"},
        "validation_errors": [],
        "warnings": [],
        "visibility_policy": {"mode": "findings_only"},
        "schedule_preview": {"kind": "interval", "every_seconds": 3600},
    }
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    preview = await svc.preview_definition(_definition_payload(), "server:1")

    assert preview.status == "valid"
    assert preview.id == "prev-1"
    server_client.preview_automation_definition.assert_awaited_once()


@pytest.mark.asyncio
async def test_preview_definition_server_owner_falls_back_local_on_unreachable(db):
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    preview = await svc.preview_definition(_definition_payload(), "server:1")

    assert preview.status == "valid"  # local validation still runs offline
    assert any(
        warning["field"] == "_owner" and warning["code"] == "server_unreachable"
        for warning in preview.warnings
    )


@pytest.mark.asyncio
async def test_save_definition_local_create(db):
    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.save_definition(_definition_payload(), "local")

    assert outcome.status == "saved"
    assert outcome.errors == []
    assert outcome.definition_id
    row = db.get_automation_definition(outcome.definition_id)
    assert row["name"] == "Daily standup question"
    assert row["family"] == "recurring_question"
    assert row["owner_id"] == "local"
    assert row["schedule"] == {"kind": "interval", "every_seconds": 3600}
    # visibility_policy on the DB row comes from the preview's wrapped
    # top-level field, not normalized_config's flat mode string.
    assert row["visibility_policy"] == {"mode": "findings_only"}
    assert row["next_run_at"]  # an interval schedule always computes one


@pytest.mark.asyncio
async def test_save_definition_local_edit(db):
    svc = SchedulingService(db=db, runtime_source="local")
    created = await svc.save_definition(_definition_payload(), "local")

    outcome = await svc.save_definition(
        _definition_payload(name="Updated standup question"),
        "local",
        definition_id=created.definition_id,
    )

    assert outcome.status == "saved"
    assert outcome.definition_id == created.definition_id
    row = db.get_automation_definition(created.definition_id)
    assert row["name"] == "Updated standup question"
    assert row["version"] == 2


@pytest.mark.asyncio
async def test_save_definition_local_invalid_writes_nothing(db):
    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.save_definition(_definition_payload(name=""), "local")

    assert outcome.status == "invalid"
    assert any(error["field"] == "name" for error in outcome.errors)
    assert db.list_automation_definitions(owner_id="local") == []


@pytest.mark.asyncio
async def test_save_definition_family_guard_writes_nothing(db):
    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.save_definition(
        _definition_payload(family="agent_task"), "local"
    )

    assert outcome.status == "invalid"
    assert outcome.errors[0]["field"] == "family"
    assert db.list_automation_definitions(owner_id="local") == []


@pytest.mark.asyncio
async def test_save_definition_unknown_definition_id_returns_error(db):
    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.save_definition(
        _definition_payload(), "local", definition_id="no-such-id"
    )

    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "not_found"


@pytest.mark.asyncio
async def test_save_definition_server_owner_online_create_mirrors_new_row(db):
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.return_value = _server_definition_echo()

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "saved"
    assert outcome.definition_id
    row = db.get_automation_definition(outcome.definition_id)
    assert row["server_id"] == "srv-def-1"
    assert row["owner_id"] == "server:1"
    server_client.create_automation_definition.assert_awaited_once_with("prev-1")


@pytest.mark.asyncio
async def test_save_definition_server_owner_online_edit_adopts_existing_row(db):
    """An edit of an already-synced row must adopt onto the SAME local
    row, never insert a second mirror for the same definition."""
    def_id = db.create_automation_definition(
        "server:1", "recurring_question", "Original", server_id="srv-def-1"
    )
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-2",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.update_automation_definition.return_value = _server_definition_echo(
        name="Updated standup question"
    )

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    outcome = await svc.save_definition(
        _definition_payload(name="Updated standup question"),
        "server:1",
        definition_id=def_id,
    )

    assert outcome.status == "saved"
    assert outcome.definition_id == def_id
    assert len(db.list_automation_definitions(owner_id="server:1")) == 1
    row = db.get_automation_definition(def_id)
    assert row["name"] == "Updated standup question"
    server_client.update_automation_definition.assert_awaited_once_with(
        "srv-def-1", "prev-2"
    )

    # The outgoing preview request carried the local row's version, per
    # the server's required_for_update check (Task 3's handoff note).
    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["definition_version"] == 1
    assert request["definition_id"] == "srv-def-1"


@pytest.mark.asyncio
async def test_save_definition_server_owner_online_preview_rejected_writes_nothing(db):
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "status": "invalid",
        "validation_errors": [
            {"field": "name", "code": "required", "message": "Name is required."}
        ],
    }
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "invalid"
    assert outcome.errors[0]["field"] == "name"
    assert db.list_automation_definitions(owner_id="server:1") == []
    server_client.create_automation_definition.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_definition_server_owner_offline_create_queues_one_mutation(db):
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "queued"
    assert outcome.definition_id
    row = db.get_automation_definition(outcome.definition_id)
    assert row["owner_id"] == "server:1"
    assert row["server_id"] is None
    assert row["lifecycle"] == "configured"

    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["local_id"] == outcome.definition_id
    assert pending[0]["payload"]["action"] == "create"
    assert pending[0]["payload"]["server_definition_id"] is None
    assert (
        pending[0]["payload"]["definition_payload"]["family"] == "recurring_question"
    )


@pytest.mark.asyncio
async def test_save_definition_server_owner_offline_create_is_a_single_atomic_db_call(
    db,
):
    """Same atomicity pin as `review_automation_result`'s: the row write
    and its outbox mutation must be ONE DB call -- not a separate
    ``record_pending_mutation`` call after -- so the two can never land in
    different transactions."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    real_create = db.create_automation_definition
    calls: list[dict] = []

    def _spy_create(*args, **kwargs):
        calls.append(kwargs)
        return real_create(*args, **kwargs)

    db.create_automation_definition = _spy_create  # type: ignore[method-assign]
    db.record_pending_mutation = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError(
            "record_pending_mutation must not be called separately from "
            "save_definition -- it must go through "
            "create_automation_definition(pending_mutation=...)"
        )
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "queued"
    assert len(calls) == 1
    assert calls[0]["pending_mutation"]["owner_id"] == "server:1"
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_save_definition_server_owner_offline_edit_queues_update_mutation(db):
    def_id = db.create_automation_definition(
        "server:1", "recurring_question", "Original", server_id="srv-def-9"
    )
    db.update_automation_definition(def_id, name="Original v2")  # bump version to 2

    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(
        _definition_payload(name="Updated standup question"),
        "server:1",
        definition_id=def_id,
    )

    assert outcome.status == "queued"
    assert outcome.definition_id == def_id
    assert len(db.list_automation_definitions(owner_id="server:1")) == 1
    row = db.get_automation_definition(def_id)
    assert row["name"] == "Updated standup question"

    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "update"
    assert pending[0]["payload"]["server_definition_id"] == "srv-def-9"
    assert pending[0]["payload"]["definition_payload"]["definition_version"] == 2


@pytest.mark.asyncio
async def test_save_definition_server_owner_offline_invalid_writes_nothing(db):
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(name=""), "server:1")

    assert outcome.status == "invalid"
    assert db.list_automation_definitions(owner_id="server:1") == []
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_save_definition_server_owner_commit_failure_falls_back_offline(db):
    """The seam can fail AFTER a valid preview too (the commit call
    itself) -- that must fall back to the same offline path as a preview
    failure, not raise or silently drop the save."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "queued"
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_save_definition_missing_family_rejected_not_crashed(db):
    """`AutomationFamily(None)` raises `ValueError` internally -- the guard
    must catch it and reject cleanly, not propagate."""
    svc = SchedulingService(db=db, runtime_source="local")
    payload = _definition_payload()
    del payload["family"]

    outcome = await svc.save_definition(payload, "local")

    assert outcome.status == "invalid"
    assert outcome.errors[0]["field"] == "family"
    assert db.list_automation_definitions(owner_id="local") == []
