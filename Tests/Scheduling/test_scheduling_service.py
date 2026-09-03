"""Tests for SchedulingService local/server routing and offline behavior."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services import SchedulingServerClient, SchedulingService
from tldw_chatbook.Scheduling.services import scheduling_service as scheduling_service_module
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientPolicyError,
    ServerClientValidationError,
    ServerUnavailableError,
)
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


# --- spans-owners `owner_id` parameter (schedules-redesign PR-2, Task 1) ---
#
# `list_tasks()` (zero args, the default `...` sentinel) must keep behaving
# EXACTLY as every test above pins it -- these add the new opt-in shape
# without touching that default path.


@pytest.mark.asyncio
async def test_list_tasks_default_still_scopes_to_current_owner(db):
    """Byte-for-byte preservation: calling with no argument at all must
    still scope reminders to `self.owner_id`, same as before this
    parameter existed."""
    svc = SchedulingService(db=db, runtime_source="local")
    db.create_reminder_task(
        owner_id="local",
        title="Local reminder",
        schedule_kind="one_time",
        run_at="2026-07-20T14:00:00+00:00",
    )
    db.create_reminder_task(
        owner_id="server:9",
        title="Server reminder",
        schedule_kind="one_time",
        run_at="2026-07-21T14:00:00+00:00",
    )

    tasks = await svc.list_tasks()

    assert [t.title for t in tasks] == ["Local reminder"]


@pytest.mark.asyncio
async def test_list_tasks_owner_id_none_spans_every_owner(db):
    """The new spans-owners seam: `owner_id=None` returns reminders from
    EVERY owner, not just `self.owner_id` -- the redesign's unified Queue
    list (plan ruling, survey SS2's cross-owner listing gap)."""
    svc = SchedulingService(db=db, runtime_source="local")
    db.create_reminder_task(
        owner_id="local",
        title="Local reminder",
        schedule_kind="one_time",
        run_at="2026-07-20T14:00:00+00:00",
    )
    db.create_reminder_task(
        owner_id="server:9",
        title="Server reminder",
        schedule_kind="one_time",
        run_at="2026-07-21T14:00:00+00:00",
    )

    tasks = await svc.list_tasks(owner_id=None)

    assert {t.title for t in tasks} == {"Local reminder", "Server reminder"}


@pytest.mark.asyncio
async def test_list_tasks_owner_id_explicit_string_scopes_to_that_owner(db):
    """A specific owner id (not the service's current one) scopes
    reminders to just that owner."""
    svc = SchedulingService(db=db, runtime_source="local")
    db.create_reminder_task(
        owner_id="local",
        title="Local reminder",
        schedule_kind="one_time",
        run_at="2026-07-20T14:00:00+00:00",
    )
    db.create_reminder_task(
        owner_id="server:9",
        title="Server reminder",
        schedule_kind="one_time",
        run_at="2026-07-21T14:00:00+00:00",
    )

    tasks = await svc.list_tasks(owner_id="server:9")

    assert [t.title for t in tasks] == ["Server reminder"]


@pytest.mark.asyncio
async def test_list_tasks_spans_owners_keeps_projections_scoped_to_current_owner(db):
    """Watchlist/briefing `list_jobs` only STAMP the owner id onto every
    row (their underlying read has no per-owner filter) -- passing them
    `owner_id=None` would fail their `ScheduledTask.owner_id: str` field.
    A spans-owners reminder listing must not do that: projections stay
    scoped to `self.owner_id` regardless of the reminder-side argument."""
    svc = SchedulingService(db=db, runtime_source="local")
    projection = MagicMock(spec=WatchlistProjection)
    projection.list_jobs.return_value = []
    svc.watchlist_projection = projection

    tasks = await svc.list_tasks(owner_id=None)

    assert tasks == []
    projection.list_jobs.assert_called_once_with(owner_id="local")


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
async def test_list_tasks_include_projections_false_skips_the_projection_reads(db):
    """redesign PR-2 Task 2 review, finding 2: the unified Queue's
    `load_tasks` immediately discards every `ScheduledTask` row, so
    `include_projections=False` must stop `list_tasks` from calling
    `list_jobs` on either projection at all -- not just from returning
    their rows. `list_jobs.assert_not_called()` is the counting-fake pin
    the review asked for; the DEFAULT (`include_projections=True`, the
    existing tests above) must keep calling both.
    """
    svc = SchedulingService(db=db, runtime_source="local")
    await svc.create_reminder(_reminder_payload("Reminder"))

    watchlist_projection = MagicMock(spec=WatchlistProjection)
    briefing_projection = MagicMock(spec=BriefingProjection)
    svc.watchlist_projection = watchlist_projection
    svc.briefing_projection = briefing_projection

    tasks = await svc.list_tasks(include_projections=False)

    watchlist_projection.list_jobs.assert_not_called()
    briefing_projection.list_jobs.assert_not_called()
    assert len(tasks) == 1
    assert isinstance(tasks[0], ReminderTask)
    assert tasks[0].title == "Reminder"


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
async def test_save_definition_local_create_strips_derived_resolved_sources(db):
    """Task 6 E2E finding: a default (`all_searchable_library`) scope's
    preview-normalized form injects a `resolved_sources` projection that
    `normalize_recurring_question_scope` itself does not accept as input
    (`SUPPORTED_SCOPE_FIELDS` has no such field). Persisting it verbatim
    made every later re-normalization of this row's stored scope --
    `automation_execution.py`'s dispatch, `automation_health.py`'s
    sources-readable check -- report a spurious "unsupported field" error
    and degrade every scheduled run. The stored row must not carry it."""
    from tldw_chatbook.Scheduling.recurring_question_scope import (
        normalize_recurring_question_scope,
    )

    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.save_definition(_definition_payload(), "local")

    row = db.get_automation_definition(outcome.definition_id)
    stored_scope = row["config"]["scope"]
    assert "resolved_sources" not in stored_scope
    assert stored_scope["mode"] == "all_searchable_library"
    # And the stored value must be safe to re-normalize, the way a real
    # scheduled dispatch or health check does.
    _normalized, errors, _warnings = normalize_recurring_question_scope(stored_scope)
    assert errors == []


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
async def test_save_definition_server_owner_online_sends_server_vocab_schedule(db):
    """Fix-round 1, finding 2 (task-3-review.md): the online server-owner
    branch was sending the CLIENT-vocab schedule straight to the server's
    preview, which passes `_validate_schedule` (kind-only) and then never
    arms server-side. Only the network-bound copy must be translated."""
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

    await svc.save_definition(_definition_payload(), "server:1")

    request = server_client.preview_automation_definition.await_args.args[0]
    assert request["schedule"] == {"kind": "interval", "seconds": 3600}


@pytest.mark.asyncio
async def test_save_definition_server_owner_offline_fallback_keeps_client_vocab_schedule(
    db,
):
    """The offline-fallback leg (same payload as the test above) must NOT
    translate: `request` also feeds the local pure preview and is queued
    verbatim as the pending mutation's `definition_payload`, which
    `SyncEngine` translates at push time (task 3) -- translating it here
    too would make the queued payload lie about its own vocabulary."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "queued"
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert pending[0]["payload"]["definition_payload"]["schedule"] == {
        "kind": "interval",
        "every_seconds": 3600,
    }


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
    """Final review C1: the queued `definition_version` must stay equal to
    the version the SERVER holds (the local column is a mirror of it), and
    an offline edit must not move that mirror -- the server checks it for
    exact equality and a drifted value is rejected (409) forever."""
    server_version = 7
    db.upsert_automation_definitions_from_server(
        "server:1",
        [
            _server_definition_echo(
                id="srv-def-9", name="Original", version=server_version
            )
        ],
    )
    def_id = db.get_automation_definition_by_server_id("server:1", "srv-def-9")["id"]

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
    assert row["version"] == server_version  # the mirror did not drift

    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "update"
    assert pending[0]["payload"]["server_definition_id"] == "srv-def-9"
    assert (
        pending[0]["payload"]["definition_payload"]["definition_version"]
        == server_version
    )

    # A SECOND offline edit REPLACES that mutation (pending_mutations is
    # UNIQUE(local_id, primitive, owner_id)) -- the replacement must still
    # carry the server's version, not a locally-bumped one.
    outcome = await svc.save_definition(
        _definition_payload(name="Updated again"), "server:1", definition_id=def_id
    )

    assert outcome.status == "queued"
    assert db.get_automation_definition(def_id)["version"] == server_version
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert (
        pending[0]["payload"]["definition_payload"]["definition_version"]
        == server_version
    )


@pytest.mark.asyncio
async def test_save_definition_server_owner_server_rejection_is_error_not_queued(db):
    """Final review C2: a server-side 4xx (here a 409 version conflict,
    mapped to `ServerClientValidationError`) is deterministic -- replaying
    it hits the identical refusal, so the save must report `error` and
    write/queue nothing, exactly like the policy refusal it sits beside."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = (
        ServerClientValidationError("scheduled_task_definition_version_conflict")
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "server_rejected"
    assert "version_conflict" in outcome.errors[0]["message"]
    assert db.list_automation_definitions(owner_id="server:1") == []
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_save_definition_server_owner_commit_rejection_is_error_not_queued(db):
    """Same as above for the COMMIT call (the preview was accepted) -- the
    PATCH is where the real 409s live."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.side_effect = ServerClientValidationError(
        "scheduled_task_schedule_invalid"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "server_rejected"
    assert db.list_automation_definitions(owner_id="server:1") == []
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_save_definition_edit_preserves_fields_the_payload_does_not_carry(db):
    """Final review I4: the v1 modal exposes neither `description` nor the
    visibility/approval/retention policies (nor `input.max_tokens`), so a
    rename must not wipe them -- in the DB row OR in the update payload
    that goes to the server."""
    db.upsert_automation_definitions_from_server(
        "server:1",
        [
            _server_definition_echo(
                id="srv-def-5",
                version=3,
                description="Digest of everything that changed",
                visibility_policy={"mode": "metadata_only"},
                approval_policy={"mode": "manual"},
                config={
                    "scope": {"mode": "all_searchable_library"},
                    "retention_policy": {"mode": "custom", "keep_days": 30},
                },
                input={"question": "What changed?", "max_tokens": 4096},
                notification_policy={"on_success": True, "channels": ["email"]},
            )
        ],
    )
    def_id = db.get_automation_definition_by_server_id("server:1", "srv-def-5")["id"]

    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    # The v1 form's payload shape: no description, no policies, no max_tokens.
    outcome = await svc.save_definition(
        {
            "family": "recurring_question",
            "mode": "update",
            "name": "Renamed",
            "input": {"question": "What changed?", "provider": None, "model": None},
            "schedule": {"kind": "interval", "every_seconds": 3600},
            "config": {
                "scope": {"mode": "all_searchable_library"},
                "generation_mode": "optional",
                "finding_policy": {"preset": "balanced_findings"},
            },
            "notification_policy": {"on_success": False, "on_failure": False},
        },
        "server:1",
        definition_id=def_id,
    )

    assert outcome.status == "queued"
    row = db.get_automation_definition(def_id)
    assert row["name"] == "Renamed"
    assert row["description"] == "Digest of everything that changed"
    assert row["visibility_policy"] == {"mode": "metadata_only"}
    assert row["approval_policy"] == {"mode": "manual"}
    assert row["config"]["retention_policy"] == {"mode": "custom", "keep_days": 30}
    assert row["input"]["max_tokens"] == 4096
    assert row["notification_policy"]["channels"] == ["email"]
    # The form's own fields still win over the stored ones.
    assert row["notification_policy"]["on_success"] is False
    assert row["input"].get("provider") is None

    outgoing = db.get_pending_mutations(
        "server:1", primitive="automation_definition"
    )[0]["payload"]["definition_payload"]
    assert outgoing["description"] == "Digest of everything that changed"
    assert outgoing["visibility_policy"] == {"mode": "metadata_only"}
    assert outgoing["approval_policy"] == {"mode": "manual"}
    assert outgoing["config"]["retention_policy"] == {"mode": "custom", "keep_days": 30}
    assert outgoing["input"]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_save_definition_edit_still_clears_a_field_the_payload_carries(db):
    """The merge fills OMITTED keys only -- a payload that does carry a
    field (here `input.provider`, which the form exposes and emits as
    `None` when blank) still clears the stored value."""
    def_id = db.create_automation_definition(
        "local",
        "recurring_question",
        "Pinned",
        input={"question": "What changed?", "provider": "openai", "model": "gpt-5"},
        schedule={"kind": "interval", "every_seconds": 3600},
    )
    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.save_definition(
        _definition_payload(
            name="Pinned",
            input={"question": "What changed?", "provider": None, "model": None},
        ),
        "local",
        definition_id=def_id,
    )

    assert outcome.status == "saved"
    row = db.get_automation_definition(def_id)
    assert row["input"]["provider"] is None
    assert row["input"]["model"] is None


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


# ----------------------------------------------------------------------
# preview_definition / save_definition fix round 1: ServerClientPolicyError
# handling + stale-mutation clearing on a successful online save
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_definition_server_owner_policy_denied_uses_policy_wording(db):
    """review finding 3: a deterministic policy refusal must not be
    reported with connectivity wording ("could not reach the server")
    since retrying will never change the outcome."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerClientPolicyError(
        "automation authoring is disabled for this account"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    preview = await svc.preview_definition(_definition_payload(), "server:1")

    assert preview.status == "valid"  # local validation still runs offline
    warning = next(w for w in preview.warnings if w["field"] == "_owner")
    assert warning["code"] == "policy_denied"
    assert "could not reach" not in warning["message"].lower()


@pytest.mark.asyncio
async def test_save_definition_server_owner_preview_policy_denied_returns_error(db):
    """review finding 1: a policy refusal on the preview call must report
    status="error" and write nothing -- NOT fall back to the offline
    queue, since a replay would hit the identical refusal and SyncEngine
    swallows it silently (the save would be "queued" forever)."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerClientPolicyError(
        "automation authoring is disabled for this account"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "policy_denied"
    assert db.list_automation_definitions(owner_id="server:1") == []
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_save_definition_server_owner_commit_policy_denied_returns_error(db):
    """Same as above, but the refusal happens on the commit call (preview
    was valid) rather than the preview call itself."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    server_client.create_automation_definition.side_effect = ServerClientPolicyError(
        "automation authoring is disabled for this account"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )

    outcome = await svc.save_definition(_definition_payload(), "server:1")

    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "policy_denied"
    assert db.list_automation_definitions(owner_id="server:1") == []
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_save_definition_online_success_clears_stale_offline_mutation(db):
    """review finding 2: a successful ONLINE save must clear any pending
    `automation_definition` mutation left queued by an earlier offline
    save on the same row -- otherwise the next sync replays the stale
    mutation, creating a duplicate server-side definition (never-synced
    row) or silently reverting this save's newer edit (already-synced
    row)."""
    offline_client = AsyncMock()
    offline_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=offline_client, runtime_source="server:1"
    )

    queued = await svc.save_definition(_definition_payload(), "server:1")
    assert queued.status == "queued"
    assert (
        len(db.get_pending_mutations("server:1", primitive="automation_definition"))
        == 1
    )

    online_client = AsyncMock()
    online_client.preview_automation_definition.return_value = {
        "id": "prev-1",
        "status": "valid",
        "validation_errors": [],
    }
    online_client.create_automation_definition.return_value = _server_definition_echo()
    svc.server_client = online_client

    outcome = await svc.save_definition(
        _definition_payload(name="Updated standup question"),
        "server:1",
        definition_id=queued.definition_id,
    )

    assert outcome.status == "saved"
    assert outcome.definition_id == queued.definition_id
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []
    # No duplicate row: the online save adopted the SAME local row.
    rows = db.list_automation_definitions(owner_id="server:1")
    assert len(rows) == 1
    assert rows[0]["server_id"] == "srv-def-1"


@pytest.mark.asyncio
async def test_create_reminder_targets_another_owner_without_flipping_owner_id(db):
    """Qodo HIGH: a cross-owner save threads its owner through the call
    instead of flipping the service's shared `owner_id` around the awaited
    network round-trip -- concurrent workers (sync, refresh, run-now) read
    that attribute and must never observe the temporary owner."""
    observed: list[str] = []

    class _ObservingClient:
        notifications_service = object()

        async def create_reminder(self, **payload):
            # Stands in for any concurrent worker reading the shared owner
            # while this call is in flight.
            observed.append(svc.owner_id)
            raise ServerUnavailableError("offline")

    svc = SchedulingService(
        db=db, server_client=_ObservingClient(), runtime_source="local"
    )

    task = await svc.create_reminder(
        _reminder_payload("Server reminder"), owner_id="server:1"
    )

    assert observed == ["local"], "the active owner must stay untouched"
    assert svc.owner_id == "local"
    assert svc.sync_engine.owner_id == "local"
    # The row and its queued push both land under the TARGET owner.
    rows = db.list_reminder_tasks(owner_id="server:1")
    assert [row["id"] for row in rows] == [task.id]
    assert db.list_reminder_tasks(owner_id="local") == []
    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"


@pytest.mark.asyncio
async def test_update_reminder_targets_another_owner_without_flipping_owner_id(db):
    """Same threading rule on the update path."""
    svc = SchedulingService(db=db, runtime_source="local")
    created = await svc.create_reminder(_reminder_payload("Server reminder"))
    # Already known to the server, so the update takes the PATCH branch.
    db.update_reminder_task(created.id, server_id="srv-1")

    offline_client = AsyncMock()
    offline_client.notifications_service = object()
    offline_client.update_reminder.side_effect = ServerUnavailableError("offline")
    svc.server_client = offline_client

    await svc.update_reminder(created.id, {"title": "Renamed"}, owner_id="server:1")

    assert svc.owner_id == "local"
    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "update"
    assert db.get_reminder_task(created.id)["title"] == "Renamed"


@pytest.mark.asyncio
async def test_reminder_owner_defaults_to_the_active_owner(db):
    """The new parameter is optional: every pre-existing caller is unchanged."""
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Default owner"))
    assert db.get_reminder_task(task.id)["owner_id"] == "local"


# ----------------------------------------------------------------------
# Transfer machine facade (schedules-handoff PR-5, Task 6, spec §6)
# ----------------------------------------------------------------------


class _FakeApp:
    """Stands in for the app -- only `active_server_id` is read by the
    transfer facade (`_active_server_owner_id`)."""

    def __init__(self, active_server_id="1"):
        self.active_server_id = active_server_id


def _connected_server_client():
    """An AsyncMock server client that reads as "a server is connected"
    (`transfer_refusal`'s `notifications_service is not None` check)."""
    client = AsyncMock()
    client.notifications_service = object()
    return client


def _transfer_service(db, *, server_client=None, active_server_id="1", **kwargs):
    app = _FakeApp(active_server_id) if active_server_id is not None else None
    kwargs.setdefault("runtime_source", "local")
    return SchedulingService(
        db=db,
        server_client=server_client,
        app_getter=(lambda: app) if app is not None else None,
        **kwargs,
    )


def _make_reminder(db, **overrides):
    kwargs = dict(
        owner_id="local",
        title="Reminder",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return db.create_reminder_task(**kwargs)


def _make_definition(db, **overrides):
    kwargs = dict(
        owner_id="local",
        family="recurring_question",
        name="Daily Q",
        schedule={"kind": "interval", "every_seconds": 3600},
        input={"question": "What happened today?"},
        config={},
    )
    kwargs.update(overrides)
    return db.create_automation_definition(**kwargs)


def _stub_health(monkeypatch, health="ready", reason=""):
    monkeypatch.setattr(
        scheduling_service_module,
        "compute_local_health",
        lambda app, row: (health, reason),
    )


# -- transfer_refusal ----------------------------------------------------


def test_transfer_refusal_no_server_connection(db):
    svc = _transfer_service(db, server_client=None)
    row = db.get_reminder_task(_make_reminder(db))
    for direction in ("to_server", "to_local"):
        reason = svc.transfer_refusal(row, direction)
        assert reason == "No server connection is configured."


def test_transfer_refusal_to_server_already_server_owned(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    row = db.get_reminder_task(_make_reminder(db, owner_id="server:1", server_id="srv-1"))
    reason = svc.transfer_refusal(row, "to_server")
    assert reason == "This row already lives on the server."


def test_transfer_refusal_to_server_no_active_server_identity(db):
    svc = _transfer_service(
        db, server_client=_connected_server_client(), active_server_id=None
    )
    row = db.get_reminder_task(_make_reminder(db))
    reason = svc.transfer_refusal(row, "to_server")
    assert reason == "No server identity is configured."


def test_transfer_refusal_to_local_not_server_owned(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    row = db.get_reminder_task(_make_reminder(db))
    reason = svc.transfer_refusal(row, "to_local")
    assert reason == "This row is not server-owned."


def test_transfer_refusal_to_local_missing_server_id(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    row = db.get_reminder_task(_make_reminder(db, owner_id="server:1"))
    reason = svc.transfer_refusal(row, "to_local")
    assert reason == "This row is not server-owned."


@pytest.mark.parametrize("direction", ["to_server", "to_local"])
def test_transfer_refusal_already_in_progress(db, direction):
    svc = _transfer_service(db, server_client=_connected_server_client())
    if direction == "to_server":
        row = db.get_reminder_task(
            _make_reminder(db, transfer_state="to_server_pending")
        )
    else:
        row = db.get_reminder_task(
            _make_reminder(
                db,
                owner_id="server:1",
                server_id="srv-1",
                transfer_state="from_server_pending",
            )
        )
    reason = svc.transfer_refusal(row, direction)
    assert reason == "A transfer is already in progress on this row."


@pytest.mark.parametrize("lifecycle", ["archived", "solved"])
def test_transfer_refusal_lifecycle_not_transferable(db, lifecycle):
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db, lifecycle=lifecycle)
    row = db.get_automation_definition(definition_id)
    reason = svc.transfer_refusal(row, "to_server")
    assert reason == f"This automation is {lifecycle} and cannot transfer."


def test_transfer_refusal_to_local_agent_task_always_refuses(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(
        db,
        owner_id="server:1",
        server_id="srv-def-1",
        family="agent_task",
    )
    row = db.get_automation_definition(definition_id)
    reason = svc.transfer_refusal(row, "to_local")
    assert reason == "Agent-task automations cannot run locally yet."


def test_transfer_refusal_to_local_recurring_question_quotes_health_reason(
    db, monkeypatch
):
    svc = _transfer_service(db, server_client=_connected_server_client())
    _stub_health(monkeypatch, health="permission_required", reason="No provider configured.")
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")
    row = db.get_automation_definition(definition_id)
    reason = svc.transfer_refusal(row, "to_local")
    assert reason == "No provider configured."


def test_transfer_refusal_allows_happy_path_both_directions(db, monkeypatch):
    svc = _transfer_service(db, server_client=_connected_server_client())
    _stub_health(monkeypatch, health="ready")

    to_server_row = db.get_reminder_task(_make_reminder(db))
    assert svc.transfer_refusal(to_server_row, "to_server") is None

    to_local_def = _make_definition(db, owner_id="server:1", server_id="srv-def-1")
    to_local_row = db.get_automation_definition(to_local_def)
    assert svc.transfer_refusal(to_local_row, "to_local") is None


# -- transfer_warnings -----------------------------------------------------


def test_transfer_warnings_reminder_imminent_one_time_warns(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    soon = (datetime.now(timezone.utc)).isoformat()
    row = db.get_reminder_task(
        _make_reminder(db, schedule_kind="one_time", run_at=soon)
    )
    warnings = svc.transfer_warnings(row, "to_server")
    assert any("5 minutes" in w for w in warnings)


def test_transfer_warnings_reminder_distant_one_time_no_warning(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    row = db.get_reminder_task(
        _make_reminder(db, schedule_kind="one_time", run_at="2099-01-01T00:00:00+00:00")
    )
    warnings = svc.transfer_warnings(row, "to_server")
    assert warnings == []


def test_transfer_warnings_reminder_timeout_seconds_warns(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    row = db.get_reminder_task(
        _make_reminder(
            db,
            schedule_kind="one_time",
            run_at="2099-01-01T00:00:00+00:00",
            timeout_seconds=30,
        )
    )
    warnings = svc.transfer_warnings(row, "to_server")
    assert any("timeout_seconds" in w for w in warnings)


def test_transfer_warnings_definition_imminent_one_time_warns(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    soon = datetime.now(timezone.utc).isoformat()
    definition_id = _make_definition(
        db, schedule={"kind": "one_time", "run_at": soon}
    )
    row = db.get_automation_definition(definition_id)
    warnings = svc.transfer_warnings(row, "to_server")
    assert any("5 minutes" in w for w in warnings)


def test_transfer_warnings_definition_never_names_timeout_seconds(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db)
    row = db.get_automation_definition(definition_id)
    warnings = svc.transfer_warnings(row, "to_server")
    assert warnings == []


# -- begin_transfer_to_server ----------------------------------------------


@pytest.mark.asyncio
async def test_begin_transfer_to_server_not_found(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    outcome = await svc.begin_transfer_to_server("reminder_task", "no-such-id")
    assert outcome.status == "not_found"


@pytest.mark.asyncio
async def test_begin_transfer_to_server_refused_records_nothing(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")

    outcome = await svc.begin_transfer_to_server("reminder_task", reminder_id)

    assert outcome.status == "refused"
    assert outcome.reason == "This row already lives on the server."
    assert db.get_reminder_task(reminder_id)["transfer_state"] is None
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_begin_transfer_to_server_reminder_happy_path(db):
    notified = []
    svc = _transfer_service(
        db,
        server_client=_connected_server_client(),
        on_queue_changed=lambda: notified.append(True),
    )
    reminder_id = _make_reminder(db, timeout_seconds=45)

    outcome = await svc.begin_transfer_to_server("reminder_task", reminder_id)

    assert outcome.status == "pending"
    row = db.get_reminder_task(reminder_id)
    assert row["transfer_state"] == "to_server_pending"
    assert row["owner_id"] == "local"  # still executes locally while queued

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    payload = pending[0]["payload"]
    assert payload["action"] == "transfer_to_server"
    # link_type/link_id are injected at push time (SyncEngine), not here.
    assert "link_type" not in payload["task_payload"]
    assert "timeout_seconds" not in payload["task_payload"]
    assert notified == [True]


@pytest.mark.asyncio
async def test_begin_transfer_to_server_definition_happy_path(db, monkeypatch):
    _stub_health(monkeypatch, health="ready")
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db)

    outcome = await svc.begin_transfer_to_server("automation_definition", definition_id)

    assert outcome.status == "pending"
    row = db.get_automation_definition(definition_id)
    assert row["transfer_state"] == "to_server_pending"

    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    payload = pending[0]["payload"]
    assert payload["action"] == "transfer_to_server"
    assert payload["definition_payload"]["family"] == "recurring_question"
    assert payload["definition_payload"]["schedule"] == {
        "kind": "interval",
        "every_seconds": 3600,
    }


@pytest.mark.asyncio
async def test_begin_transfer_to_server_no_active_identity_refuses(db):
    svc = _transfer_service(
        db, server_client=_connected_server_client(), active_server_id=None
    )
    reminder_id = _make_reminder(db)
    outcome = await svc.begin_transfer_to_server("reminder_task", reminder_id)
    assert outcome.status == "refused"
    assert outcome.reason == "No server identity is configured."


@pytest.mark.asyncio
async def test_begin_transfer_to_server_cas_race_refuses(db):
    """A concurrent begin (or an already-in-flight state) loses the CAS."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_pending", expected=(None,)
    )
    # transfer_refusal already catches this (transfer_state is set), but
    # pin the CAS backstop directly too: force the row's state out from
    # under transfer_refusal's own check by bypassing it.
    monkey_row = dict(db.get_reminder_task(reminder_id))
    monkey_row["transfer_state"] = None  # pretend the refusal gate saw a stale read
    reason = svc.transfer_refusal(monkey_row, "to_server")
    assert reason is None  # confirms the CAS, not the gate, is what would catch this
    outcome = await svc.begin_transfer_to_server("reminder_task", reminder_id)
    assert outcome.status == "refused"
    assert outcome.reason == "A transfer is already in progress on this row."


# -- begin_transfer_to_local ------------------------------------------------


@pytest.mark.asyncio
async def test_begin_transfer_to_local_not_found(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    outcome = await svc.begin_transfer_to_local("reminder_task", "no-such-id")
    assert outcome.status == "not_found"


@pytest.mark.asyncio
async def test_begin_transfer_to_local_refused_not_a_mirror(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    outcome = await svc.begin_transfer_to_local("reminder_task", reminder_id)
    assert outcome.status == "refused"
    assert outcome.reason == "This row is not server-owned."


@pytest.mark.asyncio
async def test_begin_transfer_to_local_reminder_happy_path(db):
    notified = []
    svc = _transfer_service(
        db,
        server_client=_connected_server_client(),
        on_queue_changed=lambda: notified.append(True),
    )
    mirror_id = _make_reminder(
        db, owner_id="server:1", server_id="srv-1", schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )

    outcome = await svc.begin_transfer_to_local("reminder_task", mirror_id)

    assert outcome.status == "pending"
    copy_id = outcome.row_id
    assert copy_id is not None and copy_id != mirror_id

    # Mirror untouched, still executing server-side.
    mirror_row = db.get_reminder_task(mirror_id)
    assert mirror_row["transfer_state"] is None
    assert mirror_row["owner_id"] == "server:1"

    copy_row = db.get_reminder_task(copy_id)
    assert copy_row["owner_id"] == "local"
    assert copy_row["transfer_state"] == "from_server_pending"
    assert copy_row["server_id"] is None

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["local_id"] == mirror_id  # keyed by the MIRROR (obligation 2)
    payload = pending[0]["payload"]
    assert payload["action"] == "release_from_server"
    assert payload["server_task_id"] == "srv-1"
    assert payload["local_copy_id"] == copy_id
    assert notified == [True]


@pytest.mark.asyncio
async def test_begin_transfer_to_local_definition_happy_path(db, monkeypatch):
    _stub_health(monkeypatch, health="ready")
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    outcome = await svc.begin_transfer_to_local("automation_definition", mirror_id)

    assert outcome.status == "pending"
    copy_row = db.get_automation_definition(outcome.row_id)
    assert copy_row["owner_id"] == "local"
    assert copy_row["transfer_state"] == "from_server_pending"

    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["local_id"] == mirror_id
    assert pending[0]["payload"]["server_definition_id"] == "srv-def-1"
    assert pending[0]["payload"]["local_copy_id"] == outcome.row_id


# -- cancel_transfer ---------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_transfer_not_found(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    outcome = await svc.cancel_transfer("reminder_task", "no-such-id")
    assert outcome.status == "not_found"


@pytest.mark.asyncio
async def test_cancel_transfer_unattempted_to_server_pending(db):
    notified = []
    svc = _transfer_service(
        db,
        server_client=_connected_server_client(),
        on_queue_changed=lambda: notified.append(True),
    )
    reminder_id = _make_reminder(db)
    begin_outcome = await svc.begin_transfer_to_server("reminder_task", reminder_id)
    assert begin_outcome.status == "pending"
    notified.clear()

    outcome = await svc.cancel_transfer("reminder_task", reminder_id)

    assert outcome.status == "cancelled"
    row = db.get_reminder_task(reminder_id)
    assert row["transfer_state"] is None
    assert row["owner_id"] == "local"
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []
    assert notified == [True]


@pytest.mark.asyncio
async def test_cancel_transfer_settled_definitive_failure_to_server_failed(db):
    """A retained, definitively-failed transfer mutation (transfer_errors
    embedded, SyncEngine's own settlement shape) is still cancelable."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_failed", expected=(None,)
    )
    db.record_pending_mutation(
        reminder_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {}, "transfer_errors": ["boom"]},
    )

    outcome = await svc.cancel_transfer("reminder_task", reminder_id)

    assert outcome.status == "cancelled"
    assert db.get_reminder_task(reminder_id)["transfer_state"] is None
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_cancel_transfer_to_server_sent_too_late(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_sent", expected=(None,)
    )

    outcome = await svc.cancel_transfer("reminder_task", reminder_id)

    assert outcome.status == "refused"
    assert "reverse transfer" in outcome.reason
    # Untouched -- still sent, not reverted.
    assert db.get_reminder_task(reminder_id)["transfer_state"] == "to_server_sent"


@pytest.mark.asyncio
async def test_cancel_transfer_no_transfer_in_progress_too_late(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    outcome = await svc.cancel_transfer("reminder_task", reminder_id)
    assert outcome.status == "refused"
    assert "reverse transfer" in outcome.reason


@pytest.mark.asyncio
async def test_cancel_transfer_release_unpushed_deletes_copy(db):
    notified = []
    svc = _transfer_service(
        db,
        server_client=_connected_server_client(),
        on_queue_changed=lambda: notified.append(True),
    )
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")
    begin_outcome = await svc.begin_transfer_to_local("reminder_task", mirror_id)
    copy_id = begin_outcome.row_id
    notified.clear()

    outcome = await svc.cancel_transfer("reminder_task", copy_id)

    assert outcome.status == "cancelled"
    assert db.get_reminder_task(copy_id) is None  # dormant copy deleted
    assert db.get_reminder_task(mirror_id) is not None  # server unaffected
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []
    assert notified == [True]


@pytest.mark.asyncio
async def test_cancel_transfer_release_definitively_failed_no_live_mutation(db):
    """Obligation 3: a definitively-rejected release leaves `from_server_
    pending` with NO live mutation (SyncEngine's own reject-and-clear
    settlement) -- cancel must key off transfer_state, not mutation
    existence, and still recover the dormant copy."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")
    begin_outcome = await svc.begin_transfer_to_local("reminder_task", mirror_id)
    copy_id = begin_outcome.row_id

    # Simulate SyncEngine's settlement of a definitive release failure:
    # the mutation is gone, but the copy stays from_server_pending (no
    # automatic path back to armed -- Task 5's own documented concern).
    db.delete_pending_mutation_for_record(mirror_id, "reminder_task", "server:1")
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []
    assert db.get_reminder_task(copy_id)["transfer_state"] == "from_server_pending"

    outcome = await svc.cancel_transfer("reminder_task", copy_id)

    assert outcome.status == "cancelled"
    assert db.get_reminder_task(copy_id) is None


# -- recover_inflight_transfers ----------------------------------------------


@pytest.mark.asyncio
async def test_recover_inflight_transfers_definition_cas_back_to_pending(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db, transfer_state="to_server_sent")

    await svc.recover_inflight_transfers()

    assert (
        db.get_automation_definition(definition_id)["transfer_state"]
        == "to_server_pending"
    )


@pytest.mark.asyncio
async def test_recover_inflight_transfers_reminder_found_converts_to_mirror(db):
    server_client = _connected_server_client()
    server_client.list_reminders.return_value = {
        "items": [
            {
                "id": "srv-9",
                "title": "Recovered",
                "schedule_kind": "one_time",
                "run_at": "2030-01-01T00:00:00+00:00",
                "link_type": "chatbook_transfer",
                "link_id": None,  # filled in below
            }
        ]
    }
    svc = _transfer_service(db, server_client=server_client)
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")
    server_client.list_reminders.return_value["items"][0]["link_id"] = reminder_id
    db.record_pending_mutation(
        reminder_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {}},
    )

    await svc.recover_inflight_transfers()

    row = db.get_reminder_task(reminder_id)
    assert row["owner_id"] == "server:1"
    assert row["server_id"] == "srv-9"
    assert row["transfer_state"] is None
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_recover_inflight_transfers_reminder_absent_cas_back_to_pending(db):
    server_client = _connected_server_client()
    server_client.list_reminders.return_value = {"items": []}
    svc = _transfer_service(db, server_client=server_client)
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")
    db.record_pending_mutation(
        reminder_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {}},
    )

    await svc.recover_inflight_transfers()

    row = db.get_reminder_task(reminder_id)
    assert row["transfer_state"] == "to_server_pending"
    # The mutation is left in place for the normal replay to pick up.
    assert len(db.get_pending_mutations("server:1", primitive="reminder_task")) == 1


@pytest.mark.asyncio
async def test_recover_inflight_transfers_reminder_offline_leaves_row_untouched(db):
    svc = _transfer_service(db, server_client=None)  # no server connection at all
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")

    await svc.recover_inflight_transfers()

    assert db.get_reminder_task(reminder_id)["transfer_state"] == "to_server_sent"


@pytest.mark.asyncio
async def test_recover_inflight_transfers_reminder_no_active_identity_leaves_row_untouched(
    db,
):
    svc = _transfer_service(
        db, server_client=_connected_server_client(), active_server_id=None
    )
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")

    await svc.recover_inflight_transfers()

    assert db.get_reminder_task(reminder_id)["transfer_state"] == "to_server_sent"


@pytest.mark.asyncio
async def test_recover_inflight_transfers_reminder_list_error_leaves_row_untouched(db):
    server_client = _connected_server_client()
    server_client.list_reminders.side_effect = RuntimeError("offline")
    svc = _transfer_service(db, server_client=server_client)
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")

    await svc.recover_inflight_transfers()  # must not raise

    assert db.get_reminder_task(reminder_id)["transfer_state"] == "to_server_sent"


@pytest.mark.asyncio
async def test_recover_inflight_transfers_ignores_non_stuck_rows(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    pending_id = _make_reminder(db, transfer_state="to_server_pending")
    armed_id = _make_reminder(db)

    await svc.recover_inflight_transfers()

    assert db.get_reminder_task(pending_id)["transfer_state"] == "to_server_pending"
    assert db.get_reminder_task(armed_id)["transfer_state"] is None


# ----------------------------------------------------------------------
# Fix round 1 (task-6-review.md): retry leg, vanished-row cleanup,
# refusal check order
# ----------------------------------------------------------------------


def test_transfer_refusal_to_server_failed_is_not_already_in_progress(db):
    """Obligation (f): a definitively-failed transfer is retry-eligible,
    not refused as 'already in progress'."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    row = db.get_reminder_task(_make_reminder(db, transfer_state="to_server_failed"))
    assert svc.transfer_refusal(row, "to_server") is None


@pytest.mark.asyncio
async def test_begin_transfer_to_server_retries_failed_reminder_transfer(db):
    """Obligation (f): begin on a to_server_failed row CASes back to
    to_server_pending and replaces the retained mutation, stripping
    transfer_errors, so a real replay fires again on the fake."""
    server_client = _connected_server_client()
    svc = _transfer_service(db, server_client=server_client)
    reminder_id = _make_reminder(db)
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_failed", expected=(None,)
    )
    db.record_pending_mutation(
        reminder_id,
        "reminder_task",
        "server:1",
        {
            "action": "transfer_to_server",
            "task_payload": {"title": "stale"},
            "transfer_errors": ["boom"],
            "idempotency_key": "old-key",
        },
    )

    outcome = await svc.begin_transfer_to_server("reminder_task", reminder_id)

    assert outcome.status == "pending"
    row = db.get_reminder_task(reminder_id)
    assert row["transfer_state"] == "to_server_pending"
    assert row["owner_id"] == "local"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    payload = pending[0]["payload"]
    assert "transfer_errors" not in payload
    assert payload["action"] == "transfer_to_server"

    # The row is genuinely eligible for a fresh push, not just re-armed
    # locally -- drive a real sync_now() against the fake and confirm it
    # fires.
    server_client.create_reminder.return_value = {"id": "srv-99", "title": "Reminder"}
    await svc.sync_now("server:1")
    server_client.create_reminder.assert_awaited_once()
    assert db.get_reminder_task(reminder_id)["owner_id"] == "server:1"
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_begin_transfer_to_server_retry_definition(db, monkeypatch):
    """Same retry leg, definitions side."""
    _stub_health(monkeypatch, health="ready")
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db)
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_failed", expected=(None,)
    )
    db.record_pending_mutation(
        definition_id,
        "automation_definition",
        "server:1",
        {
            "action": "transfer_to_server",
            "definition_payload": {},
            "transfer_errors": ["invalid preview"],
        },
    )

    outcome = await svc.begin_transfer_to_server("automation_definition", definition_id)

    assert outcome.status == "pending"
    assert (
        db.get_automation_definition(definition_id)["transfer_state"]
        == "to_server_pending"
    )
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert "transfer_errors" not in pending[0]["payload"]


@pytest.mark.asyncio
async def test_recover_inflight_transfers_reminder_vanished_still_deletes_mutation(
    db, monkeypatch
):
    """Fix round 1, finding 2: a row that vanished between the scan and
    the convert call must not leave a dangling mutation behind."""
    server_client = _connected_server_client()
    server_client.list_reminders.return_value = {
        "items": [
            {
                "id": "srv-9",
                "title": "Recovered",
                "schedule_kind": "one_time",
                "run_at": "2030-01-01T00:00:00+00:00",
                "link_type": "chatbook_transfer",
                "link_id": None,
            }
        ]
    }
    svc = _transfer_service(db, server_client=server_client)
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")
    server_client.list_reminders.return_value["items"][0]["link_id"] = reminder_id
    db.record_pending_mutation(
        reminder_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {}},
    )
    monkeypatch.setattr(db, "convert_row_to_server_mirror", lambda *a, **k: "vanished")

    await svc.recover_inflight_transfers()

    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


def test_transfer_refusal_checks_family_health_before_lifecycle(db):
    """Fix round 1, finding 3: spec §6.4 order -- family/health (target-
    cannot-execute) is checked before lifecycle, so an archived
    agent_task mirror reports the family reason when both apply."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(
        db,
        owner_id="server:1",
        server_id="srv-def-1",
        family="agent_task",
        lifecycle="archived",
    )
    row = db.get_automation_definition(definition_id)
    reason = svc.transfer_refusal(row, "to_local")
    assert reason == "Agent-task automations cannot run locally yet."


# ---------------------------------------------------------------------------
# Final whole-branch review fixes (2026-09-02)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_to_server_offline_still_drops_the_mutation(db):
    """C2/I3: cancel keys off the mutation's OWN owner, never today's
    active server. Offline (`active_server_id` gone) the mutation used to
    survive the cancel forever, CAS-skipped every cycle."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    assert (
        await svc.begin_transfer_to_server("reminder_task", reminder_id)
    ).status == "pending"

    # The connection drops between begin and cancel.
    offline = _transfer_service(
        db, server_client=_connected_server_client(), active_server_id=None
    )
    outcome = await offline.cancel_transfer("reminder_task", reminder_id)

    assert outcome.status == "cancelled"
    assert db.get_reminder_task(reminder_id)["transfer_state"] is None
    assert db.get_pending_mutations(primitive="reminder_task") == [], (
        "the transfer mutation must not outlive its cancel"
    )


@pytest.mark.asyncio
async def test_cancel_release_offline_still_drops_the_release_mutation(db):
    """C2: the destructive half. Offline, the old cancel deleted the
    dormant copy but left the release mutation, which then DELETED the
    task server-side on the next reconnect -- the task ended up existing
    nowhere."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")
    copy_id = (
        await svc.begin_transfer_to_local("reminder_task", mirror_id)
    ).row_id

    offline = _transfer_service(
        db, server_client=_connected_server_client(), active_server_id=None
    )
    outcome = await offline.cancel_transfer("reminder_task", copy_id)

    assert outcome.status == "cancelled"
    assert db.get_reminder_task(copy_id) is None
    assert db.get_reminder_task(mirror_id) is not None
    assert db.get_pending_mutations(primitive="reminder_task") == [], (
        "a surviving release would delete the server task on reconnect"
    )


@pytest.mark.asyncio
async def test_cancel_does_not_discard_an_unrelated_queued_edit(db):
    """The mutation-keyed cancel must only claim the TRANSFER mutation:
    a plain queued edit on the same row is the user's work, not this
    machine's bookkeeping."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_pending", expected=(None,)
    )
    db.record_pending_mutation(
        reminder_id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Edited"}},
    )

    outcome = await svc.cancel_transfer("reminder_task", reminder_id)

    assert outcome.status == "cancelled"
    remaining = db.get_pending_mutations(primitive="reminder_task")
    assert [m["payload"]["action"] for m in remaining] == ["update"]


@pytest.mark.asyncio
async def test_begin_transfer_to_local_refuses_a_second_press(db):
    """I5: the mirror carries no `transfer_state` (only the copy does), so
    a second press built a second copy while the second mutation REPLACED
    the first -- stranding copy #1 dormant forever."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")

    first = await svc.begin_transfer_to_local("reminder_task", mirror_id)
    second = await svc.begin_transfer_to_local("reminder_task", mirror_id)

    assert first.status == "pending"
    assert second.status == "refused"
    assert "already in progress" in second.reason
    dormant = [
        row
        for row in db.list_reminder_tasks(owner_id="local")
        if row["transfer_state"] == "from_server_pending"
    ]
    assert len(dormant) == 1
    assert dormant[0]["id"] == first.row_id


@pytest.mark.asyncio
async def test_begin_transfer_to_local_definition_refuses_a_second_press(db, monkeypatch):
    """I5 on the definitions leg -- same keying, same stranded copy."""
    monkeypatch.setattr(
        scheduling_service_module, "compute_local_health", lambda app, row: ("ready", "")
    )
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    first = await svc.begin_transfer_to_local("automation_definition", mirror_id)
    second = await svc.begin_transfer_to_local("automation_definition", mirror_id)

    assert first.status == "pending"
    assert second.status == "refused"


@pytest.mark.parametrize(
    "state", ["to_server_pending", "to_server_sent", "from_server_pending"]
)
def test_transfer_lock_reason_locks_every_in_flight_state(state):
    """I7: spec §6.3's read-only rule, one source of truth."""
    assert SchedulingService.transfer_lock_reason({"transfer_state": state})


@pytest.mark.parametrize("state", [None, "to_server_failed"])
def test_transfer_lock_reason_leaves_editable_rows_alone(state):
    """A failed transfer re-armed locally -- editing before a retry is
    exactly what should be possible."""
    assert SchedulingService.transfer_lock_reason({"transfer_state": state}) is None


@pytest.mark.asyncio
async def test_update_reminder_refused_while_transferring(db):
    """I7 facade layer: the create payload was snapshotted at begin time,
    so an edit now ships pre-edit content and is then overwritten by the
    mirror pull."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db, title="Original")
    db.set_transfer_state(
        "reminder_task", reminder_id, "to_server_pending", expected=(None,)
    )

    assert await svc.update_reminder(reminder_id, {"title": "Edited"}) is None
    assert db.get_reminder_task(reminder_id)["title"] == "Original"


@pytest.mark.asyncio
async def test_delete_reminder_refused_while_transferring(db):
    """I7: deleting a dormant release copy discards the only row the
    release is about to arm."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")
    copy_id = (
        await svc.begin_transfer_to_local("reminder_task", mirror_id)
    ).row_id

    assert await svc.delete_reminder(copy_id) is False
    assert db.get_reminder_task(copy_id) is not None


@pytest.mark.asyncio
async def test_save_definition_refused_while_transferring(db):
    """I7 on the definitions leg."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db, name="Original")
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_pending", expected=(None,)
    )

    outcome = await svc.save_definition(
        _definition_payload(name="Edited"),
        owner_id="local",
        definition_id=definition_id,
    )

    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "transfer_in_progress"
    assert db.get_automation_definition(definition_id)["name"] == "Original"


@pytest.mark.asyncio
async def test_set_definition_lifecycle_local_writes_without_a_mutation(db):
    """M9: the missing producer. A local row has nothing to sync."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db)

    outcome = await svc.set_definition_lifecycle(definition_id, "pause")

    assert outcome.status == "saved"
    assert db.get_automation_definition(definition_id)["lifecycle"] == "paused"
    assert db.get_pending_mutations(primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_set_definition_lifecycle_server_row_records_the_replay_mutation(db):
    """M9: a server row writes optimistically AND queues the mutation
    `SyncEngine._push_definition_lifecycle` replays -- the leg that had no
    producer at all before this."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(
        db, owner_id="server:1", server_id="srv-def-1", lifecycle="paused"
    )

    outcome = await svc.set_definition_lifecycle(definition_id, "resume")

    assert outcome.status == "saved"
    assert db.get_automation_definition(definition_id)["lifecycle"] == "configured"
    mutations = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert [m["payload"]["action"] for m in mutations] == ["resume"]
    assert mutations[0]["payload"]["server_definition_id"] == "srv-def-1"


@pytest.mark.asyncio
async def test_set_definition_lifecycle_rejects_unknown_action(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db)
    outcome = await svc.set_definition_lifecycle(definition_id, "obliterate")
    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "unknown_action"


@pytest.mark.asyncio
async def test_set_definition_lifecycle_refused_while_transferring(db):
    svc = _transfer_service(db, server_client=_connected_server_client())
    definition_id = _make_definition(db)
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_sent", expected=(None,)
    )
    outcome = await svc.set_definition_lifecycle(definition_id, "archive")
    assert outcome.status == "error"
    assert outcome.errors[0]["code"] == "transfer_in_progress"


# ----------------------------------------------------------------------
# resolve_definition (schedules-handoff PR-6, task 2)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_definition_local_row_marks_solved(db):
    svc = SchedulingService(db=db, runtime_source="local")
    definition_id = _make_definition(db)
    result_id = db.create_automation_result(
        "local", definition_id, "run-1", "finding", "Found it", "Summary", "dk-1"
    )

    outcome = await svc.resolve_definition(definition_id, solved=True, result_id=result_id)

    assert outcome.status == "saved"
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "solved"
    assert row["resolved_by"] == "local"
    assert row["resolved_result_id"] == result_id
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_resolve_definition_local_row_reopen_clears_fields(db):
    svc = SchedulingService(db=db, runtime_source="local")
    definition_id = _make_definition(db, resolution_state="solved")

    outcome = await svc.resolve_definition(definition_id, solved=False)

    assert outcome.status == "saved"
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "open"
    assert row["resolved_by"] is None
    assert row["resolved_at"] is None
    assert row["resolved_result_id"] is None


@pytest.mark.asyncio
async def test_resolve_definition_unknown_id_returns_error(db):
    svc = SchedulingService(db=db, runtime_source="local")

    outcome = await svc.resolve_definition("missing-id", solved=True)

    assert outcome.status == "error"
    assert "missing-id" in outcome.reason


@pytest.mark.asyncio
async def test_resolve_definition_server_row_online_mirrors_server_echo(db):
    server_client = AsyncMock()
    server_client.mark_automation_definition_solved.return_value = {
        "id": "srv-def-1",
        "owner_id": "server:1",
        "family": "recurring_question",
        "name": "Daily Q",
        "lifecycle": "configured",
        "resolution_state": "solved",
        "resolved_at": "2026-09-02T00:00:00+00:00",
        "resolved_by": "alice",
        "resolved_result_id": "srv-res-1",
    }
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    outcome = await svc.resolve_definition(definition_id, solved=True, result_id=None)

    assert outcome.status == "saved"
    server_client.mark_automation_definition_solved.assert_awaited_once_with(
        "srv-def-1", result_id=None
    )
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "solved"
    # Came from the server echo via the mirror upsert, not a local write
    # (a local write would have stamped resolved_by="local").
    assert row["resolved_by"] == "alice"
    assert row["resolved_result_id"] == "srv-res-1"


@pytest.mark.asyncio
async def test_resolve_definition_server_row_reopen_online(db):
    server_client = AsyncMock()
    server_client.reopen_automation_definition.return_value = {
        "id": "srv-def-1",
        "owner_id": "server:1",
        "family": "recurring_question",
        "name": "Daily Q",
        "lifecycle": "paused",
        "resolution_state": "open",
    }
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(
        db, owner_id="server:1", server_id="srv-def-1", resolution_state="solved"
    )

    outcome = await svc.resolve_definition(definition_id, solved=False)

    assert outcome.status == "saved"
    server_client.reopen_automation_definition.assert_awaited_once_with("srv-def-1")
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "open"


@pytest.mark.asyncio
async def test_resolve_definition_server_row_offline_returns_error_without_queuing(db):
    """Plan ruling 2: unlike `set_definition_lifecycle`'s optimistic-write-
    plus-queue pattern, there is NO offline queue for this action in v1 --
    an unreachable seam must be an honest error, not a pending mutation."""
    # No server_client given -> the default `SchedulingServerClient()` has
    # no `notifications_service`, so any wrapper call raises
    # ServerUnavailableError, exactly like a real disconnected server.
    svc = SchedulingService(db=db, runtime_source="server:1")
    definition_id = _make_definition(
        db, owner_id="server:1", server_id="srv-def-1", resolution_state="open"
    )

    outcome = await svc.resolve_definition(definition_id, solved=True, result_id=None)

    assert outcome.status == "error"
    assert "server connection" in outcome.reason
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "open"  # untouched
    assert db.get_pending_mutations(primitive="automation_definition") == []


@pytest.mark.asyncio
async def test_resolve_definition_server_row_missing_server_id_returns_error(db):
    svc = SchedulingService(
        db=db, server_client=AsyncMock(), runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1")  # no server_id set

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "error"


@pytest.mark.asyncio
async def test_resolve_definition_translates_local_result_id_to_server_result_id(db):
    """The facade receives Task 3's LOCAL result id (results are consumed
    from the local mirror table) but the server has never heard of a
    local UUID -- it must be translated to the mirrored result's
    server_id before the network call."""
    server_client = AsyncMock()
    server_client.mark_automation_definition_solved.return_value = {
        "id": "srv-def-1",
        "owner_id": "server:1",
        "family": "recurring_question",
        "name": "Daily Q",
        "resolution_state": "solved",
    }
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")
    result_id = db.create_automation_result(
        "server:1",
        definition_id,
        "run-1",
        "finding",
        "Found it",
        "Summary",
        "dk-1",
        server_id="srv-res-1",
    )

    await svc.resolve_definition(definition_id, solved=True, result_id=result_id)

    server_client.mark_automation_definition_solved.assert_awaited_once_with(
        "srv-def-1", result_id="srv-res-1"
    )


# ----------------------------------------------------------------------
# resolve_definition fix round 1 (task-2-review.md): transfer lock, an
# unsynced result_id, and policy-vs-connectivity error wording.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_definition_refused_while_transferring(db):
    """I7 on the resolution leg: a same-window "mark solved" mid-transfer
    would be shipped by a create snapshot taken before this row's
    resolution fields existed, then silently clobbered back to "open" by
    the first mirror pull."""
    svc = SchedulingService(db=db, runtime_source="local")
    definition_id = _make_definition(db)
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_pending", expected=(None,)
    )

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "error"
    assert outcome.reason == scheduling_service_module._TRANSFER_READ_ONLY_REASON
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "open"


@pytest.mark.asyncio
async def test_resolve_definition_resolves_normally_with_null_transfer_state(db):
    svc = SchedulingService(db=db, runtime_source="local")
    definition_id = _make_definition(db)
    assert db.get_automation_definition(definition_id)["transfer_state"] is None

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "saved"


@pytest.mark.asyncio
async def test_resolve_definition_resolves_normally_with_failed_transfer_state(db):
    """`to_server_failed` re-armed locally -- not in `IN_FLIGHT_TRANSFER_
    STATES`, so it must stay editable, same as `transfer_lock_reason`'s
    own documented exception."""
    svc = SchedulingService(db=db, runtime_source="local")
    definition_id = _make_definition(db)
    db.set_transfer_state(
        "automation_definition", definition_id, "to_server_failed", expected=(None,)
    )

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "saved"


@pytest.mark.asyncio
async def test_resolve_definition_local_result_without_server_id_fails_closed(db):
    """Low finding 1: a result that exists locally but hasn't been synced
    up yet must not forward its LOCAL uuid to the server as if it were a
    server result id."""
    server_client = AsyncMock()
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")
    result_id = db.create_automation_result(
        "server:1", definition_id, "run-1", "finding", "Found it", "Summary", "dk-1"
    )  # no server_id: not yet synced up

    outcome = await svc.resolve_definition(definition_id, solved=True, result_id=result_id)

    assert outcome.status == "error"
    assert "not been synced to the server" in outcome.reason
    server_client.mark_automation_definition_solved.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_definition_unknown_result_id_fails_closed(db):
    """Same fail-closed path for a `result_id` that isn't a local row at
    all (never `None or {}`-falls-through to forwarding the raw id)."""
    server_client = AsyncMock()
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    outcome = await svc.resolve_definition(
        definition_id, solved=True, result_id="missing-result"
    )

    assert outcome.status == "error"
    assert "not been synced to the server" in outcome.reason
    server_client.mark_automation_definition_solved.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_definition_policy_denial_gets_distinct_reason(db):
    """Low finding 2: `_seam_failure_warning`'s wording split -- a
    deterministic policy refusal reads differently from "no connection"."""
    server_client = AsyncMock()
    server_client.mark_automation_definition_solved.side_effect = ServerClientPolicyError(
        "scheduler.automations.configure.server requires server mode."
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "error"
    assert "will not resolve by retrying" in outcome.reason
    assert "requires a server connection" not in outcome.reason


@pytest.mark.asyncio
async def test_resolve_definition_archived_409_reports_the_server_reason(db):
    """Live verification task 6 round 2, D9.

    Releasing a definition to this device archives the server's copy, so
    mark-solving a result that came from it returns the server's
    non-retryable 409:

        {"detail": {"code": "scheduled_task_definition_archived",
                    "message": "Scheduled task definition is archived.",
                    "retryable": false}}

    `_call_with_retry` maps every 4xx except 404 to
    `ServerClientValidationError` and never retries it, but the old catch
    only special-cased the `ServerClientPolicyError` SUBCLASS, so this
    fell through to the connectivity branch and told a plainly-connected
    user to check their network. The reason must carry the server's own
    explanation instead.
    """
    server_client = AsyncMock()
    server_client.mark_automation_definition_solved.side_effect = (
        ServerClientValidationError(
            "API Error 409: Scheduled task definition is archived."
        )
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "error"
    assert "archived" in outcome.reason
    assert "will not resolve by retrying" in outcome.reason
    assert "requires a server connection" not in outcome.reason
    # A refusal is never optimistically written locally.
    assert db.get_automation_definition(definition_id)["resolution_state"] == "open"


@pytest.mark.asyncio
async def test_resolve_definition_connectivity_failure_gets_generic_reason(db):
    server_client = AsyncMock()
    server_client.mark_automation_definition_solved.side_effect = ServerUnavailableError(
        "offline"
    )
    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    definition_id = _make_definition(db, owner_id="server:1", server_id="srv-def-1")

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "error"
    assert "requires a server connection" in outcome.reason
    assert "will not resolve by retrying" not in outcome.reason


# ----------------------------------------------------------------------
# Qodo review, fix wave 2: per-owner recovery + atomic begin legs
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_inflight_transfers_skips_rows_sent_to_another_server(db):
    """HIGH: recovery must reconcile each stuck row against the server it
    was actually SENT to, not whichever one is connected now.

    Server A's row is absent from server B's listing purely because it
    was never sent to B. Matching it there marks a possibly-landed
    transfer as absent, re-arms it, and duplicates the task on A at the
    next reconnect."""
    server_client = _connected_server_client()
    server_client.list_reminders.return_value = {"items": []}
    # Active connection is server:2; row A was sent under server:1.
    svc = _transfer_service(db, server_client=server_client, active_server_id="2")

    row_a = _make_reminder(db, title="Sent to A", transfer_state="to_server_sent")
    db.record_pending_mutation(
        row_a,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {}},
    )
    row_b = _make_reminder(db, title="Sent to B", transfer_state="to_server_sent")
    db.record_pending_mutation(
        row_b,
        "reminder_task",
        "server:2",
        {"action": "transfer_to_server", "task_payload": {}},
    )

    await svc.recover_inflight_transfers()

    # A is deferred verbatim: state and mutation both intact.
    assert db.get_reminder_task(row_a)["transfer_state"] == "to_server_sent"
    assert len(db.get_pending_mutations("server:1", primitive="reminder_task")) == 1
    # B is the only row reconciled against the connected server.
    assert db.get_reminder_task(row_b)["transfer_state"] == "to_server_pending"


@pytest.mark.asyncio
async def test_recover_inflight_transfers_skips_the_server_listing_when_all_deferred(
    db,
):
    """No row belongs to the connected server -- nothing is listed and
    nothing is touched."""
    server_client = _connected_server_client()
    svc = _transfer_service(db, server_client=server_client, active_server_id="2")
    row_id = _make_reminder(db, transfer_state="to_server_sent")
    db.record_pending_mutation(
        row_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {}},
    )

    await svc.recover_inflight_transfers()

    server_client.list_reminders.assert_not_awaited()
    assert db.get_reminder_task(row_id)["transfer_state"] == "to_server_sent"


@pytest.mark.asyncio
async def test_begin_transfer_to_local_interleaved_double_begin_makes_one_copy(db):
    """MEDIUM: the second `begin` is refused by the check inside the
    copy's own transaction, not only by the caller's pre-check.

    Neutering the pre-check simulates a second press arriving in the
    window the pre-check cannot cover -- the exact window that made two
    copies and one (replaced) mutation, stranding copy #1 forever."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")
    db.get_pending_mutation_for_local_id = lambda *args, **kwargs: None

    first = await svc.begin_transfer_to_local("reminder_task", mirror_id)
    second = await svc.begin_transfer_to_local("reminder_task", mirror_id)

    assert first.status == "pending"
    assert second.status == "refused"
    copies = [
        row
        for row in db.list_reminder_tasks()
        if row["transfer_state"] == "from_server_pending"
    ]
    assert len(copies) == 1
    mutations = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(mutations) == 1
    assert mutations[0]["payload"]["local_copy_id"] == copies[0]["id"]


@pytest.mark.asyncio
async def test_begin_transfer_to_local_copy_and_mutation_are_one_transaction(db):
    """A failure recording the mutation rolls the copy back with it --
    no dormant copy that nothing names."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    mirror_id = _make_reminder(db, owner_id="server:1", server_id="srv-1")

    def _boom(*args, **kwargs):
        raise RuntimeError("crash between the copy and its mutation")

    db._insert_pending_mutation_conn = _boom

    with pytest.raises(RuntimeError):
        await svc.begin_transfer_to_local("reminder_task", mirror_id)

    assert [
        row
        for row in db.list_reminder_tasks()
        if row["transfer_state"] == "from_server_pending"
    ] == []
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_begin_transfer_to_server_state_and_mutation_are_one_transaction(db):
    """MEDIUM: a crash between arming `to_server_pending` and queueing the
    mutation left a read-only row that nothing would ever replay."""
    svc = _transfer_service(db, server_client=_connected_server_client())
    reminder_id = _make_reminder(db)

    def _boom(*args, **kwargs):
        raise RuntimeError("crash between the CAS and its mutation")

    db._insert_pending_mutation_conn = _boom

    with pytest.raises(RuntimeError):
        await svc.begin_transfer_to_server("reminder_task", reminder_id)

    assert db.get_reminder_task(reminder_id)["transfer_state"] is None
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


def test_set_transfer_state_refused_cas_records_no_mutation(db):
    """The mutation follows the CAS: a refused transition writes neither."""
    reminder_id = _make_reminder(db, transfer_state="to_server_sent")

    armed = db.set_transfer_state(
        "reminder_task",
        reminder_id,
        "to_server_pending",
        expected=(None,),
        pending_mutation={
            "primitive": "reminder_task",
            "owner_id": "server:1",
            "payload": {"action": "transfer_to_server"},
        },
    )

    assert armed is False
    assert db.get_reminder_task(reminder_id)["transfer_state"] == "to_server_sent"
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_resolve_definition_server_row_gone_reports_definitive_reason(db):
    """A 404 from the server (row deleted server-side) is a DEFINITIVE
    refusal, not a connectivity problem -- final-review finding 3: the
    NotFound class isn't a ValidationError subclass, so without its own
    branch the connectivity wording would blame the network."""
    from tldw_chatbook.Scheduling.services.server_client import (
        ServerClientNotFoundError,
    )

    client = AsyncMock()
    client.mark_automation_definition_solved.side_effect = ServerClientNotFoundError(
        "definition not found"
    )
    svc = SchedulingService(db=db, server_client=client, runtime_source="server:1")
    definition_id = _make_definition(
        db, owner_id="server:1", server_id="srv-def-404", resolution_state="open"
    )

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "error"
    assert "no longer has this automation" in outcome.reason
    assert "server connection" not in outcome.reason
    assert db.get_automation_definition(definition_id)["resolution_state"] == "open"
