"""Tests for the briefing projection service (briefings phase 4, task 3).

`BriefingProjection` turns `SubscriptionsDB.list_briefing_schedules` rows
(briefings phase 4, task 2 -- already shipped and tested on its own in
`Tests/Subscriptions/test_briefing_cadence_db.py`) into `ScheduledTask`
objects for the scheduler's `PriorityQueue`, mirroring `WatchlistProjection`'s
own shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    parse_briefing_task_id as handler_parse_briefing_task_id,
)
from tldw_chatbook.Scheduling.services.briefing_projection import (
    BRIEFING_TASK_PREFIX,
    BriefingProjection,
    parse_briefing_task_id,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _make_watchlist(db, name="w"):
    return WatchlistBundleService(db).create(name=name)["id"]


def _force_created_at(db, briefing_id, timestamp):
    """Overwrite a `briefings` row's `created_at` directly (second resolution)."""
    db.conn.execute(
        "UPDATE briefings SET created_at = ? WHERE id = ?", (timestamp, briefing_id)
    )
    db.conn.commit()


# --- list_jobs: shape and next_run_at ---------------------------------------


def test_projection_emits_no_task_for_a_watchlist_with_no_cadence():
    """Locked Decision 4: opt-in only -- an un-cadenced watchlist emits nothing."""
    db = SubscriptionsDB(":memory:", "test")
    _make_watchlist(db, name="never-scheduled")

    projection = BriefingProjection(db)
    assert projection.list_jobs() == []


def test_projection_never_briefed_cadenced_watchlist_is_due_now():
    """A schedule with no history is due right now, not deferred a full cadence."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db, name="Fresh Schedule")
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    projection = BriefingProjection(db)
    tasks = projection.list_jobs(now=now)

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, ScheduledTask)
    assert task.id == "briefing:" + str(watchlist_id)
    assert task.type == "briefing_job"
    assert task.title == "Fresh Schedule"
    assert task.status is TaskStatus.WAITING
    assert task.owner_id == "local"
    assert task.next_run_at == now


def test_projection_next_run_at_is_last_completed_at_plus_cadence():
    """The cadenced-and-briefed-before path: `next_run_at` is watermark + cadence."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db, name="Recurring")
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=7200)

    complete_id = db.insert_briefing(watchlist_id, status="complete")
    _force_created_at(db, complete_id, "2026-01-01 00:00:00")

    now = datetime(2026, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
    projection = BriefingProjection(db)
    tasks = projection.list_jobs(now=now)

    assert len(tasks) == 1
    expected = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=7200
    )
    assert tasks[0].next_run_at == expected
    # And it must NOT simply equal "now" -- that would be the never-briefed
    # branch leaking into the has-history branch.
    assert tasks[0].next_run_at != now


def test_projection_empty_briefing_counts_as_history_for_next_run_at():
    """`empty` is in the same allowlist as `complete` on the DB side; the
    projection must not need its own separate awareness of that -- it just
    trusts whatever `last_completed_at` says."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db, name="Quiet")
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=1800)

    empty_id = db.insert_briefing(watchlist_id, status="empty")
    _force_created_at(db, empty_id, "2026-03-01 09:00:00")

    projection = BriefingProjection(db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    expected = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=1800
    )
    assert tasks[0].next_run_at == expected


def test_projection_ignores_failed_and_generating_rows_for_next_run_at():
    """A failed or still-`generating` briefing must not advance the schedule
    -- pinned through the projection the same way task 2 pins it at the DB
    layer, since this is the value the scheduler actually dispatches on."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db, name="Flaky")
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    complete_id = db.insert_briefing(watchlist_id, status="complete")
    _force_created_at(db, complete_id, "2020-01-01 00:00:00")
    failed_id = db.insert_briefing(watchlist_id, status="failed")
    _force_created_at(db, failed_id, "2099-01-01 00:00:00")
    generating_id = db.insert_briefing(watchlist_id, status="generating")
    _force_created_at(db, generating_id, "2099-06-01 00:00:00")

    projection = BriefingProjection(db)
    tasks = projection.list_jobs()

    expected = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=3600)
    assert tasks[0].next_run_at == expected


def test_projection_multiple_schedules_each_get_their_own_task():
    db = SubscriptionsDB(":memory:", "test")
    a = _make_watchlist(db, name="A")
    b = _make_watchlist(db, name="B")
    _make_watchlist(db, name="Never")  # no cadence -- must not appear
    db.set_watchlist_briefing_settings(a, briefing_cadence_seconds=3600)
    db.set_watchlist_briefing_settings(b, briefing_cadence_seconds=86400)

    projection = BriefingProjection(db)
    tasks = projection.list_jobs()

    ids = {task.id for task in tasks}
    assert ids == {f"briefing:{a}", f"briefing:{b}"}


# --- id round trip: the ONE parser -------------------------------------------


def test_projection_id_round_trips_through_the_one_parser():
    """The id `list_jobs` builds must parse back to the same watchlist id
    through `parse_briefing_task_id` -- the single shared parser
    `briefing_handler.py` imports rather than reimplementing (the 2b
    two-copies-drift lesson)."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db, name="Round Trip")
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=900)

    projection = BriefingProjection(db)
    [task] = projection.list_jobs()

    assert task.id.startswith(BRIEFING_TASK_PREFIX + ":")
    assert parse_briefing_task_id(task.id) == watchlist_id
    # And the handler's own import of the same function agrees -- proving
    # there is exactly one implementation, not two that happen to match.
    assert handler_parse_briefing_task_id is parse_briefing_task_id
    assert handler_parse_briefing_task_id(task.id) == watchlist_id


@pytest.mark.parametrize(
    "task_id",
    [None, 42, "", "briefing", "watchlist:7", "briefing:not-a-number", "briefing:"],
)
def test_parse_briefing_task_id_rejects_malformed_ids(task_id):
    assert parse_briefing_task_id(task_id) is None


def test_parse_briefing_task_id_accepts_the_shape_list_jobs_builds():
    assert parse_briefing_task_id("briefing:123") == 123
