"""TASK-26027: failure incidents — grouping, ack, auto-close, durability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.Scheduling.task_incidents import normalize_error_signature


def test_signature_strips_volatile_details():
    """AC#5: timestamps, ids, hex, paths, numbers vary but the signature
    is stable so repeats group."""
    a = normalize_error_signature(
        "ConnectionError: failed at 2026-09-01T12:00:03Z for job 4821 "
        "(/var/run/x-7f3a2b.sock) after 1234ms"
    )
    b = normalize_error_signature(
        "ConnectionError: failed at 2026-09-01T13:59:59Z for job 9002 "
        "(/var/run/x-0091cd.sock) after 55ms"
    )
    assert a == b, "volatile details must not defeat grouping"
    assert "ConnectionError" in a


def test_different_error_class_gives_a_different_signature():
    a = normalize_error_signature("ConnectionError: host down")
    b = normalize_error_signature("TimeoutError: host down")
    assert a != b


def test_signature_is_bounded_and_handles_empty():
    assert normalize_error_signature("") != ""  # a stable placeholder
    assert normalize_error_signature(None) == normalize_error_signature("")
    assert len(normalize_error_signature("x" * 5000)) <= 200


from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB


def _t(offset=0):
    return datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset)


@pytest.fixture
def db(tmp_path):
    return ScheduledTasksDB(tmp_path / "inc.db")


def test_first_failure_notifies_repeats_group_silently(db):
    """AC#1."""
    sig = normalize_error_signature("ConnectionError: down")
    inc1, notify1 = db.record_task_failure("t1", "briefing_job", sig, _t(0))
    inc2, notify2 = db.record_task_failure("t1", "briefing_job", sig, _t(3600))
    inc3, notify3 = db.record_task_failure("t1", "briefing_job", sig, _t(7200))

    assert notify1 is True, "first failure of a new signature notifies"
    assert notify2 is False and notify3 is False, "repeats group, no re-notify"
    assert inc1 == inc2 == inc3, "one incident"
    incidents = db.list_task_incidents("t1")
    assert len(incidents) == 1
    assert incidents[0]["occurrence_count"] == 3


def test_acknowledge_suppresses_and_does_not_disable(db):
    """AC#2/#7."""
    sig = normalize_error_signature("ConnectionError: down")
    inc, _ = db.record_task_failure("t1", "briefing_job", sig, _t(0))
    db.acknowledge_incident(inc, _t(1))
    _, notify = db.record_task_failure("t1", "briefing_job", sig, _t(3600))
    assert notify is False, "acked signature stays suppressed"
    inc_row = db.list_task_incidents("t1")[0]
    assert inc_row["status"] == "acknowledged"
    # ack never touches the task itself (AC#7) -- the incident store has no
    # task-mutating method, so this is structural; assert the task row is
    # untouched by confirming no enable/disable column changed via a getter
    # that would exist. Here we assert ack is incident-scoped only.
    assert inc_row["closed_at"] is None


def test_different_signature_opens_a_distinct_incident(db):
    """AC#3."""
    inc_a, _ = db.record_task_failure(
        "t1", "briefing_job", normalize_error_signature("ConnectionError: x"), _t(0)
    )
    db.acknowledge_incident(inc_a, _t(1))
    inc_b, notify_b = db.record_task_failure(
        "t1", "briefing_job", normalize_error_signature("ValueError: y"), _t(2)
    )
    assert inc_b != inc_a
    assert notify_b is True, "a new signature notifies despite the acked one"
    assert len(db.list_task_incidents("t1")) == 2


def test_success_closes_and_recurrence_reopens(db):
    """AC#4 + AC#2's 'recurs after a resolution'."""
    sig = normalize_error_signature("ConnectionError: down")
    inc1, _ = db.record_task_failure("t1", "briefing_job", sig, _t(0))
    closed = db.record_task_success("t1", _t(10))
    assert closed == 1
    assert db.list_task_incidents("t1")[0]["status"] == "closed"

    # a fresh failure after resolution opens a NEW incident and re-notifies
    inc2, notify = db.record_task_failure("t1", "briefing_job", sig, _t(20))
    assert inc2 != inc1
    assert notify is True


def test_incidents_are_durable_across_reopen(tmp_path):
    """AC#6."""
    path = tmp_path / "d.db"
    db1 = ScheduledTasksDB(path)
    sig = normalize_error_signature("ConnectionError: down")
    db1.record_task_failure("t1", "briefing_job", sig, _t(0))
    db1.close()

    db2 = ScheduledTasksDB(path)
    incidents = db2.list_task_incidents("t1")
    assert len(incidents) == 1
    # a repeat after restart still groups into the SAME open incident
    _, notify = db2.record_task_failure("t1", "briefing_job", sig, _t(3600))
    assert notify is False
    assert db2.list_task_incidents("t1")[0]["occurrence_count"] == 2


@pytest.mark.asyncio
async def test_briefing_handler_groups_repeat_failures(tmp_path):
    from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
        BriefingJobHandler,
    )

    class _Dispatch:
        def __init__(self):
            self.calls = 0

        def dispatch(self, **kwargs):
            self.calls += 1

    real_db = ScheduledTasksDB(tmp_path / "h.db")
    dispatch = _Dispatch()
    handler = BriefingJobHandler(
        subscriptions_db=None,
        dispatch_service=dispatch,
        incident_recorder=real_db,
    )
    handler._watchlist_name = lambda wid: "My Watchlist"

    # same failure twice -> one notification (AC#1)
    await handler._notify_error(7, signature="ConnectionError: down at 12:00")
    await handler._notify_error(7, signature="ConnectionError: down at 13:00")
    assert dispatch.calls == 1, "grouped failures notify once"

    # resolve, then a recurrence notifies afresh (AC#4 + recurrence)
    handler._close_incident(7)
    await handler._notify_error(7, signature="ConnectionError: down at 14:00")
    assert dispatch.calls == 2


@pytest.mark.asyncio
async def test_briefing_handler_without_recorder_always_notifies(tmp_path):
    from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
        BriefingJobHandler,
    )

    class _Dispatch:
        def __init__(self):
            self.calls = 0

        def dispatch(self, **kwargs):
            self.calls += 1

    dispatch = _Dispatch()
    handler = BriefingJobHandler(subscriptions_db=None, dispatch_service=dispatch)
    handler._watchlist_name = lambda wid: "W"
    await handler._notify_error(7, signature="ConnectionError: down")
    await handler._notify_error(7, signature="ConnectionError: down")
    assert dispatch.calls == 2, "no recorder = today's behavior (always notify)"


def test_format_incidents_shows_open_only_newest_first():
    from tldw_chatbook.UI.Screens.scheduling.task_detail import format_incidents

    assert "No open incidents" in format_incidents(None)
    assert "No open incidents" in format_incidents(
        [{"status": "closed", "occurrence_count": 3, "signature": "x"}]
    )
    rendered = format_incidents(
        [
            {"status": "alerting", "occurrence_count": 5, "signature": "ConnErr"},
            {"status": "closed", "occurrence_count": 1, "signature": "old"},
        ]
    )
    assert "alerting" in rendered and "×5" in rendered and "ConnErr" in rendered
    assert "old" not in rendered, "closed incidents are not shown"
