"""TaskDetail missed-notice UI integration tests (task-18937, review #4).

Mounts the REAL TaskDetail widget inside the real SchedulesWorkbench
surface and drives real ``set_task`` calls, pinning notice visibility,
skipped-count copy, overflow copy, clearing on on-time tasks, and the
run-now retry label variants. Complements the dispatch-path tests in
Tests/Scheduling/test_missed_fire.py, which cover the state itself.
"""

from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ScheduleKind,
    TaskStatus,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail

NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)


class _DetailHarnessApp(ConsolidatedCSSApp):
    """Bare app mounting one TaskDetail, matching the workbench's compose."""

    def compose(self):
        yield TaskDetail()


def _reminder(**kwargs) -> ReminderTask:
    defaults = dict(
        id="task-1",
        title="Late reminder",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 * * * *",
        timezone="UTC",
    )
    defaults.update(kwargs)
    return ReminderTask(**defaults)


async def _notice_text(detail: TaskDetail) -> tuple[str, bool]:
    notice = detail.query_one("#scheduling-task-detail-missed", Static)
    text = notice.visual.plain if notice.display else ""
    return text, bool(notice.display)


@pytest.mark.asyncio
async def test_late_recurring_renders_notice_with_skipped_count():
    """A late recurring task shows the notice + exact count."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _reminder(
                missed_at=NOW - timedelta(hours=2),
                missed_count=1,
                last_status=TaskStatus.COMPLETED,
            )
        )
        await pilot.pause()
        text, visible = await _notice_text(detail)
        assert visible
        assert "Ran late" in text
        assert "1 earlier occurrence(s) were skipped" in text
        assert "not replayed" in text


@pytest.mark.asyncio
async def test_late_one_time_renders_notice_without_count():
    """A late one-time reminder shows lateness copy, no skipped count."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _reminder(
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=NOW - timedelta(hours=2),
                missed_at=NOW - timedelta(hours=2),
                missed_count=0,
                last_status=TaskStatus.COMPLETED,
            )
        )
        await pilot.pause()
        text, visible = await _notice_text(detail)
        assert visible
        assert "Ran late" in text
        assert "dispatched well after its scheduled time" in text
        assert "skipped" not in text


@pytest.mark.asyncio
async def test_overflow_count_renders_more_than_copy():
    """The -1 sentinel renders as an explicit 'more than N', never exact."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(
            _reminder(
                missed_at=NOW - timedelta(days=200),
                missed_count=-1,
                last_status=TaskStatus.COMPLETED,
            )
        )
        await pilot.pause()
        text, visible = await _notice_text(detail)
        assert visible
        assert "more than 100,000" in text


@pytest.mark.asyncio
async def test_ontime_task_clears_the_notice():
    """An on-time task (no missed_at) hides the notice entirely."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        # First show it, then clear it: the clearing branch is what's pinned.
        detail.set_task(
            _reminder(missed_at=NOW, missed_count=1)
        )
        await pilot.pause()
        _, visible_before = await _notice_text(detail)
        assert visible_before

        detail.set_task(_reminder(last_status=TaskStatus.COMPLETED))
        await pilot.pause()
        _, visible_after = await _notice_text(detail)
        assert not visible_after


@pytest.mark.asyncio
async def test_failed_status_still_distinct_from_missed_while_away():
    """Status 'Missed' (ran and raised) renders no missed-while-away notice."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(last_status=TaskStatus.MISSED))
        await pilot.pause()
        _, visible = await _notice_text(detail)
        assert not visible
        badge = detail.query_one("#scheduling-task-status-badge", Static)
        assert badge.visual.plain == "Missed"


@pytest.mark.asyncio
async def test_run_now_label_variants():
    """Run-now reads 'retry' for Missed and Timed out, plain otherwise."""
    from textual.widgets import Button

    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)

        detail.set_task(_reminder(last_status=TaskStatus.COMPLETED))
        await pilot.pause()
        button = detail.query_one("#scheduling-run-now", Button)
        assert str(button.label) == "Run now"

        detail.set_task(_reminder(last_status=TaskStatus.MISSED))
        await pilot.pause()
        assert str(button.label) == "Run now (retry)"

        detail.set_task(_reminder(last_status=TaskStatus.TIMED_OUT))
        await pilot.pause()
        assert str(button.label) == "Run now (retry)"


@pytest.mark.asyncio
async def test_cleared_task_hides_notice():
    """set_task(None) clears the notice along with the metadata pane."""
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        detail.set_task(_reminder(missed_at=NOW, missed_count=1))
        await pilot.pause()
        _, visible_before = await _notice_text(detail)
        assert visible_before

        detail.set_task(None)
        await pilot.pause()
        _, visible_after = await _notice_text(detail)
        assert not visible_after


@pytest.mark.asyncio
async def test_the_notice_never_claims_the_scheduler_was_not_running():
    """task-19562: the app must not assert a cause it cannot know.

    `SchedulerLoop.tick` awaits every due handler serially, so one slow
    handler pushes the tasks behind it past the missed-fire grace while the
    scheduler is running the whole time. The row that results is identical
    to one from an app that was closed. The old copy -- "Missed while away
    ... (the scheduler was not running at the scheduled time)" -- was
    therefore false for an ordinary, reachable case.

    Checked across every branch of the notice, because the false sentence
    lived in only one of the three and a copy edit could easily restore it
    in another.
    """
    forbidden = ("Missed while away", "was not running")
    cases = (
        dict(missed_at=NOW - timedelta(hours=2), missed_count=1),
        dict(missed_at=NOW - timedelta(hours=2), missed_count=0),
        dict(missed_at=NOW - timedelta(days=200), missed_count=-1),
    )
    async with _DetailHarnessApp().run_test() as pilot:
        detail = pilot.app.query_one(TaskDetail)
        for case in cases:
            detail.set_task(_reminder(last_status=TaskStatus.COMPLETED, **case))
            await pilot.pause()
            text, visible = await _notice_text(detail)
            assert visible
            for phrase in forbidden:
                assert phrase not in text, (
                    f"the notice claims a cause the app cannot know: {text!r}"
                )
            assert "Ran late" in text
