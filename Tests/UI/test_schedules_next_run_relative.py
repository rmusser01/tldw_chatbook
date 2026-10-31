"""task-23111: Next Run pairs the absolute time with a relative form.

"2026-08-28 09:00 UTC" alone forces the reader to do timezone
arithmetic; the detail pane now renders "2026-08-28 09:00 UTC (in 14h)"
and the queue a shorter "2026-08-28 09:00 (in 14h)". All tests inject
``now`` for determinism.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from textual.widgets import DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind, TaskStatus
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import _format_next_run


_NOW = datetime(2026, 8, 27, 19, 0, tzinfo=timezone.utc)


def _task(next_run_at: datetime, *, enabled: bool = True) -> ReminderTask:
    return ReminderTask(
        id="task-r",
        title="Relative",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=next_run_at,
        next_run_at=next_run_at,
        enabled=enabled,
    )


def test_detail_form_pairs_absolute_with_relative_hours():
    run_at = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)  # 14h ahead
    assert _format_next_run(_task(run_at), now=_NOW) == (
        "2026-08-28 09:00 UTC (in 14h)"
    )


def test_queue_form_is_shorter_but_same_relative():
    run_at = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)
    assert _format_next_run(_task(run_at), now=_NOW, compact=True) == (
        "2026-08-28 09:00 (in 14h)"
    )


def test_minutes_bucket():
    run_at = _NOW + timedelta(minutes=25)
    assert "(in 25m)" in _format_next_run(_task(run_at), now=_NOW)


def test_days_bucket():
    run_at = _NOW + timedelta(days=3, hours=2)
    assert "(in 3d)" in _format_next_run(_task(run_at), now=_NOW)


def test_under_a_minute_is_due_now():
    run_at = _NOW + timedelta(seconds=30)
    assert "(due now)" in _format_next_run(_task(run_at), now=_NOW)


def test_past_time_reads_overdue():
    run_at = _NOW - timedelta(hours=2)
    assert "(overdue 2h)" in _format_next_run(_task(run_at), now=_NOW)


def test_disabled_still_shows_no_time_and_no_relative():
    run_at = _NOW + timedelta(hours=5)
    rendered = _format_next_run(_task(run_at, enabled=False), now=_NOW)
    assert rendered == "— (disabled)"


def test_naive_next_run_treated_as_utc():
    run_at = datetime(2026, 8, 28, 9, 0)  # naive
    rendered = _format_next_run(_task(run_at), now=_NOW)
    assert "(in 14h)" in rendered


# --- consistency between the queue column and the detail pane --------------


from Tests.UI.schedules_test_helpers import MockSchedulingServiceMixin


class _Service(MockSchedulingServiceMixin):
    async def list_tasks(self, owner_id=None, include_projections=True):
        return [
            ReminderTask(
                id="task-1",
                title="Soon",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime.now(timezone.utc) + timedelta(hours=14),
                next_run_at=datetime.now(timezone.utc) + timedelta(hours=14),
            )
        ]


class _App(ConsolidatedCSSApp):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = _Service()


# ``bootstrap_profile``: see the same note in test_schedules_disabled_state.py
# (whole-PR review of Phase 9, round 1).
@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_queue_cell_and_detail_pane_agree_on_the_relative_form():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        table = workbench.query_one("#scheduling-task-table", DataTable)
        # redesign PR-2, Task 2: column 2 is now the subtitle (relative
        # next-run, then schedule summary since TASK-33641.4), replacing
        # the old standalone Next-Run column at index 3.
        cell = str(table.get_row_at(0)[2])
        detail = str(
            workbench.query_one("#scheduling-task-detail-next-run", Static).render()
        )
        assert "(in 13h)" in cell or "(in 14h)" in cell, cell
        # Same relative rendering in both surfaces.
        relative = cell[cell.index("(") : cell.index(")") + 1]
        assert relative in detail, (cell, detail)


# --- TASK-33641.4: a recurring reminder keeps its next run after a dispatch -


class _CompletedRecurringService(_Service):
    """A weekly reminder as its last successful dispatch left it."""

    async def list_tasks(self, owner_id=None, include_projections=True):
        now = datetime.now(timezone.utc)
        return [
            ReminderTask(
                id="task-1",
                title="Weekly digest",
                schedule_kind=ScheduleKind.RECURRING,
                cron="0 9 * * 1",
                timezone="UTC",
                next_run_at=now + timedelta(days=5, hours=1),
                last_run_at=now - timedelta(days=2),
                last_status=TaskStatus.COMPLETED,
            )
        ]


# ``bootstrap_profile``: see the same note in test_schedules_disabled_state.py.
@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_completed_recurring_reminder_shows_its_next_run_in_queue_and_detail():
    """Both mounted surfaces: the badge reads Completed (the last outcome)
    while the queue cell and the detail pane still name the next run."""
    app = _App()
    app.scheduling_service = _CompletedRecurringService()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        table = workbench.query_one("#scheduling-task-table", DataTable)
        cell = str(table.get_row_at(0)[2])
        detail = str(
            workbench.query_one("#scheduling-task-detail-next-run", Static).render()
        )
        badge = str(
            workbench.query_one("#scheduling-task-status-badge", Static).render()
        )
        assert "Completed" in badge, badge
        assert "(in 5d)" in cell, cell
        assert "(in 5d)" in detail, detail


# --- review F9: relative text refreshes instead of going stale ------------


class _FixedTaskService(_Service):
    """Task at a fixed absolute time so injected `now` moves the bucket."""

    RUN_AT = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [
            ReminderTask(
                id="task-1",
                title="Fixed",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=self.RUN_AT,
                next_run_at=self.RUN_AT,
            )
        ]


class _FixedApp(ConsolidatedCSSApp):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = _FixedTaskService()


@pytest.mark.asyncio
async def test_render_table_uses_one_injectable_now():
    """One shared `now` per render; re-rendering with a later reference
    moves the bucket -- the mechanism the 60s refresh timer drives."""
    app = _FixedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        table = workbench.query_one("#scheduling-task-table", DataTable)
        run_at = _FixedTaskService.RUN_AT

        workbench._render_table(now=run_at - timedelta(hours=14))
        assert "(in 14h)" in str(table.get_row_at(0)[2])

        workbench._render_table(now=run_at - timedelta(minutes=25))
        assert "(in 25m)" in str(table.get_row_at(0)[2])


@pytest.mark.asyncio
async def test_refresh_timer_exists_and_skips_when_not_current():
    app = _FixedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert workbench._next_run_refresh_timer is not None

        renders: list[int] = []
        real_render = workbench._render_table
        workbench._render_table = lambda **kw: renders.append(1)  # type: ignore[assignment]

        workbench._refresh_next_run_rendering()
        assert renders, "refresh must re-render while the screen is current"

        # Covered by another screen: the refresh must skip. (Textual's
        # is_current cannot express "covered" -- _background_screens
        # always includes the screen directly beneath the top -- so the
        # guard checks top-of-stack, and suspend pauses the timer.)
        renders.clear()
        from textual.screen import Screen

        class _Cover(Screen):
            pass

        await pilot.app.push_screen(_Cover())
        await pilot.pause()
        assert pilot.app.screen is not workbench
        workbench._refresh_next_run_rendering()
        assert not renders, "refresh must skip while another screen covers it"
        # And the suspend handler paused the cadence timer outright.
        assert workbench._next_run_refresh_timer is not None
        workbench._render_table = real_render  # type: ignore[assignment]


# --- redesign PR-2, Task 4: ticker pinned against the unified rows --------
#
# The ticker itself is unchanged from dev (survey: the ONLY set_interval
# under UI/Screens/**, adapted to `_visible_rows` in Task 2 -- see the
# guard comment on `_refresh_next_run_rendering`). These three tests pin
# the exact contract the task-4 brief calls out: a tick never reloads data,
# suspend pauses the cadence, and resume both resumes it and repaints
# immediately so nothing sits stale.


@pytest.mark.asyncio
async def test_tick_rerenders_without_reloading_data():
    """A tick must repaint the cached rows' relative text, never re-fetch
    from the service (brief: 'no reload, no DB')."""
    app = _FixedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        reload_calls: list[bool] = []
        workbench._request_tasks_refresh = (  # type: ignore[assignment]
            lambda **kw: reload_calls.append(True)
        )
        render_calls: list[bool] = []
        workbench._render_table = lambda **kw: render_calls.append(True)  # type: ignore[assignment]

        workbench._refresh_next_run_rendering()

        assert render_calls, "a tick must re-render the relative-time text"
        assert not reload_calls, "a tick must never re-fetch data (no reload, no DB)"


@pytest.mark.asyncio
async def test_on_screen_suspend_pauses_the_ticker():
    app = _FixedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        mock_timer = Mock()
        workbench._next_run_refresh_timer = mock_timer

        workbench.on_screen_suspend()

        mock_timer.pause.assert_called_once()


@pytest.mark.asyncio
async def test_on_screen_resume_resumes_and_ticks_immediately():
    app = _FixedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        mock_timer = Mock()
        workbench._next_run_refresh_timer = mock_timer
        render_calls: list[bool] = []
        workbench._render_table = lambda **kw: render_calls.append(True)  # type: ignore[assignment]

        workbench.on_screen_resume()

        mock_timer.resume.assert_called_once()
        assert render_calls, "resume must repaint immediately -- nothing stays stale"
