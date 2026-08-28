"""task-23101: a disabled task must read as Disabled, not Waiting.

``_task_status`` used to return ``last_status``, which disabling never
touches, so a disabled task kept showing "Waiting" with a concrete
future Next Run in both the queue row and the detail badge.
"""

from datetime import datetime, timezone

import pytest
from textual.widgets import DataTable, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ScheduledTask,
    ScheduleKind,
    TaskStatus,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import (
    TaskDetail,
    _format_next_run,
    _task_status,
)


_RUN_AT = datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc)


def _reminder(*, enabled: bool, last_status=TaskStatus.WAITING) -> ReminderTask:
    return ReminderTask(
        id="task-d",
        title="Backup",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=_RUN_AT,
        next_run_at=_RUN_AT,
        enabled=enabled,
        last_status=last_status,
    )


# --- derivation unit tests -------------------------------------------------


def test_disabled_reminder_status_is_disabled():
    assert _task_status(_reminder(enabled=False)) is TaskStatus.DISABLED


def test_enabled_reminder_keeps_last_status():
    assert (
        _task_status(_reminder(enabled=True, last_status=TaskStatus.MISSED))
        is TaskStatus.MISSED
    )


def test_projection_status_unaffected():
    projection = ScheduledTask(
        id="watchlist:1",
        title="Watch",
        type="watchlist_job",
        status=TaskStatus.WAITING,
        next_run_at=_RUN_AT,
    )
    assert _task_status(projection) is TaskStatus.WAITING
    assert "2099" in _format_next_run(projection)


def test_disabled_reminder_next_run_shows_no_concrete_time():
    rendered = _format_next_run(_reminder(enabled=False))
    assert "2099" not in rendered
    assert "disabled" in rendered.lower()


def test_enabled_reminder_next_run_shows_time():
    assert "2099-07-20" in _format_next_run(_reminder(enabled=True))


# --- review F6: suppression covers DISABLED/PAUSED projections too --------


def _projection(status: TaskStatus) -> ScheduledTask:
    return ScheduledTask(
        id="watchlist:9",
        title="Watch",
        type="watchlist_job",
        status=status,
        next_run_at=_RUN_AT,
    )


def test_disabled_projection_next_run_is_suppressed():
    rendered = _format_next_run(_projection(TaskStatus.DISABLED))
    assert "2099" not in rendered
    assert rendered == "— (disabled)"


def test_paused_projection_next_run_is_suppressed():
    rendered = _format_next_run(_projection(TaskStatus.PAUSED))
    assert "2099" not in rendered
    assert rendered == "— (paused)"


def test_waiting_projection_next_run_still_concrete():
    assert "2099-07-20" in _format_next_run(_projection(TaskStatus.WAITING))


# --- review F5: behavior consumers use the underlying status --------------


def test_underlying_status_ignores_the_disabled_overlay():
    from tldw_chatbook.UI.Screens.scheduling.task_detail import (
        _underlying_status,
    )

    task = _reminder(enabled=False, last_status=TaskStatus.MISSED)
    assert _task_status(task) is TaskStatus.DISABLED
    assert _underlying_status(task) is TaskStatus.MISSED


# --- mounted behavior ------------------------------------------------------


from Tests.UI.schedules_test_helpers import MockSchedulingServiceMixin


class _DisabledTaskService(MockSchedulingServiceMixin):
    def __init__(self) -> None:
        self.enabled = False

    async def list_tasks(self):
        return [_reminder(enabled=self.enabled)]

    async def update_reminder(self, task_id, fields):
        self.enabled = fields.get("enabled", self.enabled)


class _WorkbenchApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _DisabledTaskService()


@pytest.mark.asyncio
async def test_disabled_row_and_badge_read_disabled_and_survive_refresh():
    app = _WorkbenchApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = SchedulesWorkbench(app_instance=pilot.app)
        await pilot.app.push_screen(workbench)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        table = workbench.query_one("#scheduling-task-table", DataTable)
        row = table.get_row_at(0)
        # Queue row: text says Disabled, and no concrete future time.
        assert "Disabled" in str(row[2])
        assert "2099" not in str(row[3])
        assert "disabled" in str(row[3]).lower()

        badge = workbench.query_one("#scheduling-task-status-badge", Static)
        assert "Disabled" in str(badge.render())

        next_run = workbench.query_one(
            "#scheduling-task-detail-next-run", Static
        )
        assert "2099" not in str(next_run.render())

        # The derived state survives a queue refresh.
        await workbench.load_tasks()
        await pilot.pause()
        row = table.get_row_at(0)
        assert "Disabled" in str(row[2])


# --- review F5: mounted consumers of the underlying status ----------------


class _DisabledMissedService(MockSchedulingServiceMixin):
    """One disabled task whose last dispatch failed, one with a conflict."""

    async def list_tasks(self):
        return [
            _reminder_with(
                "task-m", "Failed backup", TaskStatus.MISSED, enabled=False
            ),
            _reminder_with(
                "task-c", "Conflicted", TaskStatus.CONFLICT, enabled=False
            ),
        ]


def _reminder_with(
    task_id: str, title: str, last_status: TaskStatus, *, enabled: bool
) -> ReminderTask:
    return ReminderTask(
        id=task_id,
        title=title,
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=_RUN_AT,
        next_run_at=_RUN_AT,
        enabled=enabled,
        last_status=last_status,
    )


class _MissedApp(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _DisabledMissedService()


async def _mounted_missed_workbench(pilot):
    workbench = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(workbench)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench


@pytest.mark.asyncio
async def test_disabled_missed_task_keeps_the_retry_affordance():
    """Run now explicitly works on disabled tasks; a disabled task whose
    last dispatch failed must keep 'Run now (retry)' (review F5)."""
    from textual.widgets import Button

    app = _MissedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_missed_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=0)
        await pilot.pause()

        run_now = workbench.query_one("#scheduling-run-now", Button)
        assert str(run_now.label) == "Run now (retry)"
        # Display status stays Disabled.
        badge = workbench.query_one("#scheduling-task-status-badge", Static)
        assert "Disabled" in str(badge.render())


@pytest.mark.asyncio
async def test_missed_filter_matches_a_disabled_missed_task():
    app = _MissedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_missed_workbench(pilot)
        workbench._filter_text = "missed"
        workbench._render_table()
        await pilot.pause()

        titles = [task.title for task in workbench._visible_tasks]
        assert "Failed backup" in titles, titles


@pytest.mark.asyncio
async def test_conflict_card_shows_for_a_disabled_conflicted_task():
    app = _MissedApp()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_missed_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()

        conflict_text = workbench.query_one("#scheduling-conflict-text", Static)
        assert "Conflict detected" in str(conflict_text.render())
