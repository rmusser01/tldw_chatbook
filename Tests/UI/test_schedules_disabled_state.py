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


# --- mounted behavior ------------------------------------------------------


class _DisabledTaskService:
    owner_id = "local"
    sync_engine = None

    class _DB:
        def get_sync_state(self, owner_id):
            return {}

        def get_conflicts(self, owner_id, primitive=None):
            return []

    db = _DB()

    class _ServerClient:
        notifications_service = None

    server_client = _ServerClient()

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
