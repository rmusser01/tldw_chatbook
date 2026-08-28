"""task-23106: one noun for user-created items -- "scheduled task".

User-facing copy mixed "Schedules" (nav), "New Scheduled Task" (form),
"Reminder created." (toast), "Only reminder tasks can be edited here."
(guard). Rows managed by other systems must say what they are and where
to edit them instead of exposing the internal "reminder" noun.
"""

from datetime import datetime, timezone

import pytest
from textual.widgets import DataTable, Input, Static

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
    _managed_elsewhere_notice,
)


def _projection(task_type: str = "watchlist_job") -> ScheduledTask:
    return ScheduledTask(
        id=f"{task_type}:1",
        title="Watchlist Title",
        type=task_type,
        status=TaskStatus.WAITING,
        schedule_summary="Every 1h",
        next_run_at=datetime(2099, 7, 20, 11, 0, tzinfo=timezone.utc),
    )


def test_managed_elsewhere_notice_names_the_owning_screen():
    assert _managed_elsewhere_notice(_projection("watchlist_job")) == (
        "Managed by Watchlists — edit it there."
    )
    assert _managed_elsewhere_notice(_projection("briefing_job")) == (
        "Managed by Watchlists — edit it there."
    )
    assert "reminder" not in _managed_elsewhere_notice(
        _projection("mystery_job")
    ).lower()


class _MixedService:
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
        self.created: list[dict] = []

    async def list_tasks(self):
        return [
            ReminderTask(
                id="task-1",
                title="Reminder",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 7, 20, 10, 0, tzinfo=timezone.utc),
                next_run_at=datetime(2099, 7, 20, 10, 0, tzinfo=timezone.utc),
            ),
            _projection(),
        ]

    async def create_reminder(self, payload: dict):
        self.created.append(payload)


class _App(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _MixedService()


async def _mounted_workbench(pilot):
    workbench = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(workbench)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench


@pytest.mark.asyncio
async def test_edit_guard_on_projection_names_the_owner_not_reminders():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)  # the watchlist projection
        await pilot.pause()

        workbench.action_edit_task()
        await pilot.pause()

        messages = [n.message for n in pilot.app._notifications]
        assert any("Managed by Watchlists" in m for m in messages), messages
        assert not any("reminder" in m.lower() for m in messages), messages


@pytest.mark.asyncio
async def test_toggle_guard_on_projection_names_the_owner_not_reminders():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()

        workbench.action_toggle_enabled()
        await pilot.pause()

        messages = [n.message for n in pilot.app._notifications]
        assert any("Managed by Watchlists" in m for m in messages), messages
        assert not any("reminder" in m.lower() for m in messages), messages


@pytest.mark.asyncio
async def test_detail_pane_states_projection_ownership():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_workbench(pilot)
        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(row=1)
        await pilot.pause()

        managed = workbench.query_one(
            "#scheduling-task-detail-managed", Static
        )
        assert managed.display
        assert "Managed by Watchlists" in str(managed.render())

        # Selecting the reminder row hides the ownership line again.
        table.move_cursor(row=0)
        await pilot.pause()
        assert not managed.display


@pytest.mark.asyncio
async def test_create_toast_uses_the_scheduled_task_noun():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_workbench(pilot)
        workbench._on_reminder_form_result(
            {
                "title": "T",
                "body": "",
                "schedule_kind": "one_time",
                "run_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
                "cron": None,
                "timezone": None,
            }
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        messages = [n.message for n in pilot.app._notifications]
        assert "Scheduled task created." in messages, messages


def test_no_bare_reminder_noun_in_workbench_or_detail_toasts():
    """Static sweep: notify()/tooltip copy must not expose "reminder".

    The internal identifiers (ReminderTask, create_reminder, ...) are
    deliberately untouched (task-23106) -- this checks string literals
    only, on the two modules that own the Schedules screen's copy.
    """
    import re
    from pathlib import Path

    import tldw_chatbook.UI.Screens.scheduling.schedules_workbench as wb
    import tldw_chatbook.UI.Screens.scheduling.task_detail as td

    for module in (wb, td):
        source = Path(module.__file__).read_text(encoding="utf-8")
        offenders = [
            line.strip()
            for line in source.splitlines()
            if re.search(r'"[^"]*[Rr]eminder[^"]*"', line)
            and "notify(" not in line  # message built on preceding lines
            and not line.strip().startswith("#")
            and '"""' not in line  # docstrings are internal, not copy
            and "logger" not in line
            and "Untitled reminder" not in line
        ]
        # Allow identifier-ish usage (ids, group names, event names) but
        # not sentence copy: sentence copy contains a space around the noun.
        sentence_offenders = [
            line
            for line in offenders
            if re.search(r'"[^"]*\b[Rr]eminders?\b[ .,\'][^"]*"', line)
        ]
        assert not sentence_offenders, sentence_offenders
