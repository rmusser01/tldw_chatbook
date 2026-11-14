"""TASK-36728: pending reminder edits keep their target and feedback owner."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import DataTable, Select

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail
from tldw_chatbook.Scheduling.events import (
    ReminderFieldEditRequested,
)

pytestmark = pytest.mark.asyncio


class HeldClient:
    notifications_service = None

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def update_reminder(self, server_id, **payload):
        self.calls.append((server_id, dict(payload)))
        if server_id == "server-a" and len(self.calls) == 1:
            self.entered.set()
            await self.release.wait()
        return {
            "id": server_id,
            "title": server_id,
            "schedule_kind": "recurring",
            "cron": "0 9 * * *",
            "timezone": "UTC",
            **payload,
        }


class ReminderHost(ConsolidatedCSSApp):
    CSS_PATH = list(APP_STYLESHEETS)

    def __init__(self, db, client):
        super().__init__()
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(active_server_id="1")
        )
        self.scheduling_service = SchedulingService(
            db=db, server_client=client, app_getter=lambda: self
        )


async def until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


async def settle(app, pilot):
    await pilot.pause()
    await until(lambda: not list(app.workers))
    await pilot.pause()


def seed(db, owner="server:1"):
    return [
        db.create_reminder_task(
            owner,
            name,
            server_id=server_id if owner != "local" else None,
            schedule_kind="recurring",
            cron="0 9 * * *",
            timezone="UTC",
        )
        for name, server_id in [("A", "server-a"), ("B", "server-b")]
    ]


async def select_reminder(screen, pilot, task_id):
    table = screen.query_one("#scheduling-task-table", DataTable)
    table.move_cursor(
        row=next(i for i, task in enumerate(screen._tasks) if task.id == task_id)
    )
    await pilot.pause()
    detail = screen.query_one("#scheduling-task-detail", TaskDetail)
    assert detail._current_task.id == task_id
    return detail


async def change_timezone(detail, pilot, value):
    row = detail._timezone_row
    row.focus()
    await pilot.press("enter")
    await pilot.pause()
    row.query_one(Select).value = value
    await pilot.pause()
    return row


@pytest.mark.parametrize("scenario", ["serial", "different", "same"])
@private_profile_test
async def test_pending_edits_preserve_captured_reminder_values(
    request, tmp_path, scenario
):
    db = ScheduledTasksDB(tmp_path / "reminders.db")
    client = HeldClient()
    ids = seed(db)
    app = ReminderHost(db, client)
    try:
        async with app.run_test(size=(180, 60)) as pilot:
            screen = SchedulesWorkbench(app_instance=app)
            await app.push_screen(screen)
            await settle(app, pilot)
            detail = await select_reminder(screen, pilot, ids[0])
            await change_timezone(detail, pilot, "America/New_York")
            await asyncio.wait_for(client.entered.wait(), 15)
            if scenario == "serial":
                client.release.set()
                await settle(app, pilot)
            target = ids[0] if scenario == "same" else ids[1]
            detail = await select_reminder(screen, pilot, target)
            await change_timezone(detail, pilot, "Europe/London")
            client.release.set()
            await settle(app, pilot)
            expected = (
                ["Europe/London", "UTC"]
                if scenario == "same"
                else ["America/New_York", "Europe/London"]
            )
            assert [db.get_reminder_task(i)["timezone"] for i in ids] == expected
            assert len(client.calls) == 2
            assert detail._current_task.id == target
            assert app._exception is None
    finally:
        client.release.set()
        db.close()


@pytest.mark.parametrize("completion", ["saved", "refused", "exception"])
@private_profile_test
async def test_pending_feedback_cannot_replace_another_reminders_error(
    request, tmp_path, monkeypatch, completion
):
    db = ScheduledTasksDB(tmp_path / "feedback.db")
    client = HeldClient()
    client.release.set()
    ids = seed(db)
    app = ReminderHost(db, client)
    entered, release = asyncio.Event(), asyncio.Event()
    original = app.scheduling_service.edit_reminder_fields

    async def held_edit(task_id, payload):
        entered.set()
        await release.wait()
        if completion == "exception":
            raise RuntimeError("controlled edit failure")
        if completion == "refused":
            return SimpleNamespace(
                status="error", errors=[{"message": "A was refused"}]
            )
        return await original(task_id, payload)

    monkeypatch.setattr(app.scheduling_service, "edit_reminder_fields", held_edit)
    try:
        async with app.run_test(size=(180, 60)) as pilot:
            screen = SchedulesWorkbench(app_instance=app)
            await app.push_screen(screen)
            await settle(app, pilot)
            first = await select_reminder(screen, pilot, ids[0])
            row = await change_timezone(first, pilot, "America/New_York")
            await asyncio.wait_for(entered.wait(), 15)
            second = await select_reminder(screen, pilot, ids[1])
            assert second._timezone_row is row
            # Seed an already displayed B validation error; A's completion must
            # neither erase it nor relabel it as A's own failure.
            row.show_error("B still needs correction")
            release.set()
            await settle(app, pilot)
            assert row._error_static.render().plain == "B still needs correction"
            if completion != "saved":
                assert any("edit" in n.message.lower() for n in app._notifications)
            assert app._exception is None
    finally:
        release.set()
        db.close()


@pytest.mark.parametrize("owner", ["local", "server:1"])
@private_profile_test
async def test_invalid_edit_keeps_its_row_retryable(request, tmp_path, owner):
    db = ScheduledTasksDB(tmp_path / "retry.db")
    client = HeldClient()
    client.release.set()
    ids = seed(db, owner)
    app = ReminderHost(db, client)
    try:
        async with app.run_test(size=(180, 60)) as pilot:
            screen = SchedulesWorkbench(app_instance=app)
            await app.push_screen(screen)
            await settle(app, pilot)
            detail = await select_reminder(screen, pilot, ids[0])
            row = detail._timezone_row
            screen.post_message(
                ReminderFieldEditRequested(
                    detail._current_task, {"timezone": "No/Such_Zone"}, row
                )
            )
            await settle(app, pilot)
            assert "Unknown timezone" in row._error_static.render().plain
            assert db.get_reminder_task(ids[0])["timezone"] == "UTC"
            await change_timezone(detail, pilot, "Europe/London")
            await settle(app, pilot)
            assert db.get_reminder_task(ids[0])["timezone"] == "Europe/London"
            assert row._error_static.render().plain == ""
            assert app._exception is None
    finally:
        client.release.set()
        db.close()
