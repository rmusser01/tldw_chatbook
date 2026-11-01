from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from textual.widgets import DataTable, Select
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
OUT = Path(__file__).parent


class HeldClient:
    notifications_service = None

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.calls = []

    async def update_reminder(self, server_id, **payload):
        self.calls.append((server_id, dict(payload)))
        if (
            server_id == "server-a"
            and sum(call[0] == server_id for call in self.calls) == 1
        ):
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return {
            "id": server_id,
            "title": server_id,
            "schedule_kind": "recurring",
            "cron": "0 9 * * *",
            "timezone": "UTC",
            **payload,
        }


class Host(ConsolidatedCSSApp):
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
    async with asyncio.timeout(8):
        while not predicate():
            await asyncio.sleep(0.01)


async def settle(pilot):
    await pilot.pause()
    await until(lambda: not list(pilot.app.workers))
    await pilot.pause()


async def choose_and_edit(pilot, screen, task_id, value):
    table = screen.query_one("#scheduling-task-table", DataTable)
    index = next(i for i, task in enumerate(screen._tasks) if task.id == task_id)
    table.move_cursor(row=index)
    await pilot.pause()
    detail = screen.query_one("#scheduling-task-detail", TaskDetail)
    assert detail._current_task.id == task_id
    row = detail._timezone_row
    row.focus()
    await pilot.press("enter")
    await pilot.pause()
    editor = row.query_one(Select)
    editor.value = value
    await pilot.pause()
    return row


@pytest.mark.parametrize("scenario", ["serial", "overlap", "same_entity_latest"])
async def test_reminder_inline_publication(tmp_path, scenario):
    db = ScheduledTasksDB(tmp_path / "reminders.db")
    client = HeldClient()
    ids = [
        db.create_reminder_task(
            "server:1",
            name,
            server_id=server_id,
            schedule_kind="recurring",
            cron="0 9 * * *",
            timezone="UTC",
        )
        for name, server_id in [("A", "server-a"), ("B", "server-b")]
    ]
    app = Host(db, client)
    record = {
        "scenario": scenario,
        "ids": ids,
        "source": str(Path(__file__).resolve()),
        "production_modules": {
            kind.__name__: sys.modules[kind.__module__].__file__
            for kind in (SchedulesWorkbench, TaskDetail, SchedulingService)
        },
    }
    try:
        async with app.run_test(size=(180, 60)) as pilot:
            screen = SchedulesWorkbench(app_instance=app)
            await app.push_screen(screen)
            await settle(pilot)
            record["mounted_screen"] = screen.is_mounted
            record["rows"] = screen.query_one(
                "#scheduling-task-table", DataTable
            ).row_count
            row_a = await choose_and_edit(pilot, screen, ids[0], "America/New_York")
            await asyncio.wait_for(client.entered.wait(), 8)
            record["first_worker_live_before_second_edit"] = any(
                w.group == "schedules-edit-reminder-field-timezone" and w.is_running
                for w in app.workers
            )
            if scenario == "serial":
                client.release.set()
                await settle(pilot)
            later_id = ids[0] if scenario == "same_entity_latest" else ids[1]
            row_b = await choose_and_edit(pilot, screen, later_id, "Europe/London")
            record["row_widget_reused"] = row_a is row_b
            record["first_cancelled_after_second_edit"] = client.cancelled
            client.release.set()
            await settle(pilot)
            record["calls"] = client.calls
            record["stored_timezones"] = [
                db.get_reminder_task(i)["timezone"] for i in ids
            ]
            record["selected_id"] = screen.query_one(
                "#scheduling-task-detail", TaskDetail
            )._current_task.id
            record["notifications"] = [n.message for n in app._notifications]
            assert record["rows"] == 2
            assert len(client.calls) == 2
            assert (
                record["stored_timezones"][0 if scenario == "same_entity_latest" else 1]
                == "Europe/London"
            )
            if scenario == "serial":
                assert record["stored_timezones"][0] == "America/New_York"
                assert not client.cancelled
            elif scenario == "overlap":
                assert client.cancelled
                assert record["stored_timezones"][0] == "UTC"
            else:
                assert client.cancelled
                assert record["stored_timezones"][1] == "UTC"
    finally:
        client.release.set()
        record["first_cancelled_final"] = client.cancelled
        (OUT / f"reminder-inline-{scenario}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        db.close()
