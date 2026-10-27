"""TASK-37032: full ReminderForm Saves retain every admitted mutation."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from textual.widgets import Button, DataTable, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.test_schedules_reminder_edit_identity import (
    ReminderHost,
    seed,
    select_reminder,
    settle,
    until,
)
from tldw_chatbook.Notifications.server_notifications_service import (
    ServerNotificationsService,
)
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerUnavailableError,
)
from tldw_chatbook.Scheduling.services.sync_engine import SyncEngine
from tldw_chatbook.UI.Screens.scheduling.forms.new_task_choice_modal import (
    NewTaskChoiceModal,
)
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench

pytestmark = pytest.mark.asyncio


class AcceptedClient:
    """Only the external response is controlled; service/mirror writes are real."""

    notifications_service = None

    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls, self.effects, self.cancelled = [], {}, []

    async def create_reminder(self, **payload):
        server_id = "server-a" if payload["title"] == "EditedA" else "server-b"
        return await self.update_reminder(server_id, **payload)

    async def update_reminder(self, server_id, **payload):
        self.calls.append((server_id, dict(payload)))
        # A controlled external effect precedes the held acknowledgement.
        self.effects[server_id] = dict(payload)
        if server_id == "server-a" and len(self.calls) == 1:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.append(server_id)
                raise
        return {
            "id": server_id,
            "title": server_id,
            "schedule_kind": "recurring",
            "cron": "0 9 * * *",
            "timezone": "UTC",
            **payload,
        }


class ReplayAPI:
    """Controlled API transport below the real notification method/schema."""

    def __init__(self):
        self.updates = []

    async def list_reminder_tasks(self):
        return {
            "items": [
                {
                    "id": server_id,
                    "title": title,
                    "schedule_kind": "recurring",
                    "cron": "0 9 * * *",
                    "timezone": "UTC",
                    "sync_version": 0,
                }
                for server_id, title in [("server-a", "A"), ("server-b", "B")]
            ]
        }

    async def update_reminder_task(self, task_id, request):
        payload = request.model_dump(mode="json", exclude_unset=True)
        self.updates.append((task_id, payload))
        return {"id": task_id, **payload}


def assert_worktree_sources():
    root = Path(__file__).resolve().parents[2]
    for obj in (
        SchedulesWorkbench,
        ReminderForm,
        SchedulingService,
        ScheduledTasksDB,
        SyncEngine,
        SchedulingServerClient,
        ServerNotificationsService,
    ):
        assert Path(inspect.getfile(obj)).resolve().is_relative_to(root)


async def open_form(app, screen, pilot, task_id, title):
    screen.query_one("#scheduling-task-table", DataTable).focus()
    if task_id is None:
        await pilot.press("n")
        await until(lambda: isinstance(app.screen, NewTaskChoiceModal))
        await pilot.press("enter")
    else:
        await select_reminder(screen, pilot, task_id)
        screen.query_one("#scheduling-task-table", DataTable).focus()
        await pilot.press("e")
    await until(lambda: isinstance(app.screen, ReminderForm))
    await pilot.pause()
    form = app.screen
    field = form.query_one("#reminder-title", Input)
    field.focus()
    await pilot.press("home", "shift+end", "backspace", *title)
    await pilot.pause()
    assert field.value == title
    if task_id is None:
        form.query_one("#reminder-run-at", Input).focus()
        await pilot.press(
            "home", "shift+end", "backspace", *"2099-01-01T09:00:00+00:00"
        )
        await pilot.pause()
    return form


async def save_form(app, screen, pilot, form, *, repeated=False):
    button = form.query_one("#reminder-save", Button)
    button.scroll_visible()
    await pilot.pause()
    assert button.visible and button.region.area > 0
    button.focus()
    await pilot.press(*(["enter", "enter"] if repeated else ["enter"]))
    await until(lambda: app.screen is screen)
    await pilot.pause()


def assert_queue_titles(screen, titles):
    table = screen.query_one("#scheduling-task-table", DataTable)
    painted = " ".join(str(cell) for key in table.rows for cell in table.get_row(key))
    for title in titles:
        assert title in painted


@pytest.mark.parametrize("operation", ["edit", "create"])
@pytest.mark.parametrize("scenario", ["serial", "overlap"])
@private_profile_test
async def test_full_forms_keep_both_accepted_saves(
    request, tmp_path, monkeypatch, scenario, operation
):
    assert_worktree_sources()
    db = ScheduledTasksDB(tmp_path / "fullforms.db")
    client = AcceptedClient()
    ids = seed(db) if operation == "edit" else [None, None]
    app = ReminderHost(db, client)
    app.scheduling_service.set_owner("server:1")
    notices = []
    notify = app.notify

    def observe(message, **kwargs):
        notices.append((message, kwargs.get("severity")))
        return notify(message, **kwargs)

    monkeypatch.setattr(app, "notify", observe)
    try:
        async with app.run_test(size=(180, 60)) as pilot:
            screen = SchedulesWorkbench(app_instance=app)
            await app.push_screen(screen)
            try:
                await settle(app, pilot)
                form = await open_form(app, screen, pilot, ids[0], "EditedA")
                await save_form(
                    app, screen, pilot, form, repeated=operation == "create"
                )
                await asyncio.wait_for(client.entered.wait(), 15)
                if scenario == "serial":
                    client.release.set()
                    await settle(app, pilot)
                form = await open_form(app, screen, pilot, ids[1], "EditedB")
                await save_form(
                    app, screen, pilot, form, repeated=operation == "create"
                )
                await until(lambda: "server-b" in client.effects)
                app.scheduling_service.set_owner("local")
                client.release.set()
                await settle(app, pilot)
                assert [server_id for server_id, _ in client.calls] == [
                    "server-a",
                    "server-b",
                ]
                assert [
                    row["owner_id"] for row in db.list_reminder_tasks(owner_id=None)
                ] == ["server:1", "server:1"]
                assert [
                    client.effects[key]["title"] for key in ("server-a", "server-b")
                ] == ["EditedA", "EditedB"]
                assert sorted(
                    row["title"] for row in db.list_reminder_tasks(owner_id=None)
                ) == ["EditedA", "EditedB"]
                assert_queue_titles(screen, ["EditedA", "EditedB"])
                assert client.cancelled == []
                verb = "updated" if operation == "edit" else "created"
                assert notices.count((f"Scheduled task {verb}.", "information")) == 2
                assert app._exception is None
            finally:
                client.release.set()
                await settle(app, pilot)
    finally:
        client.release.set()
        db.close()


@pytest.mark.parametrize(
    "outcome",
    ["write_failure", "transfer", "deleted", "offline_recurring", "offline_one_time"],
)
@private_profile_test
async def test_full_form_refusal_failure_and_offline_outcomes(
    request, tmp_path, monkeypatch, outcome
):
    assert_worktree_sources()
    db = ScheduledTasksDB(tmp_path / "outcomes.db")
    client = AcceptedClient()
    client.release.set()
    ids = seed(db, "server:1" if outcome.startswith("offline") else "local")
    app = ReminderHost(db, client)
    notices = []
    original_notify = app.notify

    def observe(message, **kwargs):
        notices.append((message, kwargs.get("severity")))
        return original_notify(message, **kwargs)

    monkeypatch.setattr(app, "notify", observe)
    try:
        async with app.run_test(size=(180, 60)) as pilot:
            screen = SchedulesWorkbench(app_instance=app)
            await app.push_screen(screen)
            try:
                await settle(app, pilot)
                form = await open_form(app, screen, pilot, ids[0], "EditedA")
                if outcome == "offline_one_time":
                    kind = form.query_one("#reminder-kind", Select)
                    kind.focus()
                    await pilot.press("enter", "home", "enter")
                    await pilot.pause()
                    assert kind.value == "one_time"
                    form.query_one("#reminder-run-at", Input).focus()
                    await pilot.press(
                        "home", "shift+end", "backspace", *"2099-01-01T09:00:00+00:00"
                    )
                    await pilot.pause()
                original_write = db.update_reminder_task
                if outcome == "write_failure":

                    def fail_write(task_id, **payload):
                        if task_id == ids[0]:
                            raise RuntimeError("controlled SQLite write failure")
                        return original_write(task_id, **payload)

                    monkeypatch.setattr(db, "update_reminder_task", fail_write)
                elif outcome == "transfer":
                    db.set_transfer_state(
                        "reminder_task", ids[0], "to_server_sent", expected=(None,)
                    )
                elif outcome == "deleted":
                    db.delete_reminder_task(ids[0])
                else:

                    async def offline(server_id, **payload):
                        raise ServerUnavailableError("controlled offline response")

                    monkeypatch.setattr(client, "update_reminder", offline)
                await save_form(app, screen, pilot, form)
                await settle(app, pilot)
                if outcome.startswith("offline"):
                    assert db.get_reminder_task(ids[0])["title"] == "EditedA"
                    pending = db.get_pending_mutations(owner_id="server:1")
                    assert len(pending) == 1
                    stable_key = pending[0]["payload"]["idempotency_key"]
                    assert stable_key
                    assert (
                        db.get_pending_mutations("server:1")[0]["payload"][
                            "idempotency_key"
                        ]
                        == stable_key
                    )
                    fields = pending[0]["payload"]["fields"]
                    assert fields["title"] == "EditedA"
                    assert "next_run_at" not in fields
                    if outcome == "offline_one_time":
                        assert fields["run_at"] == "2099-01-01T09:00:00+00:00"
                        assert fields["cron"] is None
                    else:
                        assert fields["run_at"] is None
                        assert fields["cron"] == "0 9 * * *"
                        assert fields["timezone"] == "UTC"
                    api = ReplayAPI()
                    replay_client = SchedulingServerClient(
                        ServerNotificationsService(api)
                    )
                    engine = SyncEngine(db, replay_client, owner_id="server:1")
                    result = await engine._sync_reminders("server:1")
                    assert result.status == "ok", result
                    assert db.get_pending_mutations(owner_id="server:1") == []
                    assert len(api.updates) == 1
                    server_id, sent = api.updates[0]
                    assert server_id == "server-a"
                    assert sent["title"] == "EditedA"
                    assert "idempotency_key" not in sent
                    if outcome == "offline_one_time":
                        assert sent["run_at"] == "2099-01-01T09:00:00+00:00"
                    else:
                        assert sent["cron"] == "0 9 * * *"
                        assert sent["timezone"] == "UTC"
                    # A later partial update retains explicit nulls without
                    # expanding into a full task snapshot or a derived clock.
                    await app.scheduling_service.update_reminder(
                        ids[0], {"body": None}, owner_id="server:1"
                    )
                    partial = db.get_pending_mutations(owner_id="server:1")
                    assert partial[0]["payload"]["fields"] == {"body": None}
                    assert_queue_titles(screen, ["EditedA", "B"])
                    assert ("Scheduled task updated.", "information") in notices
                else:
                    assert ("Scheduled task updated.", "information") not in notices
                    assert any(
                        severity in {"warning", "error"} for _, severity in notices
                    )
                    if outcome == "deleted":
                        assert db.get_reminder_task(ids[0]) is None
                        assert [task.id for task in screen._tasks] == [ids[1]]
                    else:
                        assert db.get_reminder_task(ids[0])["title"] == "A"
                        assert_queue_titles(screen, ["A", "B"])
                    if outcome == "write_failure":
                        monkeypatch.setattr(db, "update_reminder_task", original_write)
                        form = await open_form(app, screen, pilot, ids[0], "RetriedA")
                        await save_form(app, screen, pilot, form)
                        await settle(app, pilot)
                        assert db.get_reminder_task(ids[0])["title"] == "RetriedA"
                        assert_queue_titles(screen, ["RetriedA", "B"])
                assert app._exception is None
            finally:
                client.release.set()
                await settle(app, pilot)
    finally:
        db.close()
