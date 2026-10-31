"""Mounted reminder batches retain admission, captured scope and truthful outcomes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import DataTable

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench


class _HeldServerClient:
    notifications_service = object()

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls = []

    async def _request(self, verb, server_id, payload):
        self.calls.append((verb, server_id, dict(payload)))
        if not self.entered.is_set():
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return {
            "id": server_id,
            "title": server_id,
            "schedule_kind": "one_time",
            "run_at": "2099-01-01T00:00:00+00:00",
            **payload,
        }

    async def update_reminder(self, server_id, **payload):
        return await self._request("toggle", server_id, payload)

    async def delete_reminder(self, server_id):
        return await self._request("delete", server_id, {})


class _App(ConsolidatedCSSApp):
    def __init__(self, db, client):
        super().__init__()
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(active_server_id="1")
        )
        self.scheduling_service = SchedulingService(
            db=db, server_client=client, app_getter=lambda: self
        )


@pytest.fixture
def reminder_db(tmp_path):
    db = ScheduledTasksDB(tmp_path / "reminders.db")
    try:
        yield db
    finally:
        db.close()


def _seed(db, *, owner="server:1", count=3):
    return [
        db.create_reminder_task(
            owner,
            f"Task {index}",
            server_id=f"srv{index}" if owner != "local" else None,
            schedule_kind="one_time",
            run_at=f"2099-01-0{index + 1}T00:00:00+00:00",
            next_run_at=f"2099-01-0{index + 1}T00:00:00+00:00",
            enabled=True,
        )
        for index in range(count)
    ]


async def _settle(pilot):
    await pilot.pause()
    async with asyncio.timeout(8):
        while list(pilot.app.workers):
            await pilot.pause()
    await pilot.pause()


async def _mounted(pilot):
    screen = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(screen)
    await _settle(pilot)
    screen.query_one("#scheduling-task-table", DataTable).focus()
    await pilot.press("x", "down", "x")
    assert len(screen._marked_ids) == 2
    return screen


async def _gesture(pilot, verb):
    await pilot.press("space" if verb == "toggle" else "d")
    await pilot.pause()
    if verb == "delete" and pilot.app.screen.query("#confirm-button"):
        assert await pilot.click("#confirm-button")
        await pilot.pause()


def _messages(app):
    return [notification.message for notification in app._notifications]


def _assert_completed(db, ids, verb):
    if verb == "delete":
        assert [db.get_reminder_task(task_id) for task_id in ids[:2]] == [None, None]
    else:
        assert [db.get_reminder_task(task_id)["enabled"] for task_id in ids[:2]] == [
            0,
            0,
        ]
    assert db.get_reminder_task(ids[2])["enabled"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,later",
    [
        ("toggle", "toggle"),
        ("delete", "delete"),
        ("toggle", "delete"),
        ("delete", "toggle"),
    ],
)
@private_profile_test
async def test_pending_batch_refuses_same_and_cross_verb_overlap(
    request, reminder_db, first, later
):
    ids = _seed(reminder_db)
    client = _HeldServerClient()
    app = _App(reminder_db, client)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _mounted(pilot)
            assert screen._marked_ids == set(ids[:2])
            await _gesture(pilot, first)
            await asyncio.wait_for(client.entered.wait(), 8)
            await pilot.press("down")
            await _gesture(pilot, later)
            assert any(
                "busy" in message.lower() for message in _messages(app)
            ), _messages(app)
            assert not client.cancelled.is_set()
            assert len(client.calls) == 1
            client.release.set()
            await _settle(pilot)
            _assert_completed(reminder_db, ids, first)
            assert [(verb, task_id) for verb, task_id, _ in client.calls] == [
                (first, "srv0"),
                (first, "srv1"),
            ]
            assert not screen._marked_ids
            summary = "Toggled" if first == "toggle" else "Deleted"
            assert f"{summary} 2 marked tasks." in _messages(app)
            # Admission retires: a later deliberate selection can be acted on.
            await pilot.press("x")
            assert screen._marked_ids
    finally:
        client.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_pending_batch_refuses_mark_and_clear_gestures(request, reminder_db):
    ids = _seed(reminder_db)
    client = _HeldServerClient()
    app = _App(reminder_db, client)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _mounted(pilot)
            await _gesture(pilot, "toggle")
            await asyncio.wait_for(client.entered.wait(), 8)
            await pilot.press("escape")
            assert screen._marked_ids == set(ids[:2])
            await pilot.press("down", "x")
            assert screen.query_one("#scheduling-task-table", DataTable).cursor_row == 2
            assert screen._marked_ids == set(ids[:2])
            assert sum("busy" in message.lower() for message in _messages(app)) >= 2
            client.release.set()
            await _settle(pilot)
            _assert_completed(reminder_db, ids, "toggle")
            await pilot.press("x")
            assert screen._marked_ids == {ids[2]}
    finally:
        client.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_admission_freezes_ids_owners_and_desired_toggle_values(
    request, reminder_db, monkeypatch
):
    ids = _seed(reminder_db)
    client = _HeldServerClient()
    app = _App(reminder_db, client)
    release, entered = asyncio.Event(), asyncio.Event()
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _mounted(pilot)
            original_run = screen.run_worker

            def held_worker(work, **kwargs):
                if kwargs.get("group") != "schedules-bulk-toggle":
                    return original_run(work, **kwargs)

                async def held():
                    entered.set()
                    await release.wait()
                    await work()

                return original_run(held, **kwargs)

            monkeypatch.setattr(screen, "run_worker", held_worker)
            await _gesture(pilot, "toggle")
            await asyncio.wait_for(entered.wait(), 8)
            for task in screen._tasks[:2]:
                task.owner_id = "local"
                task.enabled = False
            app.scheduling_service.set_owner("local")
            # A refreshed model must not alter the already accepted operation.
            client.release.set()
            release.set()
            await _settle(pilot)
            assert client.calls == [
                ("toggle", "srv0", {"enabled": False}),
                ("toggle", "srv1", {"enabled": False}),
            ]
            _assert_completed(reminder_db, ids, "toggle")
    finally:
        release.set()
        client.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["toggle", "delete"])
@private_profile_test
async def test_submission_failure_releases_admission_for_retry(
    request, reminder_db, monkeypatch, verb
):
    ids = _seed(reminder_db)
    client = _HeldServerClient()
    client.release.set()
    app = _App(reminder_db, client)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _mounted(pilot)
        original_run = screen.run_worker

        def failed_submission(work, **kwargs):
            if kwargs.get("group") == f"schedules-bulk-{verb}":
                raise RuntimeError("test: worker submission refused")
            return original_run(work, **kwargs)

        monkeypatch.setattr(screen, "run_worker", failed_submission)
        if verb == "toggle":
            screen.action_toggle_enabled()
        else:
            screen._on_bulk_delete_confirmed(True, screen._marked_reminder_tasks())
        await pilot.pause()
        assert screen._marked_ids == set(ids[:2])
        assert not client.calls
        assert any("could not start" in message.lower() for message in _messages(app))
        monkeypatch.setattr(screen, "run_worker", original_run)
        await _gesture(pilot, verb)
        await _settle(pilot)
        _assert_completed(reminder_db, ids, verb)


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["toggle", "delete"])
@pytest.mark.parametrize("owner", ["local", "server:1"])
@private_profile_test
async def test_ordinary_batch_completes_and_clears_its_marks(
    request, reminder_db, owner, verb
):
    ids = _seed(reminder_db, owner=owner)
    client = _HeldServerClient()
    client.release.set()
    app = _App(reminder_db, client)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _mounted(pilot)
        await _gesture(pilot, verb)
        await _settle(pilot)
        _assert_completed(reminder_db, ids, verb)
        assert not screen._marked_ids
        assert len(client.calls) == (2 if owner == "server:1" else 0)
        summary = "Toggled" if verb == "toggle" else "Deleted"
        assert f"{summary} 2 marked tasks." in _messages(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["toggle", "delete"])
@private_profile_test
async def test_mixed_success_refusal_and_failure_reports_actual_count(
    request, reminder_db, monkeypatch, verb
):
    ids = _seed(reminder_db, owner="local")
    reminder_db.set_transfer_state(
        "reminder_task", ids[1], "to_server_pending", expected=(None,)
    )
    client = _HeldServerClient()
    app = _App(reminder_db, client)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _mounted(pilot)
        await pilot.press("down", "x")
        assert screen._marked_ids == set(ids)
        method = "update_reminder_task" if verb == "toggle" else "delete_reminder_task"
        original_write = getattr(reminder_db, method)

        def fail_one(task_id, **kwargs):
            if task_id == ids[2]:
                raise RuntimeError("test: SQLite write failed")
            return original_write(task_id, **kwargs)

        monkeypatch.setattr(reminder_db, method, fail_one)
        await _gesture(pilot, verb)
        await _settle(pilot)
        if verb == "delete":
            assert reminder_db.get_reminder_task(ids[0]) is None
        else:
            assert reminder_db.get_reminder_task(ids[0])["enabled"] == 0
        assert reminder_db.get_reminder_task(ids[1])["enabled"] == 1
        assert (
            reminder_db.get_reminder_task(ids[1])["transfer_state"]
            == "to_server_pending"
        )
        assert reminder_db.get_reminder_task(ids[2])["enabled"] == 1
        summary = "Toggled" if verb == "toggle" else "Deleted"
        assert f"{summary} 1 marked task (2 failed)." in _messages(app)
        assert not client.calls
        assert not screen._marked_ids
        # Refusals and exceptions both settle; the failed row can be retried.
        monkeypatch.setattr(reminder_db, method, original_write)
        table = screen.query_one("#scheduling-task-table", DataTable)
        table.move_cursor(
            row=next(
                i
                for i, row in enumerate(screen._visible_rows)
                if row.source_row.id == ids[2]
            )
        )
        await pilot.pause()
        await pilot.press("x")
        await _gesture(pilot, verb)
        await _settle(pilot)
        if verb == "delete":
            assert reminder_db.get_reminder_task(ids[2]) is None
        else:
            assert reminder_db.get_reminder_task(ids[2])["enabled"] == 0
