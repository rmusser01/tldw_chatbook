from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    NotificationPreferencesUpdateRequest,
    NotificationSnoozeRequest,
    ReminderTaskCreateRequest,
    ReminderTaskUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_notifications_client_routes_feed_control_and_preferences(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "items": [
                    {
                        "id": 7,
                        "user_id": "user-1",
                        "kind": "reminder_due",
                        "title": "Due",
                        "message": "Time to follow up",
                        "severity": "info",
                        "created_at": "2026-04-24T12:00:00Z",
                    }
                ],
                "total": 1,
            },
            {"unread_count": 3},
            {"updated": 2},
            {"dismissed": True},
            {"task_id": "task-1", "run_at": "2026-04-24T12:30:00Z"},
            {"cancelled": True, "deleted_tasks": 1},
            {
                "user_id": "user-1",
                "reminder_enabled": True,
                "job_completed_enabled": True,
                "job_failed_enabled": False,
                "updated_at": "2026-04-24T12:00:00Z",
            },
            {
                "user_id": "user-1",
                "reminder_enabled": False,
                "job_completed_enabled": True,
                "job_failed_enabled": False,
                "updated_at": "2026-04-24T12:05:00Z",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_notifications(limit=25, offset=5, include_archived=True)
    unread = await client.get_notifications_unread_count()
    marked = await client.mark_notifications_read([7, 8])
    dismissed = await client.dismiss_notification(7)
    snoozed = await client.snooze_notification(7, NotificationSnoozeRequest(minutes=30))
    cancelled = await client.cancel_notification_snooze(7)
    preferences = await client.get_notification_preferences()
    updated_preferences = await client.update_notification_preferences(
        NotificationPreferencesUpdateRequest(reminder_enabled=False)
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/notifications")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "limit": 25,
        "offset": 5,
        "include_archived": True,
        "only_snoozed": False,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/notifications/unread-count")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/notifications/mark-read")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"ids": [7, 8]}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/notifications/7/dismiss")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/notifications/7/snooze")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"minutes": 30}
    assert mocked.await_args_list[5].args[:2] == ("DELETE", "/api/v1/notifications/7/snooze")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/notifications/preferences")
    assert mocked.await_args_list[7].args[:2] == ("PATCH", "/api/v1/notifications/preferences")
    assert mocked.await_args_list[7].kwargs["json_data"] == {"reminder_enabled": False}

    assert listed.items[0].id == 7
    assert unread.unread_count == 3
    assert marked.updated == 2
    assert dismissed.dismissed is True
    assert snoozed.task_id == "task-1"
    assert cancelled.deleted_tasks == 1
    assert preferences.reminder_enabled is True
    assert updated_preferences.reminder_enabled is False


@pytest.mark.asyncio
async def test_reminder_tasks_client_routes_crud(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    task_payload = {
        "id": "task-1",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "title": "Follow up",
        "schedule_kind": "one_time",
        "run_at": "2026-04-24T12:00:00Z",
        "enabled": True,
        "created_at": "2026-04-24T11:00:00Z",
        "updated_at": "2026-04-24T11:00:00Z",
    }
    mocked = AsyncMock(
        side_effect=[
            task_payload,
            {"items": [task_payload], "total": 1},
            task_payload,
            {**task_payload, "title": "Updated"},
            {"deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_reminder_task(
        ReminderTaskCreateRequest(
            title="Follow up",
            schedule_kind="one_time",
            run_at="2026-04-24T12:00:00Z",
        )
    )
    listed = await client.list_reminder_tasks()
    fetched = await client.get_reminder_task("task-1")
    updated = await client.update_reminder_task("task-1", ReminderTaskUpdateRequest(title="Updated"))
    deleted = await client.delete_reminder_task("task-1")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/tasks")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "title": "Follow up",
        "schedule_kind": "one_time",
        "run_at": "2026-04-24T12:00:00Z",
        "enabled": True,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/tasks")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/tasks/task-1")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/tasks/task-1")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"title": "Updated"}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/tasks/task-1")

    assert created.id == "task-1"
    assert listed.total == 1
    assert fetched.id == "task-1"
    assert updated.title == "Updated"
    assert deleted.deleted is True
