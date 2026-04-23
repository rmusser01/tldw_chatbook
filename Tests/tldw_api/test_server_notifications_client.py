from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.tldw_api import (
    NotificationCancelSnoozeResponse,
    NotificationDismissResponse,
    NotificationPreferencesResponse,
    NotificationPreferencesUpdateRequest,
    NotificationResponse,
    NotificationSnoozeRequest,
    NotificationSnoozeResponse,
    NotificationsListResponse,
    NotificationsMarkReadResponse,
    NotificationsUnreadCountResponse,
    ReminderTaskCreateRequest,
    ReminderTaskDeleteResponse,
    ReminderTaskListResponse,
    ReminderTaskResponse,
    ReminderTaskUpdateRequest,
    ServerNotificationStreamEvent,
    TLDWAPIClient,
)


def _reminder_payload(task_id: str = "task-1") -> dict:
    return {
        "id": task_id,
        "user_id": "7",
        "tenant_id": "tenant-1",
        "title": "Review claims",
        "body": "Follow up on reviewed claims",
        "link_type": "claim",
        "link_id": "claim-1",
        "link_url": None,
        "schedule_kind": "one_time",
        "run_at": "2026-04-23T20:00:00Z",
        "cron": None,
        "timezone": None,
        "enabled": True,
        "last_run_at": None,
        "next_run_at": "2026-04-23T20:00:00Z",
        "last_status": None,
        "created_at": "2026-04-23T19:00:00Z",
        "updated_at": "2026-04-23T19:00:00Z",
    }


def _notification_payload(notification_id: int = 11) -> dict:
    return {
        "id": notification_id,
        "user_id": "7",
        "kind": "reminder_due",
        "title": "Reminder due",
        "message": "Review claims",
        "severity": "info",
        "source_task_id": "task-1",
        "source_task_run_id": 5,
        "source_job_id": None,
        "source_domain": "notifications",
        "source_job_type": "reminder_due",
        "link_type": "claim",
        "link_id": "claim-1",
        "link_url": None,
        "dedupe_key": None,
        "retention_until": None,
        "archived_at": None,
        "created_at": "2026-04-23T20:00:00Z",
        "read_at": None,
        "dismissed_at": None,
        "snooze_until": None,
    }


@pytest.mark.asyncio
async def test_client_routes_reminder_task_crud_calls():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            _reminder_payload("task-1"),
            {"items": [_reminder_payload("task-1")], "total": 1},
            _reminder_payload("task-1"),
            _reminder_payload("task-1") | {"enabled": False},
            {"deleted": True},
        ]
    )

    created = await client.create_reminder_task(
        ReminderTaskCreateRequest(
            title="Review claims",
            body="Follow up on reviewed claims",
            schedule_kind="one_time",
            run_at="2026-04-23T20:00:00Z",
            link_type="claim",
            link_id="claim-1",
        )
    )
    listed = await client.list_reminder_tasks()
    detail = await client.get_reminder_task("task-1")
    updated = await client.update_reminder_task("task-1", ReminderTaskUpdateRequest(enabled=False))
    deleted = await client.delete_reminder_task("task-1")

    assert isinstance(created, ReminderTaskResponse)
    assert isinstance(listed, ReminderTaskListResponse)
    assert isinstance(detail, ReminderTaskResponse)
    assert isinstance(updated, ReminderTaskResponse)
    assert isinstance(deleted, ReminderTaskDeleteResponse)
    assert listed.total == 1
    assert updated.enabled is False
    assert deleted.deleted is True
    assert [call.args for call in client._request.await_args_list] == [
        ("POST", "/api/v1/tasks"),
        ("GET", "/api/v1/tasks"),
        ("GET", "/api/v1/tasks/task-1"),
        ("PATCH", "/api/v1/tasks/task-1"),
        ("DELETE", "/api/v1/tasks/task-1"),
    ]
    assert client._request.await_args_list[0].kwargs["json_data"] == {
        "title": "Review claims",
        "body": "Follow up on reviewed claims",
        "schedule_kind": "one_time",
        "run_at": "2026-04-23T20:00:00Z",
        "link_type": "claim",
        "link_id": "claim-1",
        "enabled": True,
    }
    assert client._request.await_args_list[3].kwargs["json_data"] == {"enabled": False}


@pytest.mark.asyncio
async def test_client_routes_notification_feed_actions():
    client = TLDWAPIClient("http://example.test", "token")
    preferences_payload = {
        "user_id": "7",
        "reminder_enabled": True,
        "job_completed_enabled": True,
        "job_failed_enabled": True,
        "updated_at": "2026-04-23T20:00:00Z",
    }
    client._request = AsyncMock(
        side_effect=[
            {"items": [_notification_payload(11)], "total": 1},
            {"unread_count": 1},
            {"updated": 1},
            {"dismissed": True},
            {"task_id": "snooze-1", "run_at": "2026-04-23T21:00:00Z"},
            {"cancelled": True, "deleted_tasks": 1},
            preferences_payload,
            preferences_payload | {"job_failed_enabled": False},
        ]
    )

    listed = await client.list_server_notifications(limit=25, offset=5, include_archived=True, only_snoozed=False)
    unread = await client.get_server_notifications_unread_count()
    read = await client.mark_server_notifications_read([11])
    dismissed = await client.dismiss_server_notification(11)
    snoozed = await client.snooze_server_notification(11, NotificationSnoozeRequest(minutes=45))
    cancelled = await client.cancel_server_notification_snooze(11)
    prefs = await client.get_server_notification_preferences()
    updated_prefs = await client.update_server_notification_preferences(
        NotificationPreferencesUpdateRequest(job_failed_enabled=False)
    )

    assert isinstance(listed, NotificationsListResponse)
    assert isinstance(listed.items[0], NotificationResponse)
    assert isinstance(unread, NotificationsUnreadCountResponse)
    assert isinstance(read, NotificationsMarkReadResponse)
    assert isinstance(dismissed, NotificationDismissResponse)
    assert isinstance(snoozed, NotificationSnoozeResponse)
    assert isinstance(cancelled, NotificationCancelSnoozeResponse)
    assert isinstance(prefs, NotificationPreferencesResponse)
    assert isinstance(updated_prefs, NotificationPreferencesResponse)
    assert updated_prefs.job_failed_enabled is False
    assert [call.args for call in client._request.await_args_list] == [
        ("GET", "/api/v1/notifications"),
        ("GET", "/api/v1/notifications/unread-count"),
        ("POST", "/api/v1/notifications/mark-read"),
        ("POST", "/api/v1/notifications/11/dismiss"),
        ("POST", "/api/v1/notifications/11/snooze"),
        ("DELETE", "/api/v1/notifications/11/snooze"),
        ("GET", "/api/v1/notifications/preferences"),
        ("PATCH", "/api/v1/notifications/preferences"),
    ]
    assert client._request.await_args_list[0].kwargs["params"] == {
        "limit": 25,
        "offset": 5,
        "include_archived": True,
        "only_snoozed": False,
    }
    assert client._request.await_args_list[2].kwargs["json_data"] == {"ids": [11]}
    assert client._request.await_args_list[4].kwargs["json_data"] == {"minutes": 45}
    assert client._request.await_args_list[7].kwargs["json_data"] == {"job_failed_enabled": False}


@pytest.mark.asyncio
async def test_client_streams_server_notifications_from_sse_path():
    async def fake_stream():
        yield ServerNotificationStreamEvent(
            event="notification",
            id="11",
            data={"notification_id": 11, "title": "Reminder due"},
        )
        yield ServerNotificationStreamEvent(
            event="heartbeat",
            id=None,
            data={},
        )

    client = TLDWAPIClient("http://example.test", "token")
    client._stream_sse_request = Mock(return_value=fake_stream())

    events = [
        event async for event in client.stream_server_notifications(after=10)
    ]

    assert events[0].event == "notification"
    assert events[0].data["notification_id"] == 11
    assert events[1].event == "heartbeat"
    client._stream_sse_request.assert_called_once_with(
        "/api/v1/notifications/stream",
        params={"after": 10},
        event_model=ServerNotificationStreamEvent,
    )
