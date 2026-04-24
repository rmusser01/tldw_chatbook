from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Notifications.server_notifications_scope_service import ServerNotificationsScopeService
from tldw_chatbook.Notifications.server_notifications_service import ServerNotificationsService
from tldw_chatbook.tldw_api import ServerNotificationStreamEvent


def _reminder_payload(task_id: str = "task-1") -> dict:
    return {
        "id": task_id,
        "user_id": "7",
        "tenant_id": "tenant-1",
        "title": "Review claims",
        "body": "Follow up",
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


class _PolicyRecorder:
    def __init__(self) -> None:
        self.action_ids: list[str] = []

    def require_allowed(self, *, action_id: str):
        self.action_ids.append(action_id)
        return SimpleNamespace(allowed=True)


@pytest.mark.asyncio
async def test_server_notifications_service_normalizes_reminders_and_feed_rows():
    client = AsyncMock()
    client.list_reminder_tasks.return_value = SimpleNamespace(
        model_dump=lambda mode="json": {"items": [_reminder_payload()], "total": 1}
    )
    client.list_server_notifications.return_value = SimpleNamespace(
        model_dump=lambda mode="json": {"items": [_notification_payload()], "total": 1}
    )
    service = ServerNotificationsService(client=client)

    reminders = await service.list_reminders()
    feed = await service.list_feed(limit=25, offset=0)

    assert reminders["items"][0]["id"] == "server:reminder_task:task-1"
    assert reminders["items"][0]["entity_kind"] == "reminder_task"
    assert reminders["items"][0]["task_id"] == "task-1"
    assert feed["items"][0]["id"] == "server:notification:11"
    assert feed["items"][0]["entity_kind"] == "server_notification"
    assert feed["items"][0]["notification_id"] == 11
    client.list_reminder_tasks.assert_awaited_once()
    client.list_server_notifications.assert_awaited_once_with(
        limit=25,
        offset=0,
        include_archived=False,
        only_snoozed=False,
    )


@pytest.mark.asyncio
async def test_server_notifications_service_routes_preferences():
    preferences_payload = {
        "user_id": "7",
        "reminder_enabled": True,
        "job_completed_enabled": True,
        "job_failed_enabled": False,
        "updated_at": "2026-04-23T20:00:00Z",
    }
    client = AsyncMock()
    client.get_server_notification_preferences.return_value = preferences_payload
    client.update_server_notification_preferences.return_value = preferences_payload | {
        "job_failed_enabled": True
    }
    service = ServerNotificationsService(client=client)

    preferences = await service.get_preferences()
    updated = await service.update_preferences({"job_failed_enabled": True})

    assert preferences["job_failed_enabled"] is False
    assert updated["job_failed_enabled"] is True
    client.get_server_notification_preferences.assert_awaited_once()
    client.update_server_notification_preferences.assert_awaited_once()
    request = client.update_server_notification_preferences.await_args.args[0]
    assert request.model_dump(exclude_none=True) == {"job_failed_enabled": True}


@pytest.mark.asyncio
async def test_scope_service_enforces_remote_policy_and_rejects_local_mode():
    server_service = AsyncMock()
    server_service.list_reminders.return_value = {"items": [], "total": 0}
    server_service.save_reminder.return_value = {"id": "server:reminder_task:task-1"}
    server_service.delete_reminder.return_value = {"deleted": True}
    server_service.list_feed.return_value = {"items": [], "total": 0}
    server_service.mark_notification_read.return_value = {"updated": 1}
    server_service.dismiss_notification.return_value = {"dismissed": True}
    server_service.snooze_notification.return_value = {"task_id": "snooze-1"}
    server_service.cancel_notification_snooze.return_value = {"cancelled": True}
    server_service.get_preferences.return_value = {"job_failed_enabled": False}
    server_service.update_preferences.return_value = {"job_failed_enabled": True}
    policy = _PolicyRecorder()
    scope = ServerNotificationsScopeService(server_service=server_service, policy_enforcer=policy)

    await scope.list_reminders(runtime_backend="server")
    await scope.save_reminder(runtime_backend="server", payload={"title": "Review"})
    await scope.delete_reminder(runtime_backend="server", task_id="server:reminder_task:task-1")
    await scope.list_feed(runtime_backend="server")
    await scope.mark_notification_read(runtime_backend="server", notification_id="server:notification:11")
    await scope.dismiss_notification(runtime_backend="server", notification_id="server:notification:11")
    await scope.snooze_notification(runtime_backend="server", notification_id="server:notification:11", minutes=30)
    await scope.cancel_notification_snooze(runtime_backend="server", notification_id="server:notification:11")
    preferences = await scope.get_feed_preferences(runtime_backend="server")
    updated_preferences = await scope.update_feed_preferences(
        runtime_backend="server",
        payload={"job_failed_enabled": True},
    )

    assert preferences == {"job_failed_enabled": False}
    assert updated_preferences == {"job_failed_enabled": True}
    assert policy.action_ids == [
        "notifications.reminders.list.server",
        "notifications.reminders.configure.server",
        "notifications.reminders.configure.server",
        "notifications.feed.list.server",
        "notifications.feed.observe.server",
        "notifications.feed.observe.server",
        "notifications.reminders.launch.server",
        "notifications.reminders.launch.server",
        "notifications.feed.list.server",
        "notifications.feed.configure.server",
    ]
    server_service.delete_reminder.assert_awaited_once_with("task-1")
    server_service.mark_notification_read.assert_awaited_once_with(11)
    server_service.update_preferences.assert_awaited_once_with({"job_failed_enabled": True})

    with pytest.raises(ValueError, match="server mode"):
        await scope.list_feed(runtime_backend="local")


@pytest.mark.asyncio
async def test_scope_service_streams_feed_events_with_policy():
    async def fake_stream():
        yield {"event": "notification", "id": "11", "data": {"notification_id": 11}}

    server_service = Mock()
    server_service.stream_feed_events = Mock(return_value=fake_stream())
    policy = _PolicyRecorder()
    scope = ServerNotificationsScopeService(server_service=server_service, policy_enforcer=policy)

    events = [
        event async for event in scope.stream_feed_events(runtime_backend="server", after=10)
    ]

    assert events == [{"event": "notification", "id": "11", "data": {"notification_id": 11}}]
    assert policy.action_ids == ["notifications.feed.observe.server"]
    server_service.stream_feed_events.assert_called_once_with(after=10)


@pytest.mark.asyncio
async def test_server_notifications_service_streams_plain_dict_events():
    async def fake_stream():
        yield ServerNotificationStreamEvent(
            event="notification",
            id="11",
            data={"notification_id": 11, "title": "Reminder due"},
        )

    client = AsyncMock()
    client.stream_server_notifications = Mock(return_value=fake_stream())
    service = ServerNotificationsService(client=client)

    events = [event async for event in service.stream_feed_events(after=10)]

    assert events == [
        {
            "event": "notification",
            "id": "11",
            "data": {"notification_id": 11, "title": "Reminder due"},
        }
    ]
    client.stream_server_notifications.assert_called_once_with(after=10)
