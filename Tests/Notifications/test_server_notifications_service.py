from unittest.mock import Mock

import pytest

from tldw_chatbook.Notifications import ServerNotificationsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeServerNotificationsClient:
    def __init__(self):
        self.calls = []

    async def list_notifications(self, **kwargs):
        self.calls.append(("list_notifications", kwargs))
        return type(
            "Response",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "items": [
                        {
                            "id": 7,
                            "kind": "reminder_due",
                            "title": "Due",
                            "message": "Time to follow up",
                        }
                    ],
                    "total": 1,
                }
            },
        )()

    async def mark_notifications_read(self, ids):
        self.calls.append(("mark_notifications_read", ids))
        return {"updated": len(ids)}

    async def dismiss_notification(self, notification_id):
        self.calls.append(("dismiss_notification", notification_id))
        return {"dismissed": True}

    async def create_reminder_task(self, request_data):
        self.calls.append(("create_reminder_task", request_data.model_dump(exclude_none=True, mode="json")))
        return type(
            "Response",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "id": "task-1",
                    "title": "Follow up",
                    "schedule_kind": "one_time",
                }
            },
        )()

    async def list_reminder_tasks(self):
        self.calls.append(("list_reminder_tasks",))
        return type(
            "Response",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "items": [{"id": "task-1", "title": "Follow up"}],
                    "total": 1,
                }
            },
        )()

    async def update_reminder_task(self, task_id, request_data):
        self.calls.append(("update_reminder_task", task_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": task_id, "title": "Updated"}

    async def delete_reminder_task(self, task_id):
        self.calls.append(("delete_reminder_task", task_id))
        return {"deleted": True}


@pytest.mark.asyncio
async def test_server_notifications_service_routes_feed_and_reminders_with_policy_actions():
    client = FakeServerNotificationsClient()
    policy = Mock()
    service = ServerNotificationsService(client=client, policy_enforcer=policy)

    feed = await service.list_feed(limit=25, offset=5, include_archived=True)
    marked = await service.mark_read([7])
    dismissed = await service.dismiss(7)
    created = await service.create_reminder(
        title="Follow up",
        schedule_kind="one_time",
        run_at="2026-04-24T12:00:00Z",
    )
    reminders = await service.list_reminders()
    updated = await service.update_reminder("task-1", title="Updated")
    deleted = await service.delete_reminder("task-1")

    assert feed["total"] == 1
    assert marked["updated"] == 1
    assert dismissed["dismissed"] is True
    assert created["id"] == "task-1"
    assert reminders["total"] == 1
    assert updated["title"] == "Updated"
    assert deleted["deleted"] is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "notifications.feed.list.server",
        "notifications.feed.update.server",
        "notifications.feed.update.server",
        "notifications.reminders.configure.server",
        "notifications.reminders.list.server",
        "notifications.reminders.configure.server",
        "notifications.reminders.configure.server",
    ]
    assert client.calls[0] == (
        "list_notifications",
        {"limit": 25, "offset": 5, "include_archived": True, "only_snoozed": False},
    )
    assert client.calls[1] == ("mark_notifications_read", [7])
    assert client.calls[2] == ("dismiss_notification", 7)


@pytest.mark.asyncio
async def test_server_notifications_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeServerNotificationsClient()
    service = ServerNotificationsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_feed()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
