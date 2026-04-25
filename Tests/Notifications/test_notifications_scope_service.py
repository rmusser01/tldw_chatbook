import pytest

from tldw_chatbook.Notifications.notifications_scope_service import NotificationsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerNotificationsService:
    def __init__(self):
        self.calls = []

    async def list_feed(self, **kwargs):
        self.calls.append(("list_feed", kwargs))
        return {
            "items": [{"id": 7, "kind": "reminder_due", "title": "Due"}],
            "total": 1,
        }

    async def unread_count(self):
        self.calls.append(("unread_count",))
        return {"unread_count": 3}

    async def mark_read(self, ids):
        self.calls.append(("mark_read", ids))
        return {"updated": len(ids)}

    async def create_reminder(self, **kwargs):
        self.calls.append(("create_reminder", kwargs))
        return {"id": "task-1", "title": kwargs["title"]}

    async def list_reminders(self):
        self.calls.append(("list_reminders",))
        return {"items": [{"id": "task-1", "title": "Follow up"}], "total": 1}

    async def observe_feed(self, **kwargs):
        self.calls.append(("observe_feed", kwargs))
        yield {"event": "notification", "data": {"id": 8, "title": "Observed"}, "event_id": "evt-8"}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_notifications_scope_service_routes_server_feed_reminders_and_observe():
    server = FakeServerNotificationsService()
    policy = FakePolicyEnforcer()
    scope = NotificationsScopeService(server_service=server, policy_enforcer=policy)

    feed = await scope.list_feed(mode="server", limit=25)
    unread = await scope.unread_count(mode="server")
    marked = await scope.mark_read(mode="server", ids=[7])
    created = await scope.create_reminder(mode="server", title="Follow up", schedule_kind="one_time")
    reminders = await scope.list_reminders(mode="server")
    observed = [event async for event in scope.observe_feed(mode="server", after=7)]

    assert feed["items"][0]["record_id"] == "server:notification:7"
    assert feed["items"][0]["backend"] == "server"
    assert unread["backend"] == "server"
    assert marked["backend"] == "server"
    assert created["record_id"] == "server:reminder_task:task-1"
    assert reminders["items"][0]["record_id"] == "server:reminder_task:task-1"
    assert observed[0]["backend"] == "server"
    assert observed[0]["data"]["record_id"] == "server:notification:8"
    assert server.calls == [
        ("list_feed", {"limit": 25}),
        ("unread_count",),
        ("mark_read", [7]),
        ("create_reminder", {"title": "Follow up", "schedule_kind": "one_time"}),
        ("list_reminders",),
        ("observe_feed", {"after": 7}),
    ]
    assert policy.calls == [
        "notifications.feed.list.server",
        "notifications.feed.list.server",
        "notifications.feed.update.server",
        "notifications.reminders.configure.server",
        "notifications.reminders.list.server",
        "notifications.feed.observe.server",
    ]


@pytest.mark.asyncio
async def test_notifications_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeServerNotificationsService()
    scope = NotificationsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Server reminders and notification feeds are server-only"):
        await scope.list_feed(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_notifications_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeServerNotificationsService()
    scope = NotificationsScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_reminder(mode="server", title="Follow up", schedule_kind="one_time")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_notifications_scope_service_reports_known_unsupported_capabilities():
    scope = NotificationsScopeService(server_service=FakeServerNotificationsService())

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "notifications.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server reminders and notification feeds are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
