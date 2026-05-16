import pytest

from tldw_chatbook.Notifications import EventStateRepository
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


class FakeLocalNotificationsService:
    def __init__(self):
        self.calls = []
        self.rows = [
            {
                "id": 3,
                "category": "watchlists",
                "title": "Local alert",
                "message": "Local item changed.",
                "severity": "warning",
                "is_read": False,
                "is_dismissed": False,
                "created_at": "2026-04-25T00:00:00+00:00",
            },
            {
                "id": 2,
                "category": "media",
                "title": "Older",
                "message": "Older local item.",
                "severity": "information",
                "is_read": True,
                "is_dismissed": False,
                "created_at": "2026-04-24T00:00:00+00:00",
            },
        ]

    def list_queue(self, **kwargs):
        self.calls.append(("list_queue", kwargs))
        return list(self.rows)

    def observe_queue(self, **kwargs):
        self.calls.append(("observe_queue", kwargs))
        after_id = int(kwargs.get("after_id", 0))
        return [row for row in self.rows if row["id"] > after_id]

    def update_notification(self, notification_id, **kwargs):
        self.calls.append(("update_notification", notification_id, kwargs))
        for row in self.rows:
            if row["id"] == int(notification_id):
                row.update({key: value for key, value in kwargs.items() if value is not None})
                return dict(row)
        raise KeyError(notification_id)

    def get_settings(self):
        self.calls.append(("get_settings",))
        return {
            "enabled": True,
            "toast_enabled": True,
            "persist_enabled": True,
            "category_preferences": {},
        }

    def update_settings(self, **settings):
        self.calls.append(("update_settings", settings))
        return {
            "enabled": settings.get("enabled", True),
            "toast_enabled": settings.get("toast_enabled", True),
            "persist_enabled": True,
            "category_preferences": settings.get("category_preferences", {}),
        }


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
async def test_notifications_scope_service_can_observe_server_feed_into_event_state(tmp_path):
    server = FakeServerNotificationsService()
    policy = FakePolicyEnforcer()
    repo = EventStateRepository(tmp_path / "events.db")
    scope = NotificationsScopeService(
        server_service=server,
        policy_enforcer=policy,
        event_state_repository=repo,
        server_event_scope_provider=lambda: {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "stream_instance_id": "workspace-1",
        },
    )

    result = await scope.observe_server_feed_events(mode="server", max_events=1)
    feed = scope.list_observed_server_feed(mark_presented=True)

    assert result.handled_events == 1
    assert feed["items"][0]["record_id"] == "server:notification:8"
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "evt-8"
    assert server.calls == [("observe_feed", {"after": 0, "last_event_id": None})]
    assert policy.calls == [
        "notifications.feed.observe.server",
        "notifications.feed.list.server",
    ]


def test_notifications_scope_service_enforces_policy_for_observed_server_feed(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    scope = NotificationsScopeService(
        policy_enforcer=FakePolicyEnforcer(denied_reason="server_feed_denied"),
        event_state_repository=repo,
        server_event_scope_provider=lambda: {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "stream_instance_id": "workspace-1",
        },
    )

    with pytest.raises(PolicyDeniedError) as exc:
        scope.list_observed_server_feed()

    assert exc.value.action_id == "notifications.feed.list.server"


@pytest.mark.asyncio
async def test_notifications_scope_service_requires_local_backend_for_local_feed():
    server = FakeServerNotificationsService()
    scope = NotificationsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Local notifications backend is unavailable"):
        await scope.list_feed(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_notifications_scope_service_routes_local_feed_to_client_queue():
    local = FakeLocalNotificationsService()
    server = FakeServerNotificationsService()
    policy = FakePolicyEnforcer()
    scope = NotificationsScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    feed = await scope.list_feed(mode="local", limit=10, category="watchlists")
    unread = await scope.unread_count(mode="local")
    marked = await scope.mark_read(mode="local", ids=[3])
    dismissed = await scope.dismiss(3, mode="local")
    observed = [event async for event in scope.observe_feed(mode="local", after_id=2, limit=10)]

    assert feed["items"][0]["record_id"] == "local:notification:3"
    assert feed["items"][0]["backend"] == "local"
    assert unread == {"backend": "local", "unread_count": 1}
    assert marked == {"backend": "local", "updated": 1}
    assert dismissed["record_id"] == "local:notification:3"
    assert dismissed["is_dismissed"] is True
    assert observed[0]["backend"] == "local"
    assert observed[0]["data"]["record_id"] == "local:notification:3"
    assert server.calls == []
    assert local.calls == [
        ("list_queue", {"limit": 10, "include_dismissed": False, "category": "watchlists"}),
        ("list_queue", {"limit": 1000, "include_dismissed": False}),
        ("update_notification", 3, {"is_read": True}),
        ("update_notification", 3, {"is_dismissed": True}),
        ("observe_queue", {"after_id": 2, "limit": 10, "include_dismissed": False}),
    ]
    assert policy.calls == [
        "notifications.queue.list.local",
        "notifications.queue.list.local",
        "notifications.queue.update.local",
        "notifications.queue.update.local",
        "notifications.queue.observe.local",
    ]


@pytest.mark.asyncio
async def test_notifications_scope_service_routes_local_settings_to_client_service():
    local = FakeLocalNotificationsService()
    policy = FakePolicyEnforcer()
    scope = NotificationsScopeService(local_service=local, policy_enforcer=policy)

    settings = await scope.get_settings(mode="local")
    updated = await scope.update_settings(mode="local", enabled=False, toast_enabled=False)

    assert settings["record_id"] == "local:notification_settings"
    assert settings["backend"] == "local"
    assert settings["enabled"] is True
    assert updated["record_id"] == "local:notification_settings"
    assert updated["enabled"] is False
    assert updated["toast_enabled"] is False
    assert local.calls == [
        ("get_settings",),
        ("update_settings", {"enabled": False, "toast_enabled": False}),
    ]
    assert policy.calls == [
        "notifications.settings.list.local",
        "notifications.settings.update.local",
    ]


@pytest.mark.asyncio
async def test_notifications_scope_service_keeps_local_reminders_server_owned():
    local = FakeLocalNotificationsService()
    scope = NotificationsScopeService(local_service=local, server_service=FakeServerNotificationsService())

    with pytest.raises(ValueError, match="Server reminders are server-only"):
        await scope.create_reminder(mode="local", title="Follow up", schedule_kind="one_time")

    assert local.calls == []


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
            "operation_id": "notifications.reminders.local",
            "source": "local",
            "supported": False,
            "reason_code": "server_authority_required",
            "user_message": "Server reminders are unavailable in local/offline mode; local Chatbook notifications use the local queue.",
            "affected_action_ids": [],
        }
    ]
