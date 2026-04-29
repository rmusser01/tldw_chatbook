import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Notifications.server_notifications_service as notifications_module
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


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class FreshClientProvider:
    def __init__(self, factory):
        self.factory = factory
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = self.factory()
        self.clients.append(client)
        return client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


def test_server_notifications_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(notifications_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_notifications_service_direct_client_takes_precedence_over_provider():
    client = FakeServerNotificationsClient()
    provider = ExplodingClientProvider()
    service = ServerNotificationsService(client=client, client_provider=provider)

    feed = await service.list_feed(limit=25, offset=5, include_archived=True)

    assert feed["total"] == 1
    assert provider.build_calls == 0
    assert client.calls == [
        (
            "list_notifications",
            {"limit": 25, "offset": 5, "include_archived": True, "only_snoozed": False},
        )
    ]


@pytest.mark.asyncio
async def test_server_notifications_service_from_server_context_provider_is_lazy():
    client = FakeServerNotificationsClient()
    provider = FakeClientProvider(client)
    service = ServerNotificationsService.from_server_context_provider(provider)

    assert isinstance(service, ServerNotificationsService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    feed = await service.list_feed(limit=25, offset=5, include_archived=True)

    assert feed["total"] == 1
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [
        (
            "list_notifications",
            {"limit": 25, "offset": 5, "include_archived": True, "only_snoozed": False},
        )
    ]


@pytest.mark.asyncio
async def test_server_notifications_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider(FakeServerNotificationsClient)
    service = ServerNotificationsService.from_server_context_provider(provider)

    await service.list_feed(limit=25)
    await service.list_feed(limit=10)

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [
        (
            "list_notifications",
            {"limit": 25, "offset": 0, "include_archived": False, "only_snoozed": False},
        )
    ]
    assert provider.clients[1].calls == [
        (
            "list_notifications",
            {"limit": 10, "offset": 0, "include_archived": False, "only_snoozed": False},
        )
    ]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


@pytest.mark.asyncio
async def test_server_notifications_service_from_config_returns_provider_backed_service(monkeypatch):
    provider = FakeClientProvider(FakeServerNotificationsClient())
    build_provider_calls = []

    def build_provider(app_config):
        build_provider_calls.append(app_config)
        return provider

    monkeypatch.setattr(notifications_module, "build_runtime_api_client_provider_from_config", build_provider)

    config = {"tldw_api": {"base_url": "https://example.com"}}
    service = ServerNotificationsService.from_config(config)

    assert isinstance(service, ServerNotificationsService)
    assert service.client is None
    assert service.client_provider is provider
    assert build_provider_calls == [config]
    assert provider.build_calls == 0

    feed = await service.list_feed(limit=25)

    assert feed["total"] == 1
    assert service.client is None
    assert provider.build_calls == 1


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
