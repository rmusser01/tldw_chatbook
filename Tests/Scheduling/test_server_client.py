import asyncio
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientConfig,
    ServerClientError,
    ServerClientNotFoundError,
    ServerClientServerError,
    ServerClientTimeoutError,
    ServerClientValidationError,
    ServerUnavailableError,
)


class FakeNotificationsService:
    def __init__(self):
        self.calls = []

    async def create_reminder(self, **payload):
        self.calls.append(("create_reminder", payload))
        return {"id": "task-1", "title": payload.get("title")}

    async def update_reminder(self, task_id, **payload):
        self.calls.append(("update_reminder", task_id, payload))
        return {"id": task_id, **payload}

    async def delete_reminder(self, task_id):
        self.calls.append(("delete_reminder", task_id))
        return {"deleted": True}

    async def list_reminders(self):
        self.calls.append(("list_reminders",))
        return {"items": [{"id": "task-1"}], "total": 1}

    async def get_reminder(self, task_id):
        self.calls.append(("get_reminder", task_id))
        return {"id": task_id}


@pytest.fixture
def service():
    return FakeNotificationsService()


@pytest.fixture
def client(service):
    return SchedulingServerClient(notifications_service=service)


@pytest.mark.asyncio
async def test_create_reminder_delegates_to_notifications_service(client, service):
    result = await client.create_reminder(
        title="Test", schedule_kind="one_time", run_at="2026-04-24T12:00:00Z"
    )

    assert result == {"id": "task-1", "title": "Test"}
    assert service.calls == [
        (
            "create_reminder",
            {
                "title": "Test",
                "schedule_kind": "one_time",
                "run_at": "2026-04-24T12:00:00Z",
            },
        )
    ]


@pytest.mark.asyncio
async def test_update_reminder_delegates_to_notifications_service(client, service):
    result = await client.update_reminder("task-1", title="Updated")

    assert result == {"id": "task-1", "title": "Updated"}
    assert service.calls == [("update_reminder", "task-1", {"title": "Updated"})]


@pytest.mark.asyncio
async def test_delete_reminder_delegates_to_notifications_service(client, service):
    result = await client.delete_reminder("task-1")

    assert result == {"deleted": True}
    assert service.calls == [("delete_reminder", "task-1")]


@pytest.mark.asyncio
async def test_list_reminders_delegates_to_notifications_service(client, service):
    result = await client.list_reminders()

    assert result == {"items": [{"id": "task-1"}], "total": 1}
    assert service.calls == [("list_reminders",)]


@pytest.mark.asyncio
async def test_get_reminder_delegates_to_notifications_service(client, service):
    result = await client.get_reminder("task-1")

    assert result == {"id": "task-1"}
    assert service.calls == [("get_reminder", "task-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name,args,kwargs",
    [
        ("create_reminder", [], {"title": "Test", "schedule_kind": "one_time"}),
        ("update_reminder", ["task-1"], {"title": "Updated"}),
        ("delete_reminder", ["task-1"], {}),
        ("list_reminders", [], {}),
        ("get_reminder", ["task-1"], {}),
    ],
)
async def test_unavailable_client_raises_server_unavailable_error(
    method_name, args, kwargs
):
    client = SchedulingServerClient(notifications_service=None)

    method = getattr(client, method_name)
    with pytest.raises(ServerUnavailableError, match="server not available"):
        await method(*args, **kwargs)


def test_exception_hierarchy():
    assert issubclass(ServerClientNotFoundError, ServerClientError)
    assert issubclass(ServerClientServerError, ServerClientError)
    assert issubclass(ServerClientTimeoutError, ServerClientError)
    assert issubclass(ServerClientValidationError, ServerClientError)
    assert issubclass(ServerUnavailableError, ServerClientError)


def test_server_client_config_defaults():
    cfg = ServerClientConfig()
    assert cfg.timeout == 10.0
    assert cfg.max_retries == 3
    assert cfg.retry_delay == 1.0


@pytest.mark.asyncio
async def test_create_reminder_not_retried_on_timeout():
    service = AsyncMock()
    service.create_reminder.side_effect = asyncio.TimeoutError
    client = SchedulingServerClient(notifications_service=service)

    with pytest.raises(ServerClientTimeoutError):
        await client.create_reminder(title="T", schedule_kind="one_time")

    assert service.create_reminder.call_count == 1


@pytest.mark.asyncio
async def test_update_reminder_retries_then_raises_server_error():
    service = AsyncMock()
    service.update_reminder.side_effect = ServerClientServerError("boom")
    client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(max_retries=2, retry_delay=0.0),
    )

    with pytest.raises(ServerClientServerError):
        await client.update_reminder("srv-1", title="T")

    assert service.update_reminder.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_update_reminder_not_retried_on_validation_error():
    service = AsyncMock()
    service.update_reminder.side_effect = ServerClientValidationError("bad")
    client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(max_retries=2, retry_delay=0.0),
    )

    with pytest.raises(ServerClientValidationError):
        await client.update_reminder("srv-1", title="T")

    assert service.update_reminder.call_count == 1


@pytest.mark.asyncio
async def test_create_reminder_strips_idempotency_key():
    service = AsyncMock()
    service.create_reminder.return_value = {"id": "srv-1", "title": "T"}
    client = SchedulingServerClient(notifications_service=service)

    await client.create_reminder(
        title="T", schedule_kind="one_time", idempotency_key="abc-123"
    )

    _, kwargs = service.create_reminder.call_args
    assert "idempotency_key" not in kwargs


@pytest.mark.asyncio
async def test_policy_denied_maps_to_validation_error():
    service = AsyncMock()
    service.create_reminder.side_effect = PolicyDeniedError(
        action_id="x",
        reason_code="denied",
        user_message="no",
        effective_source="local",
        authority_owner="local",
    )
    client = SchedulingServerClient(notifications_service=service)

    with pytest.raises(ServerClientValidationError):
        await client.create_reminder(title="T", schedule_kind="one_time")


@pytest.mark.asyncio
async def test_list_reminders_uses_shorter_timeout():
    async def slow(*args, **kwargs):
        await asyncio.sleep(0.5)
        return {"items": []}

    service = AsyncMock()
    service.list_reminders.side_effect = slow
    client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(timeout=0.05),
    )

    with pytest.raises(ServerClientTimeoutError):
        await client.list_reminders()


@pytest.mark.asyncio
async def test_create_reminder_uses_longer_timeout():
    async def slow(*args, **kwargs):
        await asyncio.sleep(0.5)
        return {"id": "srv-1"}

    service = AsyncMock()
    service.create_reminder.side_effect = slow
    client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(timeout=0.05),
    )

    with pytest.raises(ServerClientTimeoutError):
        await client.create_reminder(title="T", schedule_kind="one_time")


@pytest.mark.asyncio
async def test_set_notifications_service_replaces_service():
    old_service = AsyncMock()
    new_service = AsyncMock()
    new_service.list_reminders.return_value = {"items": []}
    client = SchedulingServerClient(notifications_service=old_service)
    client.set_notifications_service(new_service)

    await client.list_reminders()
    assert new_service.list_reminders.called
    assert not old_service.list_reminders.called
