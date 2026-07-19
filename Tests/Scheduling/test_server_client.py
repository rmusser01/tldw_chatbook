import pytest

from tldw_chatbook.Scheduling.services import (
    SchedulingServerClient,
    ServerUnavailableError,
)
from tldw_chatbook.Scheduling.services.server_client import (
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
