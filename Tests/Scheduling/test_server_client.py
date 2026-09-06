import pytest

from tldw_chatbook.Scheduling.services import (
    SchedulingServerClient,
    ServerClientConfig,
    ServerClientNotFoundError,
    ServerClientServerError,
    ServerUnavailableError,
)


class FakeNotificationsService:
    def __init__(self):
        self.calls = []
        # task-3 (schedules UAT remediation ruling 5) capabilities
        # handshake -- overridden per test below; defaults to "present".
        self.capabilities_response = {"items": []}
        self.capabilities_error = None

    async def get_scheduled_automation_capabilities(self):
        self.calls.append(("get_scheduled_automation_capabilities",))
        if self.capabilities_error is not None:
            raise self.capabilities_error
        return self.capabilities_response

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
    result = await client.create_reminder(title="Test", schedule_kind="one_time", run_at="2026-04-24T12:00:00Z")

    assert result == {"id": "task-1", "title": "Test"}
    assert service.calls == [
        (
            "create_reminder",
            {"title": "Test", "schedule_kind": "one_time", "run_at": "2026-04-24T12:00:00Z"},
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
async def test_unavailable_client_raises_server_unavailable_error(method_name, args, kwargs):
    client = SchedulingServerClient(notifications_service=None)

    method = getattr(client, method_name)
    with pytest.raises(ServerUnavailableError, match="server not available"):
        await method(*args, **kwargs)


# -- get_capabilities (task-3, schedules UAT remediation ruling 5) --------
# The handshake: probe once, cache the verdict for this connection. A 404
# (the whole route is absent -- a server old enough to predate Scheduled
# Tasks automation) degrades to `None`, never raised; anything else
# re-raises uncached, so a transient blip is never misread as "too old".


@pytest.mark.asyncio
async def test_get_capabilities_returns_response_when_present(client, service):
    result = await client.get_capabilities()

    assert result == {"items": []}
    assert service.calls == [("get_scheduled_automation_capabilities",)]


@pytest.mark.asyncio
async def test_get_capabilities_caches_a_successful_probe(client, service):
    first = await client.get_capabilities()
    second = await client.get_capabilities()

    assert first == second == {"items": []}
    assert service.calls == [("get_scheduled_automation_capabilities",)], (
        "the second call must reuse the cached verdict, not re-probe"
    )


@pytest.mark.asyncio
async def test_get_capabilities_returns_none_when_route_absent(client, service):
    service.capabilities_error = ServerClientNotFoundError("not found")

    result = await client.get_capabilities()

    assert result is None


@pytest.mark.asyncio
async def test_get_capabilities_caches_an_absent_verdict_too(client, service):
    service.capabilities_error = ServerClientNotFoundError("not found")

    first = await client.get_capabilities()
    second = await client.get_capabilities()

    assert first is None
    assert second is None
    assert service.calls == [("get_scheduled_automation_capabilities",)], (
        "an 'absent' verdict is cached exactly like a 'present' one"
    )


@pytest.mark.asyncio
async def test_get_capabilities_transient_failure_is_not_cached(service):
    # A no-retry-delay config: the "not cached" behavior under test is
    # orthogonal to `_call_with_retry`'s own (separately tested) backoff.
    fast_client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(max_retries=0, retry_delay=0),
    )
    service.capabilities_error = ServerClientServerError("503")

    with pytest.raises(ServerClientServerError):
        await fast_client.get_capabilities()
    calls_after_failure = len(service.calls)

    # The server recovers; a fresh probe must actually be attempted, not
    # answered from a (nonexistent) cached failure.
    service.capabilities_error = None
    result = await fast_client.get_capabilities()
    assert result == {"items": []}
    assert len(service.calls) == calls_after_failure + 1, (
        "the recovered probe must be a real second attempt"
    )


@pytest.mark.asyncio
async def test_set_notifications_service_clears_the_capabilities_cache(client, service):
    """A reconnect (or a switch to a different server) may answer the
    probe differently -- the cache must not survive it."""
    await client.get_capabilities()
    assert len(service.calls) == 1

    other_service = FakeNotificationsService()
    other_service.capabilities_response = {"items": [{"family": "agent_task"}]}
    client.set_notifications_service(other_service)

    result = await client.get_capabilities()
    assert result == {"items": [{"family": "agent_task"}]}
    assert other_service.calls == [("get_scheduled_automation_capabilities",)]
