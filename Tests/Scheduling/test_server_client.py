import pytest

from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerUnavailableError,
)


class FakeNotificationsService:
    def __init__(self):
        self.client = object()
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


@pytest.mark.asyncio
async def test_create_reminder_delegates_to_notifications_service():
    service = FakeNotificationsService()
    client = SchedulingServerClient(notifications_service=service)

    result = await client.create_reminder(title="Test", schedule_kind="one_time", run_at="2026-04-24T12:00:00Z")

    assert result == {"id": "task-1", "title": "Test"}
    assert service.calls == [
        (
            "create_reminder",
            {"title": "Test", "schedule_kind": "one_time", "run_at": "2026-04-24T12:00:00Z"},
        )
    ]


@pytest.mark.asyncio
async def test_unavailable_client_raises_server_unavailable_error():
    client = SchedulingServerClient(notifications_service=None)

    with pytest.raises(ServerUnavailableError, match="server not available"):
        await client.create_reminder(title="Test", schedule_kind="one_time")
