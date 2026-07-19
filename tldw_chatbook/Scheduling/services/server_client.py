"""Server client for scheduling reminders."""

from __future__ import annotations


class ServerUnavailableError(Exception):
    """Raised when the server client is invoked while no server is connected."""


class SchedulingServerClient:
    def __init__(self, notifications_service=None, api_client=None):
        self.notifications_service = notifications_service
        self.api_client = api_client

    def _is_available(self) -> bool:
        return self.notifications_service is not None and getattr(self.notifications_service, "client", None) is not None

    async def create_reminder(self, **payload):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.create_reminder(**payload)

    async def update_reminder(self, task_id, **payload):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.update_reminder(task_id, **payload)

    async def delete_reminder(self, task_id):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.delete_reminder(task_id)

    async def list_reminders(self):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.list_reminders()

    async def get_reminder(self, task_id):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.get_reminder(task_id)
