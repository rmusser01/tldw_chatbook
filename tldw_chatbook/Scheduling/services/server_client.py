"""Server client for scheduling reminders."""

from __future__ import annotations

from typing import Any


class ServerUnavailableError(Exception):
    """Raised when the server client is invoked while no server is connected."""


class SchedulingServerClient:
    def __init__(self, notifications_service: Any | None = None) -> None:
        self.notifications_service = notifications_service

    def _is_available(self) -> bool:
        return self.notifications_service is not None

    async def create_reminder(self, **payload: Any) -> dict[str, Any]:
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.create_reminder(**payload)

    async def update_reminder(self, task_id: str, **payload: Any) -> dict[str, Any]:
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.update_reminder(task_id, **payload)

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.delete_reminder(task_id)

    async def list_reminders(self) -> dict[str, Any]:
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.list_reminders()

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.get_reminder(task_id)
