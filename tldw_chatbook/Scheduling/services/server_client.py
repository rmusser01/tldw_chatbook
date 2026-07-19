"""Server client for scheduling reminders."""

from __future__ import annotations

from typing import Any


class ServerUnavailableError(Exception):
    """Raised when the server client is invoked while no server is connected."""


class SchedulingServerClient:
    """Async client that delegates scheduling operations to a notifications service.

    The client is a thin wrapper around an injected notifications service. All
    methods raise :class:`ServerUnavailableError` when no service has been
    configured, so callers can distinguish "server missing" from actual request
    failures.
    """

    def __init__(self, notifications_service: Any | None = None) -> None:
        """Initialize the client.

        Args:
            notifications_service: Service that implements the reminder CRUD
                contract, or ``None`` if no scheduling server is connected.
        """
        self.notifications_service = notifications_service

    def _is_available(self) -> bool:
        """Return whether a notifications service is available."""
        return self.notifications_service is not None

    async def create_reminder(self, **payload: Any) -> dict[str, Any]:
        """Create a new reminder.

        Args:
            **payload: Reminder fields to pass to the notifications service.

        Returns:
            The created reminder as returned by the service.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
        """
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")
        return await service.create_reminder(**payload)

    async def update_reminder(self, task_id: str, **payload: Any) -> dict[str, Any]:
        """Update an existing reminder.

        Args:
            task_id: Identifier of the reminder to update.
            **payload: Reminder fields to update.

        Returns:
            The updated reminder as returned by the service.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
        """
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")
        return await service.update_reminder(task_id, **payload)

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        """Delete a reminder.

        Args:
            task_id: Identifier of the reminder to delete.

        Returns:
            The service response after deletion.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
        """
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")
        return await service.delete_reminder(task_id)

    async def list_reminders(self) -> dict[str, Any]:
        """List all reminders.

        Returns:
            The service response containing the reminder list.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
        """
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")
        return await service.list_reminders()

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        """Fetch a single reminder.

        Args:
            task_id: Identifier of the reminder to retrieve.

        Returns:
            The requested reminder as returned by the service.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
        """
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")
        return await service.get_reminder(task_id)
