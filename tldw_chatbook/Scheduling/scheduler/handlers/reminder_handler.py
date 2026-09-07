"""Handler for scheduled reminder tasks."""

from __future__ import annotations

from typing import Any, Callable

from tldw_chatbook.Notifications.notification_dispatch_service import (
    NotificationDispatchService,
)


class ReminderHandler:
    """Dispatch a reminder notification for a scheduled task."""

    def __init__(
        self,
        dispatch_service: NotificationDispatchService,
        app_getter: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            dispatch_service: Service used to persist the reminder as an
                inbox notification and attempt transient (toast) delivery.
            app_getter: Zero-arg getter for the running app, resolved fresh
                per dispatch (BriefingJobHandler's chachanotes_db_getter
                discipline): the handler is constructed before app wiring
                completes, and dispatch() only attempts transient toast
                delivery when given a live app handle.
        """
        self.dispatch_service = dispatch_service
        self.app_getter = app_getter

    async def handle(self, task: dict[str, Any]) -> None:
        """Dispatch a reminder notification.

        Args:
            task: A scheduled task row from ``reminder_tasks``.
        """
        app = self.app_getter() if self.app_getter is not None else None
        self.dispatch_service.dispatch(
            app=app,
            category="reminder",
            title=task.get("title", "Reminder"),
            message=task.get("body") or "",
            source_entity_kind="scheduled_task",
            source_entity_id=task.get("id"),
        )

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
