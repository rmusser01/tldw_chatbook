"""Handler for scheduled reminder tasks."""

from __future__ import annotations

from typing import Any


class ReminderHandler:
    """Dispatch a reminder notification for a scheduled task."""

    def __init__(self, dispatch_service: Any) -> None:
        self.dispatch_service = dispatch_service

    async def handle(self, task: dict) -> None:
        """Dispatch a reminder notification.

        Args:
            task: A scheduled task row from ``reminder_tasks``.
        """
        self.dispatch_service.dispatch(
            category="reminder",
            title=task.get("title", "Reminder"),
            message=task.get("body") or "",
            source_entity_kind="scheduled_task",
            source_entity_id=task.get("id"),
        )
