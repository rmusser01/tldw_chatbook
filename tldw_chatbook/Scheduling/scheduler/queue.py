"""In-memory priority queue for scheduled tasks.

For this phase the queue is a simple sorted list that can be rebuilt from the
underlying database on startup. A proper heap-backed implementation can be
introduced later when sub-second dispatch latency is required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


class PriorityQueue:
    """Simple in-memory task queue ordered by ``next_run_at``."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self._items: list[dict[str, Any]] = []

    def load(self, now: Optional[datetime] = None) -> None:
        """Rebuild the queue from the database.

        If ``now`` is provided, only tasks scheduled at or before that time are
        loaded; otherwise all enabled tasks with a ``next_run_at`` value are
        loaded.
        """
        if now is None:
            self._items = self.db.list_reminder_tasks(enabled=True)
            # Filter out tasks without a next run time and sort by it.
            self._items = [
                item for item in self._items if item.get("next_run_at")
            ]
        else:
            self._items = self.db.reminders_due_before(now)

        self._items.sort(key=lambda item: item["next_run_at"])

    def push(self, item: dict[str, Any]) -> None:
        """Insert a task, keeping the list sorted by ``next_run_at``."""
        self._items.append(item)
        self._items.sort(key=lambda item: item["next_run_at"])

    def peek(self) -> Optional[dict[str, Any]]:
        """Return the next task without removing it."""
        return self._items[0] if self._items else None

    def pop_due(self, now: datetime) -> list[dict[str, Any]]:
        """Remove and return all tasks scheduled at or before ``now``."""
        due: list[dict[str, Any]] = []
        while self._items and self._items[0].get("next_run_at", "") <= now.isoformat():
            due.append(self._items.pop(0))
        return due

    def __len__(self) -> int:
        return len(self._items)
