"""In-memory priority queue for scheduled tasks.

For this phase the queue is a simple sorted list that can be rebuilt from the
underlying database on startup. A proper heap-backed implementation can be
introduced later when sub-second dispatch latency is required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection

_FUTURE_SORT_KEY = "9999-12-31T23:59:59+00:00"
_DEFAULT_OWNER_ID = "local"


class PriorityQueue:
    """Simple in-memory task queue ordered by ``next_run_at``."""

    def __init__(
        self,
        db: Any,
        watchlist_projection: WatchlistProjection | None = None,
        briefing_projection: BriefingProjection | None = None,
    ) -> None:
        self.db = db
        self.watchlist_projection = watchlist_projection
        self.briefing_projection = briefing_projection
        self._items: list[dict[str, Any]] = []

    @staticmethod
    def _sort_key(item: dict[str, Any]) -> str:
        """Return the sort key for a task, missing run times sort last."""
        return item.get("next_run_at") or _FUTURE_SORT_KEY

    def load(self, now: Optional[datetime] = None) -> None:
        """Rebuild the queue from the database.

        When called without arguments, loads all future-enabled reminder tasks
        that have a ``next_run_at`` value, appends any projected watchlist and
        briefing jobs that have a ``next_run_at``, and sorts the combined list
        by ``next_run_at``.

        The ``now`` parameter is retained for back-compat and tests: when
        provided, only reminder tasks scheduled at or before ``now`` are loaded;
        projections are still appended unconditionally.
        """
        if now is None:
            self._items = self.db.list_reminder_tasks(enabled=True)
            # Filter out tasks without a next run time and sort by it.
            self._items = [item for item in self._items if item.get("next_run_at")]
        else:
            self._items = self.db.reminders_due_before(now)

        self._append_projected(self.watchlist_projection)
        self._append_projected(self.briefing_projection)

        self._items.sort(key=self._sort_key)

    def _append_projected(self, projection: Any) -> None:
        """Append one projection's due-having jobs to ``self._items``.

        Shared by ``watchlist_projection`` and ``briefing_projection``
        (generalized minimally -- a second named parameter, not a
        projections *list* -- see ``PriorityQueue.__init__``): both are
        optional, both emit ``ScheduledTask`` objects, and both are
        appended the same way, so the loop body lived in exactly one place
        rather than twice.
        """
        if projection is None:
            return
        for task in projection.list_jobs(owner_id=_DEFAULT_OWNER_ID):
            item = task.model_dump(mode="json")
            if item.get("next_run_at"):
                self._items.append(item)

    def reload(self) -> None:
        """Explicitly rebuild the queue from the database."""
        self.load()

    def push(self, item: dict[str, Any]) -> None:
        """Insert a task, keeping the list sorted by ``next_run_at``."""
        self._items.append(item)
        self._items.sort(key=self._sort_key)

    def peek(self) -> Optional[dict[str, Any]]:
        """Return the next task without removing it."""
        return self._items[0] if self._items else None

    def pop_due(self, now: datetime) -> list[dict[str, Any]]:
        """Remove and return all tasks scheduled at or before ``now``."""
        due: list[dict[str, Any]] = []
        while self._items:
            run_at_str = self._items[0].get("next_run_at")
            if not run_at_str:
                # Tasks without a run time cannot be dispatched; discard them.
                self._items.pop(0)
                continue

            run_at = datetime.fromisoformat(run_at_str)
            if run_at <= now:
                due.append(self._items.pop(0))
            else:
                break

        return due

    def __len__(self) -> int:
        return len(self._items)
