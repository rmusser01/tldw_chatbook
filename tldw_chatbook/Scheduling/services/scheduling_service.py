"""Local-first facade for scheduled task operations.

The ``SchedulingService`` is the single entry point used by the UI. It routes
reads and writes to the local ``ScheduledTasksDB`` cache, and prefers the server
API when a ``SchedulingServerClient`` is available and the current owner is a
server identity (``server:<user_id>``).
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerUnavailableError,
)
from tldw_chatbook.Scheduling.services.sync_engine import SyncEngine

_REMINDER_PRIMITIVE = "reminder_task"


class SchedulingService:
    """Facade for scheduling CRUD and sync operations.

    Args:
        db: The local scheduled-tasks database.
        server_client: Optional wrapper around the server reminder API.
        runtime_source: Initial owner identity; ``"local"`` or ``"server:<user_id>"``.
    """

    def __init__(
        self,
        db: ScheduledTasksDB,
        server_client: SchedulingServerClient | None = None,
        runtime_source: str = "local",
    ) -> None:
        self.db = db
        self.server_client = server_client
        self.runtime_source = runtime_source
        self.owner_id = runtime_source
        self.sync_engine = SyncEngine(db, server_client, self.owner_id)

    def set_owner(self, owner_id: str) -> None:
        """Switch the active owner and propagate it to the sync engine."""
        self.owner_id = owner_id
        self.sync_engine.owner_id = owner_id

    async def create_reminder(self, payload: dict[str, Any]) -> ReminderTask:
        """Create a reminder, preferring the server API when connected.

        If the server is unreachable, the reminder is stored locally and a
        pending mutation is recorded so the sync engine can push it later.
        """
        if self._use_server():
            try:
                response = await self.server_client.create_reminder(**payload)
                return await self._persist_server_reminder_response(response)
            except ServerUnavailableError:
                pass

        task_id = self.db.create_reminder_task(owner_id=self.owner_id, **payload)
        if self._use_server():
            self.db.record_pending_mutation(
                task_id,
                _REMINDER_PRIMITIVE,
                self.owner_id,
                {"action": "create", "fields": dict(payload)},
            )

        row = self.db.get_reminder_task(task_id)
        return self._row_to_reminder(row)

    async def list_reminders(self) -> list[ReminderTask]:
        """Return reminders for the current owner from the local cache."""
        rows = self.db.list_reminder_tasks(owner_id=self.owner_id)
        return [self._row_to_reminder(row) for row in rows]

    async def get_reminder(self, task_id: str) -> ReminderTask | None:
        """Fetch a single reminder by local id."""
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return None
        return self._row_to_reminder(row)

    async def update_reminder(
        self, task_id: str, payload: dict[str, Any]
    ) -> ReminderTask | None:
        """Update a reminder, preferring the server API when connected.

        Falls back to a local update plus a pending mutation if the server is
        unavailable.
        """
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return None

        if self._use_server():
            server_id = row.get("server_id")
            try:
                if server_id:
                    response = await self.server_client.update_reminder(
                        server_id, **payload
                    )
                else:
                    response = await self.server_client.create_reminder(**payload)
                return await self._persist_server_reminder_response(
                    response, local_id=task_id
                )
            except ServerUnavailableError:
                pass

        self.db.update_reminder_task(task_id, **payload)
        if self._use_server():
            self.db.record_pending_mutation(
                task_id,
                _REMINDER_PRIMITIVE,
                self.owner_id,
                {"action": "update", "fields": dict(payload)},
            )

        row = self.db.get_reminder_task(task_id)
        return self._row_to_reminder(row)

    async def delete_reminder(self, task_id: str) -> bool:
        """Delete a reminder locally and on the server when connected.

        If the server is unavailable, a tombstone is recorded so the delete can
        be pushed later.
        """
        row = self.db.get_reminder_task(task_id)
        if row is None:
            return False

        if self._use_server():
            server_id = row.get("server_id")
            try:
                if server_id:
                    await self.server_client.delete_reminder(server_id)
                self.db.delete_reminder_task(task_id)
                self.db.delete_sync_mapping(
                    task_id, _REMINDER_PRIMITIVE, self.owner_id
                )
                self.db.delete_pending_mutation_for_record(
                    task_id, _REMINDER_PRIMITIVE, self.owner_id
                )
                return True
            except ServerUnavailableError:
                self.db.record_tombstone(
                    task_id, _REMINDER_PRIMITIVE, self.owner_id
                )
                self.db.delete_reminder_task(task_id)
                self.db.delete_pending_mutation_for_record(
                    task_id, _REMINDER_PRIMITIVE, self.owner_id
                )
                return True

        return self.db.delete_reminder_task(task_id)

    async def sync_now(self) -> None:
        """Trigger a full sync for the current owner."""
        await self.sync_engine.sync_now()

    def _use_server(self) -> bool:
        """Return True when server operations should be attempted."""
        return self.server_client is not None and self.owner_id.startswith("server:")

    def _map_server_response_to_local(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert a server reminder response into local reminder-task fields."""
        local: dict[str, Any] = {}

        server_id = response.get("id")
        if server_id is not None:
            local["server_id"] = server_id

        for key in (
            "title",
            "body",
            "schedule_kind",
            "run_at",
            "cron",
            "timezone",
            "enabled",
            "last_status",
            "next_run_at",
            "last_run_at",
            "missed_at",
            "link_type",
            "link_id",
            "link_url",
            "created_at",
            "updated_at",
        ):
            if key in response:
                local[key] = response[key]

        return local

    async def _persist_server_reminder_response(
        self,
        response: dict[str, Any],
        local_id: str | None = None,
    ) -> ReminderTask:
        """Insert or update the local cache from a server reminder response."""
        local_fields = self._map_server_response_to_local(response)
        server_id = response.get("id")

        if local_id is not None:
            self.db.update_reminder_task(local_id, **local_fields)
            task_id = local_id
        else:
            existing = None
            if server_id:
                existing = self.db.get_reminder_task_by_server_id(
                    self.owner_id, server_id
                )
            if existing is not None:
                task_id = existing["id"]
                self.db.update_reminder_task(task_id, **local_fields)
            else:
                task_id = self.db.create_reminder_task(
                    owner_id=self.owner_id, **local_fields
                )

        if server_id:
            self.db.set_sync_mapping(
                task_id, server_id, _REMINDER_PRIMITIVE, self.owner_id
            )

        row = self.db.get_reminder_task(task_id)
        return self._row_to_reminder(row)

    @staticmethod
    def _row_to_reminder(row: dict[str, Any]) -> ReminderTask:
        """Build a ``ReminderTask`` from a DB row.

        Removes ``None`` values for fields that have Pydantic defaults so the
        defaults are applied instead of failing validation.
        """
        data = dict(row)
        if data.get("last_status") is None:
            data.pop("last_status", None)
        return ReminderTask(**data)
