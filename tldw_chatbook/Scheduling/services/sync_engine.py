"""Sync engine for reconciling local scheduled tasks with the server."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import SchedulingServerClient


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SyncEngine:
    """Pull, push, and reconcile scheduled-task state with tldw_server.

    The engine operates on a single owner at a time. ``_pull`` fetches server
    reminders and updates the local cache, creating new local records (and
    their sync mappings) when they do not yet exist.
    """

    def __init__(
        self,
        db: ScheduledTasksDB,
        server_client: SchedulingServerClient | None,
        owner_id: str,
    ) -> None:
        self.db = db
        self.server_client = server_client
        self.owner_id = owner_id

    async def _pull(self) -> None:
        """Pull server reminders for the current owner and insert/update local cache."""
        if self.server_client is None:
            return

        response = await self.server_client.list_reminders()
        items = response.get("items", [])

        for server_item in items:
            server_id = server_item.get("id")
            if not server_id:
                continue

            normalized = self._normalize_reminder(server_item)
            mapping = self.db.get_sync_mapping_by_server_id(
                server_id, "reminder_task", self.owner_id
            )

            if mapping:
                local_id = mapping["local_id"]
                self.db.update_reminder_task(local_id, **normalized)
            else:
                local_id = self.db.create_reminder_task(
                    owner_id=self.owner_id,
                    server_id=server_id,
                    **normalized,
                )
                self.db.set_sync_mapping(
                    local_id, server_id, "reminder_task", self.owner_id
                )

        self.db.update_sync_state(self.owner_id, last_pull_at=now_utc_iso())

    def _normalize_reminder(self, server_item: dict[str, Any]) -> dict[str, Any]:
        """Convert server reminder fields to local DB column names.

        Known fields are mapped directly; unknown fields are dropped. Required
        local fields that are missing from the server payload receive sensible
        defaults so that partially-populated server records can still be cached.
        """
        field_map = {
            "title": "title",
            "body": "body",
            "schedule_kind": "schedule_kind",
            "run_at": "run_at",
            "cron": "cron",
            "timezone": "timezone",
            "enabled": "enabled",
            "last_run_at": "last_run_at",
            "next_run_at": "next_run_at",
            "last_status": "last_status",
            "link_type": "link_type",
            "link_id": "link_id",
            "link_url": "link_url",
            "created_at": "created_at",
            "updated_at": "updated_at",
        }

        result: dict[str, Any] = {}
        for server_key, local_key in field_map.items():
            if server_key in server_item:
                result[local_key] = server_item[server_key]

        # Provide defaults for required local fields when the server omits them.
        if "schedule_kind" not in result:
            result["schedule_kind"] = "one_time"

        return result
