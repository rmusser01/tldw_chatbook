"""Sync engine for reconciling local scheduled tasks with the server."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerUnavailableError,
)


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

    async def pull(self) -> None:
        """Public entry point to pull server reminders for the current owner."""
        await self._pull()

    async def _pull(self) -> None:
        """Pull server reminders for the current owner and insert/update local cache."""
        if self.server_client is None:
            return

        try:
            response = await self.server_client.list_reminders()
        except ServerUnavailableError as exc:
            logger.warning(
                f"Server unavailable during sync pull for {self.owner_id}: {exc}"
            )
            self._record_sync_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - sync must never crash the app
            logger.exception(f"Sync pull failed for {self.owner_id}: {exc}")
            self._record_sync_error(str(exc))
            return

        items = response.get("items", [])

        for server_item in items:
            server_id = server_item.get("id")
            if not server_id:
                continue

            normalized = self._whitelist_reminder_fields(server_item)
            mapping = self.db.get_sync_mapping_by_server_id(
                server_id, "reminder_task", self.owner_id
            )

            if mapping:
                local_id = mapping["local_id"]
                self.db.update_reminder_task(local_id, **normalized)
            else:
                existing = self.db.get_reminder_task_by_server_id(
                    self.owner_id, server_id
                )
                if existing:
                    local_id = existing["id"]
                    self.db.update_reminder_task(local_id, **normalized)
                    self.db.set_sync_mapping(
                        local_id, server_id, "reminder_task", self.owner_id
                    )
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

    def _record_sync_error(self, message: str) -> None:
        """Store a sync error for the current owner."""
        self.db.update_sync_state(
            self.owner_id,
            sync_errors=[{"message": message, "timestamp": now_utc_iso()}],
        )

    def _whitelist_reminder_fields(self, server_item: dict[str, Any]) -> dict[str, Any]:
        """Return a dict of local reminder-task fields from a server payload.

        Only known fields are copied; unknown fields are dropped. Missing
        required local fields receive safe defaults so that partially-populated
        server records can still be cached.
        """
        local_fields = (
            "title",
            "body",
            "schedule_kind",
            "run_at",
            "cron",
            "timezone",
            "enabled",
            "last_run_at",
            "next_run_at",
            "last_status",
            "link_type",
            "link_id",
            "link_url",
            "created_at",
            "updated_at",
        )

        result: dict[str, Any] = {
            local_key: server_item[local_key]
            for local_key in local_fields
            if local_key in server_item
        }

        if not result.get("title"):
            result["title"] = "Untitled reminder"
        if "schedule_kind" not in result:
            result["schedule_kind"] = "one_time"

        return result
