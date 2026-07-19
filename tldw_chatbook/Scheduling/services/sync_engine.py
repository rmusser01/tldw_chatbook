"""Sync engine for reconciling local scheduled tasks with the server."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerUnavailableError,
)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: Any) -> datetime:
    """Parse an ISO-8601 string or return the epoch for missing/invalid values."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value or not isinstance(value, str):
        return _EPOCH
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return _EPOCH


class SyncEngine:
    """Pull, push, and reconcile scheduled-task state with tldw_server.

    The engine operates on a single owner at a time. ``pull`` fetches server
    reminders and updates the local cache. ``_push`` sends locally queued
    mutations to the server. ``_push_tombstones`` propagates local deletes.
    Conflicts are recorded when a newer server state collides with a local
    pending mutation; server wins by default.
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

    async def sync_now(self) -> None:
        """Orchestrate a full sync: pull, push mutations, push tombstones."""
        await self.pull()
        await self._push()
        await self._push_tombstones()

    async def pull(self) -> None:
        """Public entry point to pull server reminders for the current owner."""
        await self._pull()

    async def _pull(self) -> None:
        """Pull server reminders for the current owner and reconcile local cache."""
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
        seen_server_ids: set[str] = set()

        for server_item in items:
            server_id = server_item.get("id")
            if not server_id:
                continue

            seen_server_ids.add(server_id)
            local_row = self._find_local_row(server_id)

            if local_row is None:
                local_id = self.db.create_reminder_task(
                    owner_id=self.owner_id,
                    server_id=server_id,
                    **self._whitelist_reminder_fields(server_item),
                )
                self.db.set_sync_mapping(
                    local_id, server_id, "reminder_task", self.owner_id
                )
            else:
                await self._reconcile_record(server_item, local_row)
                self.db.set_sync_mapping(
                    local_row["id"], server_id, "reminder_task", self.owner_id
                )

        self._detect_server_deletions(seen_server_ids)
        self.db.update_sync_state(self.owner_id, last_pull_at=now_utc_iso())

    async def _reconcile_record(
        self, server_item: dict[str, Any], local_row: dict[str, Any]
    ) -> None:
        """Compare server and local state and apply server-wins reconciliation."""
        local_id = local_row["id"]
        server_updated_raw = server_item.get("updated_at")
        server_updated = parse_iso(server_updated_raw)
        local_updated = parse_iso(local_row.get("updated_at"))

        # If the server provides no updated_at, treat the server item as newer.
        server_has_timestamp = server_updated_raw is not None
        if server_has_timestamp and server_updated <= local_updated:
            return

        pending = self.db.get_pending_mutations(self.owner_id, primitive="reminder_task")
        has_pending = any(m["local_id"] == local_id for m in pending)

        if has_pending:
            self.db.record_conflict(
                local_id=local_id,
                primitive="reminder_task",
                owner_id=self.owner_id,
                server_state=dict(server_item),
                local_state=dict(local_row),
            )
            self.db.delete_pending_mutation_for_record(
                local_id, "reminder_task", self.owner_id
            )

        self.db.update_reminder_task(
            local_id, **self._whitelist_reminder_fields(server_item)
        )

    def _detect_server_deletions(self, seen_server_ids: set[str]) -> None:
        """Record conflicts for live local rows that no longer exist on the server."""
        local_rows = self.db.list_reminder_tasks(owner_id=self.owner_id)
        for local_row in local_rows:
            server_id = local_row.get("server_id")
            if not server_id or server_id in seen_server_ids:
                continue

            local_id = local_row["id"]
            tombstones = self.db.get_tombstones(self.owner_id, primitive="reminder_task")
            has_tombstone = any(t["local_id"] == local_id for t in tombstones)

            if has_tombstone:
                self.db.delete_reminder_task(local_id)
                self.db.delete_tombstone(local_id, "reminder_task", self.owner_id)
            else:
                self.db.record_conflict(
                    local_id=local_id,
                    primitive="reminder_task",
                    owner_id=self.owner_id,
                    server_state={},
                    local_state=dict(local_row),
                )

    async def _push(self) -> None:
        """Push pending local mutations to the server for the current owner."""
        if self.server_client is None:
            return

        mutations = self.db.get_pending_mutations(self.owner_id, primitive="reminder_task")
        pushed_any = False

        for mutation in mutations:
            local_id = mutation["local_id"]
            payload = mutation.get("payload") or {}
            action = payload.get("action", "update")
            idempotency_key = payload.get("idempotency_key") or str(uuid.uuid4())

            try:
                if action == "create":
                    response = await self.server_client.create_reminder(
                        idempotency_key=idempotency_key,
                        **payload.get("fields", {}),
                    )
                elif action == "update":
                    server_id = self._server_id_for_local(local_id)
                    if server_id is None:
                        logger.warning(
                            f"Cannot push update for {local_id}: no server id mapping"
                        )
                        continue
                    response = await self.server_client.update_reminder(
                        server_id,
                        idempotency_key=idempotency_key,
                        **payload.get("fields", {}),
                    )
                elif action == "delete":
                    server_id = self._server_id_for_local(local_id)
                    if server_id is None:
                        logger.warning(
                            f"Cannot push delete for {local_id}: no server id mapping"
                        )
                        continue
                    response = await self.server_client.delete_reminder(server_id)
                else:
                    logger.warning(f"Unknown pending mutation action {action!r}")
                    continue

                self._update_local_from_push_response(local_id, response)
                self.db.delete_pending_mutation(mutation["id"])
                pushed_any = True
            except ServerUnavailableError as exc:
                logger.warning(
                    f"Server unavailable during push for {self.owner_id}: {exc}"
                )
                self._record_sync_error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Push failed for mutation {mutation['id']}: {exc}")
                self._record_sync_error(str(exc))
                return

        if pushed_any:
            self.db.update_sync_state(self.owner_id, last_push_at=now_utc_iso())

    async def _push_tombstones(self) -> None:
        """Push local delete tombstones to the server for the current owner."""
        if self.server_client is None:
            return

        tombstones = self.db.get_tombstones(self.owner_id, primitive="reminder_task")
        pushed_any = False

        for tombstone in tombstones:
            local_id = tombstone["local_id"]
            server_id = self._server_id_for_local(local_id, from_mapping_only=True)

            if server_id is None:
                # Local-only record; no server copy to delete.
                self.db.delete_tombstone(local_id, "reminder_task", self.owner_id)
                continue

            try:
                await self.server_client.delete_reminder(server_id)
                self.db.delete_tombstone(local_id, "reminder_task", self.owner_id)
                pushed_any = True
            except ServerUnavailableError as exc:
                logger.warning(
                    f"Server unavailable during tombstone push for {self.owner_id}: {exc}"
                )
                self._record_sync_error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Tombstone push failed for {local_id}: {exc}")
                self._record_sync_error(str(exc))
                return

        if pushed_any:
            self.db.update_sync_state(self.owner_id, last_push_at=now_utc_iso())

    def resolve_conflict(self, conflict_id: str, resolution: str = "server") -> bool:
        """Resolve a recorded conflict.

        ``resolution`` is either ``server`` (default, server wins) or ``local``
        (re-queue the local change for another push attempt).
        """
        conflict = self.db.get_conflict_by_id(conflict_id)
        if conflict is None:
            return False

        local_id = conflict["local_id"]

        if resolution == "local":
            local_state = conflict.get("local_state") or {}
            fields = {
                key: value
                for key, value in local_state.items()
                if key in self._REMINDER_MUTABLE_FIELDS
            }
            self.db.record_pending_mutation(
                local_id=local_id,
                primitive="reminder_task",
                owner_id=self.owner_id,
                payload={"action": "update", "fields": fields},
            )
            self.db.increment_conflict_retry_count(conflict_id)

        self.db.resolve_conflict(conflict_id, resolution)
        return True

    def _find_local_row(self, server_id: str) -> dict[str, Any] | None:
        """Find a local reminder row by server id, using mapping or direct lookup."""
        mapping = self.db.get_sync_mapping_by_server_id(
            server_id, "reminder_task", self.owner_id
        )
        if mapping:
            return self.db.get_reminder_task(mapping["local_id"])

        return self.db.get_reminder_task_by_server_id(self.owner_id, server_id)

    def _server_id_for_local(
        self, local_id: str, from_mapping_only: bool = False
    ) -> str | None:
        """Return the server id mapped to ``local_id`` if any."""
        row = self.db.get_reminder_task(local_id)
        if row:
            server_id = row.get("server_id")
            if server_id:
                return server_id

        if from_mapping_only:
            mapping = self.db.get_sync_mapping_by_local_id(
                local_id, "reminder_task", self.owner_id
            )
            return mapping.get("server_id") if mapping else None

        mapping = self.db.get_sync_mapping_by_local_id(
            local_id, "reminder_task", self.owner_id
        )
        if mapping:
            return mapping.get("server_id")

        return None

    def _update_local_from_push_response(
        self, local_id: str, response: dict[str, Any]
    ) -> None:
        """Update the local cache from a successful server push response."""
        server_id = response.get("id")
        if server_id:
            self.db.set_sync_mapping(
                local_id, server_id, "reminder_task", self.owner_id
            )
            self.db.update_reminder_task(
                local_id,
                server_id=server_id,
                **self._whitelist_reminder_fields(response),
            )
        else:
            self.db.update_reminder_task(
                local_id, **self._whitelist_reminder_fields(response)
            )

    def _record_sync_error(self, message: str) -> None:
        """Store a sync error for the current owner."""
        self.db.update_sync_state(
            self.owner_id,
            sync_errors=[{"message": message, "timestamp": now_utc_iso()}],
        )

    _REMINDER_MUTABLE_FIELDS = {
        "title",
        "body",
        "schedule_kind",
        "run_at",
        "cron",
        "timezone",
        "enabled",
        "next_run_at",
        "last_run_at",
        "last_status",
        "link_type",
        "link_id",
        "link_url",
    }

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
