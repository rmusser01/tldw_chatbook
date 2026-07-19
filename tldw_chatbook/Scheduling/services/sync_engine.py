"""Sync engine for reconciling local scheduled tasks with the server."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientError,
    ServerClientNotFoundError,
)


_REMINDER_PRIMITIVE = "reminder_task"
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
    """Pull, push, and reconcile scheduled-task state with tldw_server."""

    def __init__(
        self,
        db: ScheduledTasksDB,
        server_client: SchedulingServerClient | None,
        owner_id: str,
    ) -> None:
        self.db = db
        self.server_client = server_client
        self.owner_id = owner_id

    async def pull(self, owner_id: str | None = None) -> None:
        """Public entry point to pull server reminders for the given owner."""
        await self.sync_now(owner_id)

    async def sync_now(self, owner_id: str | None = None) -> None:
        target_owner = owner_id if owner_id is not None else self.owner_id
        if self.server_client is None:
            return

        try:
            (
                pulled_items,
                staged_outcomes,
                conflicts,
                tombstone_ids,
                pending_local_ids,
                mutations,
            ) = await self._network_phase(target_owner)
        except ServerClientError as exc:
            self._record_sync_error(str(exc), target_owner)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync network phase failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)
            return

        try:
            with self.db.transaction() as conn:
                # Ensure pulled items have safe defaults before the DB helper
                # copies fields verbatim.
                for item in pulled_items:
                    if not item.get("title"):
                        item["title"] = "Untitled reminder"

                pull_conflicts = self.db._apply_pulled_reminders(
                    conn, target_owner, pulled_items, pending_local_ids
                )
                # The DB helper records the local record only; enrich with the
                # pending mutation so conflict resolution can re-queue it.
                mutation_payloads = {
                    m["local_id"]: m.get("payload") or {} for m in mutations
                }
                for conflict in pull_conflicts:
                    conflict.setdefault("local_state", {})
                    conflict["local_state"]["pending_mutation"] = mutation_payloads.get(
                        conflict["local_id"], {}
                    )

                all_conflicts = conflicts + pull_conflicts
                for conflict in all_conflicts:
                    self.db._record_conflict_conn(
                        conn,
                        local_id=conflict["local_id"],
                        primitive=_REMINDER_PRIMITIVE,
                        owner_id=target_owner,
                        server_state=conflict["server_state"],
                        local_state=conflict["local_state"],
                    )
                for outcome in staged_outcomes:
                    local_id = outcome["local_id"]
                    server_id = outcome.get("server_id")
                    if server_id:
                        self.db._set_sync_mapping_conn(
                            conn, local_id, server_id, _REMINDER_PRIMITIVE, target_owner
                        )
                        self.db._update_reminder_task_conn(
                            conn, local_id, server_id=server_id
                        )
                    if outcome.get("delete_local"):
                        self.db._delete_reminder_task_conn(conn, local_id)
                        self.db._delete_sync_mapping_conn(
                            conn, local_id, _REMINDER_PRIMITIVE, target_owner
                        )
                mutation_ids = [o["mutation_id"] for o in staged_outcomes if o.get("mutation_id")]
                self.db._purge_pending_mutations(conn, target_owner, mutation_ids)
                for local_id in tombstone_ids:
                    self.db._delete_tombstone_conn(
                        conn, local_id, _REMINDER_PRIMITIVE, target_owner
                    )
                seen_server_ids = {
                    item["id"] for item in pulled_items if item.get("id")
                }
                self.db._detect_server_deletions_conn(
                    conn, target_owner, seen_server_ids
                )
                self.db._update_sync_state_conn(
                    conn,
                    target_owner,
                    last_pull_at=now_utc_iso(),
                    last_push_at=now_utc_iso() if staged_outcomes or tombstone_ids else None,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync transaction failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)

    async def _network_phase(
        self, owner_id: str
    ) -> tuple[list[dict], list[dict], list[dict], list[str], set[str], list[dict]]:
        """Return (pulled_items, staged_outcomes, conflicts, tombstone_ids_to_delete,
        pending_local_ids, mutations).

        On a retryable server error, the whole phase aborts and the caller records
        a single sync error. Non-retryable 404s are converted to conflicts and the
        pending mutation is staged for removal.
        """
        pulled_items: list[dict] = []
        staged_outcomes: list[dict] = []
        conflicts: list[dict] = []
        tombstone_ids_to_delete: list[str] = []

        response = await self.server_client.list_reminders()
        if not isinstance(response, dict):
            response = {}
        pulled_items = response.get("items", [])

        mutations = self.db.get_pending_mutations(owner_id, primitive=_REMINDER_PRIMITIVE)
        pending_local_ids = {m["local_id"] for m in mutations}
        for mutation in mutations:
            outcome = await self._push_mutation(mutation, owner_id)
            if outcome.get("conflict"):
                conflicts.append(outcome["conflict"])
                # The mutation that caused a 404 is staged for deletion.
                staged_outcomes.append({
                    "local_id": outcome["conflict"]["local_id"],
                    "mutation_id": mutation["id"],
                })
            else:
                staged_outcomes.append(outcome)

        tombstones = self.db.get_tombstones(owner_id, primitive=_REMINDER_PRIMITIVE)
        for tombstone in tombstones:
            outcome = await self._push_tombstone(tombstone, owner_id)
            staged_outcomes.append(outcome)
            tombstone_ids_to_delete.append(tombstone["local_id"])

        return (
            pulled_items,
            staged_outcomes,
            conflicts,
            tombstone_ids_to_delete,
            pending_local_ids,
            mutations,
        )

    async def _push_mutation(
        self, mutation: dict, owner_id: str
    ) -> dict[str, Any] | None:
        local_id = mutation["local_id"]
        payload = mutation.get("payload") or {}
        action = payload.get("action", "update")
        fields = payload.get("fields", {})
        idempotency_key = payload.get("idempotency_key")

        try:
            if action == "create":
                response = await self.server_client.create_reminder(
                    idempotency_key=idempotency_key, **fields
                )
                response = response if isinstance(response, dict) else {}
                return {
                    "local_id": local_id,
                    "server_id": response.get("id"),
                    "mutation_id": mutation["id"],
                }
            if action == "update":
                server_id = self._server_id_for_local(local_id, owner_id=owner_id)
                if server_id is None:
                    # The local task was created offline and has never been synced.
                    # Convert this update into a create so the data is not lost.
                    response = await self.server_client.create_reminder(
                        idempotency_key=idempotency_key, **fields
                    )
                    response = response if isinstance(response, dict) else {}
                    return {
                        "local_id": local_id,
                        "server_id": response.get("id"),
                        "mutation_id": mutation["id"],
                    }
                response = await self.server_client.update_reminder(
                    server_id, idempotency_key=idempotency_key, **fields
                )
                response = response if isinstance(response, dict) else {}
                return {
                    "local_id": local_id,
                    "server_id": response.get("id", server_id),
                    "mutation_id": mutation["id"],
                }
            if action == "delete":
                server_id = self._server_id_for_local(
                    local_id, owner_id=owner_id, from_mapping_only=True
                )
                if server_id is None:
                    return {"local_id": local_id, "mutation_id": mutation["id"]}
                await self.server_client.delete_reminder(server_id)
                return {
                    "local_id": local_id,
                    "mutation_id": mutation["id"],
                    "delete_local": True,
                }
            logger.warning(f"Unknown pending mutation action {action!r}")
            return {"local_id": local_id, "mutation_id": mutation["id"]}
        except ServerClientNotFoundError:
            local_row = self.db.get_reminder_task(local_id)
            return {
                "conflict": {
                    "local_id": local_id,
                    "server_state": {},
                    "local_state": {
                        "record": dict(local_row) if local_row else {},
                        "pending_mutation": payload,
                    },
                }
            }
        except ServerClientError:
            # Abort the whole push phase; caller records one sync error.
            raise

    async def _push_tombstone(
        self, tombstone: dict, owner_id: str
    ) -> dict[str, Any] | None:
        local_id = tombstone["local_id"]
        server_id = self._server_id_for_local(
            local_id, owner_id=owner_id, from_mapping_only=True
        )
        if server_id is None:
            return {"local_id": local_id, "delete_tombstone": True}
        try:
            await self.server_client.delete_reminder(server_id)
            return {"local_id": local_id, "delete_tombstone": True}
        except ServerClientNotFoundError:
            return {"local_id": local_id, "delete_tombstone": True}
        except ServerClientError:
            raise

    def resolve_conflict(self, conflict_id: str, resolution: str = "server") -> bool:
        conflict = self.db.get_conflict_by_id(conflict_id)
        if conflict is None:
            return False

        local_id = conflict["local_id"]
        owner_id = conflict["owner_id"]
        server_state = conflict.get("server_state") or {}
        local_state = conflict.get("local_state") or {}
        pending_mutation = (
            local_state.get("pending_mutation")
            if isinstance(local_state, dict)
            else None
        )

        if resolution == "server":
            if not server_state:
                self.db.delete_reminder_task(local_id)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
                self.db.delete_tombstone(local_id, _REMINDER_PRIMITIVE, owner_id)
            else:
                self.db.update_reminder_task(
                    local_id, **self._whitelist_reminder_fields(server_state)
                )
        elif resolution == "local":
            if not server_state and pending_mutation:
                self.db.update_reminder_task(local_id, server_id=None)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    pending_mutation,
                )
            elif not server_state:
                row = self.db.get_reminder_task(local_id)
                self.db.update_reminder_task(local_id, server_id=None)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, owner_id)
                fields = {
                    key: row.get(key)
                    for key in self._REMINDER_MUTABLE_FIELDS
                    if row.get(key) is not None
                }
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    {"action": "create", "fields": fields},
                )
            elif pending_mutation:
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    pending_mutation,
                )
            else:
                fields = {
                    key: value
                    for key, value in (local_state.get("record") or local_state).items()
                    if key in self._REMINDER_MUTABLE_FIELDS
                }
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    owner_id,
                    {"action": "update", "fields": fields},
                )
            self.db.increment_conflict_retry_count(conflict_id)

        self.db.resolve_conflict(conflict_id, resolution)
        return True

    def _find_local_row(self, server_id: str) -> dict[str, Any] | None:
        """Find a local reminder row by server id, using mapping or direct lookup."""
        mapping = self.db.get_sync_mapping_by_server_id(
            server_id, _REMINDER_PRIMITIVE, self.owner_id
        )
        if mapping:
            return self.db.get_reminder_task(mapping["local_id"])

        return self.db.get_reminder_task_by_server_id(self.owner_id, server_id)

    def _server_id_for_local(
        self,
        local_id: str,
        owner_id: str | None = None,
        from_mapping_only: bool = False,
    ) -> str | None:
        """Return the server id mapped to ``local_id`` if any."""
        target_owner = owner_id if owner_id is not None else self.owner_id
        if not from_mapping_only:
            row = self.db.get_reminder_task(local_id)
            if row and row.get("server_id"):
                return row["server_id"]

        mapping = self.db.get_sync_mapping_by_local_id(
            local_id, _REMINDER_PRIMITIVE, target_owner
        )
        return mapping.get("server_id") if mapping else None

    def _record_sync_error(
        self, message: str, owner_id: str | None = None
    ) -> None:
        target_owner = owner_id if owner_id is not None else self.owner_id
        self.db._append_sync_error(target_owner, message)

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
