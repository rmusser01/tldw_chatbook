"""Scheduled tasks database layer.

ScheduledTasksDB extends BaseDB and provides CRUD operations for reminder
tasks, plus the shared schema used by automation definitions, previews, audit
events, and sync helpers.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterator, Optional, Union, cast
from zoneinfo import ZoneInfo

from croniter import croniter
from loguru import logger

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.DB.sql_validation import validate_identifier
from tldw_chatbook.Scheduling.schedule_compute import compute_next_run_at
from tldw_chatbook.Scheduling.schedule_vocabulary import to_local_schedule

#: ``transfer_state`` values that make a row dormant: excluded from every
#: armable filter (DB-query and ``PriorityQueue.load`` layers, both
#: primitives) and refused by every run-now seam. Per
#: spec-2026-08-31-schedules-handoff-parity.md §6.1 ruling 2, a row only
#: goes dark once a push attempt actually starts (``to_server_sent``) or a
#: server-owned mirror's local release copy is queued
#: (``from_server_pending``) -- ``NULL``, ``to_server_pending`` (merely
#: queued, not yet attempted), and ``to_server_failed`` (send failed, the
#: row re-arms) all keep executing locally. This corrects the pre-PR-5 code,
#: which excluded ANY non-NULL ``transfer_state`` on the definitions side.
DORMANT_TRANSFER_STATES = ("to_server_sent", "from_server_pending")


class ScheduledTasksDB(BaseDB):
    """Database operations for scheduled tasks and reminders."""

    _CURRENT_SCHEMA_VERSION = 6

    #: Defensive cap on `list_armable_automation_definitions` -- mirrors the
    #: Automations tab's `AUTOMATIONS_LOAD_MAX_ROWS` cap-500 precedent
    #: (`UI/Screens/scheduling/schedules_workbench.py`). Not a design limit
    #: on how many local automations may exist -- see the truncation
    #: warning this method logs when the cap is hit.
    _ARMABLE_DEFINITIONS_CAP = 500

    _REMINDER_TASK_COLUMNS = {
        "id",
        "server_id",
        "owner_id",
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
        "missed_count",
        "timeout_seconds",
        "transfer_state",
        "link_type",
        "link_id",
        "link_url",
        "created_at",
        "updated_at",
        "sync_version",
    }

    _AUTOMATION_DEFINITION_COLUMNS = {
        "id",
        "server_id",
        "owner_id",
        "family",
        "name",
        "description",
        "lifecycle",
        "health",
        "schedule",
        "input",
        "config",
        "visibility_policy",
        "notification_policy",
        "approval_policy",
        "version",
        "preview_id",
        "created_by",
        "updated_by",
        "created_at",
        "updated_at",
        "archived_at",
        "disabled_lock_kind",
        "disabled_reason",
        "resolution_state",
        "resolved_at",
        "resolved_by",
        "resolved_result_id",
        "finding_policy",
        "retention_policy",
        "next_run_at",
        "transfer_state",
    }

    _AUTOMATION_AUDIT_EVENT_COLUMNS = {
        "id",
        "definition_id",
        "owner_id",
        "event_type",
        "actor",
        "summary",
        "before",
        "after",
        "request_id",
        "idempotency_key",
        "created_at",
    }

    _REMINDER_JSON_FIELDS: set[str] = set()

    _AUTOMATION_JSON_FIELDS = {
        "schedule",
        "input",
        "config",
        "visibility_policy",
        "notification_policy",
        "approval_policy",
        "finding_policy",
        "retention_policy",
    }

    _AUDIT_JSON_FIELDS = {
        "before",
        "after",
    }

    _RUNS_RETAINED_PER_DEFINITION = 200

    _AUTOMATION_RUN_COLUMNS = {
        "server_id", "status", "outcome", "schedule_slot",
        "scope_snapshot", "finding_policy_snapshot", "rag_request_snapshot",
        "run_summary", "evidence_summary", "failure_reason",
        "updated_at", "started_at", "ended_at",
    }
    _AUTOMATION_RUN_JSON_FIELDS = {
        "scope_snapshot", "finding_policy_snapshot", "rag_request_snapshot",
        "run_summary", "evidence_summary", "failure_reason",
    }

    _AUTOMATION_RESULT_COLUMNS = {
        "server_id", "answer", "answer_mode", "confidence", "source_refs",
        "visibility_destination", "review_state", "reviewed_at",
        "reviewed_by", "review_note", "updated_at",
    }
    _AUTOMATION_RESULT_JSON_FIELDS = {
        "answer", "confidence", "source_refs", "visibility_destination",
    }

    _DATETIME_FIELDS = {
        "run_at",
        "next_run_at",
        "last_run_at",
        "missed_at",
        "created_at",
        "updated_at",
        "archived_at",
        "expires_at",
        "consumed_at",
    }

    _RESERVED_FIELDS = {"id"}

    _SYNC_STATE_COLUMNS = {
        "last_pull_at",
        "last_push_at",
        "last_conflict_at",
        "sync_errors",
    }

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str = "default",
        check_integrity_on_startup: bool = False,
    ):
        super().__init__(db_path, client_id, check_integrity_on_startup)

    def _get_connection(self) -> sqlite3.Connection:
        """Open one fresh, caller-closed connection (see class usage).

        task-22224 EXCEPTION -- deliberately keeps the legacy default
        isolation level instead of ``isolation_level = None`` (the
        held-connection rule in ``Library_Ingest_Jobs_DB.py``'s module
        docstring, the store template). This store does not HOLD
        connections: every caller opens one here and closes it per
        operation (``closing(...)`` / ``transaction()``'s ``finally``), so
        an implicit transaction cannot leak across operations and nothing
        issues an explicit BEGIN outside migration scripts -- the
        degradation mechanism cannot fire. Write bodies rely on implicit
        transactions (``transaction()`` has no explicit BEGIN; migrations
        pair multi-statement spans with ``conn.commit()``), so flipping to
        autocommit here would strip their atomicity; converting means
        adding explicit BEGIN and auditing the ~22 ``transaction()`` bodies
        plus the ``Scheduling/db/migrations`` version stamps -- its own
        task. Do NOT copy this pattern into a store that holds connections.
        """
        conn = super()._get_connection()
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local reminder/automation
        # store) and avoids an fsync per commit. This DB opens a fresh
        # connection per operation (`closing(self._get_connection())` /
        # `transaction()` throughout this file), so synchronous must be
        # re-applied on every open, not just the first (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _initialize_schema(self) -> None:
        """Create tables, indexes, and schema version row, migrating forward.

        Each migration checks its own applicability on the connection it
        will migrate. For a ``:memory:`` database every
        ``_get_connection()`` is a fresh empty database -- a version check
        done on one connection tells nothing about the next -- so the chain
        must not consult ``get_schema_version()`` between migrations the
        way a file-backed database could. The migrations themselves detect
        their condition structurally (the presence of their column), which
        is memory-correct: an empty memory database runs v0_to_v1, finds
        no ``missed_count``, adds it, finds no ``timeout_seconds``, adds
        it, finds no ``automation_runs``/``automation_results`` tables,
        adds those and the v4 columns, and every step lands on a
        consistent v4 schema even though each step sees its own
        connection.
        """
        if self._schema_is_current():
            # Warm-boot fast path (ADR-097 boot ratchet): a fully-migrated
            # file DB skips importing the four migration modules entirely,
            # keeping them out of the `_ui_ready` module census on every
            # boot after the first. Only a PROOF of completeness skips --
            # any probe failure (missing table on a fresh or `:memory:`
            # per-connection DB, an older recorded version) falls through
            # to the full chain below, whose idempotence remains the
            # correctness backstop.
            return

        from tldw_chatbook.Scheduling.db.migrations.v0_to_v1 import (
            migrate as migrate_v0_to_v1,
        )
        from tldw_chatbook.Scheduling.db.migrations.v1_to_v2 import (
            migrate as migrate_v1_to_v2,
        )
        from tldw_chatbook.Scheduling.db.migrations.v2_to_v3 import (
            migrate as migrate_v2_to_v3,
        )
        from tldw_chatbook.Scheduling.db.migrations.v3_to_v4 import (
            migrate as migrate_v3_to_v4,
        )
        from tldw_chatbook.Scheduling.db.migrations.v4_to_v5 import (
            migrate as migrate_v4_to_v5,
        )
        from tldw_chatbook.Scheduling.db.migrations.v5_to_v6 import (
            migrate as migrate_v5_to_v6,
        )

        migrate_v0_to_v1(self)
        migrate_v1_to_v2(self)
        migrate_v2_to_v3(self)
        migrate_v3_to_v4(self)
        migrate_v4_to_v5(self)
        migrate_v5_to_v6(self)

    def _schema_is_current(self) -> bool:
        """Return True when the recorded version proves the chain already ran.

        A single pre-check before the migration chain, not a version
        consultation between steps -- the `:memory:` discipline in
        `_initialize_schema`'s docstring is untouched: a memory DB's fresh
        connection has no ``schema_version`` table, the probe returns
        False, and the chain runs exactly as before.
        """
        try:
            with closing(self._get_connection()) as conn:
                row = conn.execute(
                    "SELECT MAX(version) FROM schema_version"
                ).fetchone()
        except sqlite3.Error:
            return False
        return bool(
            row
            and row[0] is not None
            and int(row[0]) >= self._CURRENT_SCHEMA_VERSION
        )

    def get_schema_version(self) -> int:
        """Return the currently recorded schema version."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside a SQLite transaction.

        Commits on clean exit and rolls back on any exception.
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Connection-aware helpers
    # ------------------------------------------------------------------

    def _get_reminder_task_by_server_id_conn(
        self, conn: sqlite3.Connection, owner_id: str, server_id: str
    ) -> Optional[dict[str, Any]]:
        cursor = conn.execute(
            "SELECT * FROM reminder_tasks WHERE owner_id = ? AND server_id = ?",
            (owner_id, server_id),
        )
        return self._row_to_dict(cursor.fetchone())

    def _create_reminder_task_conn(
        self, conn: sqlite3.Connection, owner_id: str, title: str, **kwargs: Any
    ) -> str:
        self._validate_kwargs(kwargs, self._REMINDER_TASK_COLUMNS, "reminder task")
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        fields: dict[str, Any] = {
            "id": task_id,
            "owner_id": owner_id,
            "title": title,
            "created_at": self._to_utc_iso(now),
            "updated_at": self._to_utc_iso(now),
            "enabled": 1,
            "sync_version": 0,
        }
        for key, value in kwargs.items():
            if key == "enabled":
                fields[key] = 1 if value else 0
            elif key in self._DATETIME_FIELDS:
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value
        self._validate_sql_identifiers(list(fields.keys()))
        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))
        conn.execute(
            f"INSERT INTO reminder_tasks ({columns}) VALUES ({placeholders})",
            list(fields.values()),
        )
        return task_id

    def _update_reminder_task_conn(
        self, conn: sqlite3.Connection, task_id: str, **kwargs: Any
    ) -> bool:
        if not kwargs:
            return False
        self._validate_kwargs(kwargs, self._REMINDER_TASK_COLUMNS, "reminder task")
        updates: list[str] = []
        params: list[Any] = []
        for key, value in kwargs.items():
            if key == "enabled":
                updates.append("enabled = ?")
                params.append(1 if value else 0)
            elif key in self._DATETIME_FIELDS:
                updates.append(f"{key} = ?")
                params.append(self._to_utc_iso(value))
            else:
                updates.append(f"{key} = ?")
                params.append(value)
        if not updates:
            return False
        self._validate_sql_identifiers([key.split(" ", 1)[0] for key in updates])
        updates.append("updated_at = ?")
        params.append(self._to_utc_iso(datetime.now(timezone.utc)))
        params.append(task_id)
        cursor = conn.execute(
            f"UPDATE reminder_tasks SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        return cursor.rowcount > 0

    def _set_sync_mapping_conn(
        self,
        conn: sqlite3.Connection,
        local_id: str,
        server_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        conn.execute(
            """
            INSERT OR REPLACE INTO sync_mapping
            (local_id, server_id, primitive, owner_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (local_id, server_id, primitive, owner_id, self._to_utc_iso(now)),
        )

    def _delete_reminder_task_conn(
        self, conn: sqlite3.Connection, task_id: str
    ) -> bool:
        cursor = conn.execute("DELETE FROM reminder_tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0

    def _delete_sync_mapping_conn(
        self,
        conn: sqlite3.Connection,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        conn.execute(
            """
            DELETE FROM sync_mapping
            WHERE local_id = ? AND primitive = ? AND owner_id = ?
            """,
            (local_id, primitive, owner_id),
        )

    def _delete_tombstone_conn(
        self,
        conn: sqlite3.Connection,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        conn.execute(
            """
            DELETE FROM sync_tombstones
            WHERE local_id = ? AND primitive = ? AND owner_id = ?
            """,
            (local_id, primitive, owner_id),
        )

    def _detect_server_deletions_conn(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        seen_server_ids: set[str],
    ) -> None:
        """Record conflicts for local rows whose server id is no longer returned.

        Rows with a local tombstone are deleted instead of becoming conflicts.
        Must run inside an existing transaction.
        """
        cursor = conn.execute(
            "SELECT * FROM reminder_tasks WHERE owner_id = ? AND server_id IS NOT NULL",
            (owner_id,),
        )
        for local_row in self._rows_to_dicts(cursor.fetchall()):
            server_id = local_row.get("server_id")
            if not server_id or server_id in seen_server_ids:
                continue

            existing_conflict = conn.execute(
                """
                SELECT 1 FROM sync_conflicts
                WHERE local_id = ? AND primitive = ? AND owner_id = ? AND resolved_at IS NULL
                """,
                (local_row["id"], "reminder_task", owner_id),
            ).fetchone()
            if existing_conflict is not None:
                continue

            tombstone = conn.execute(
                """
                SELECT 1 FROM sync_tombstones
                WHERE local_id = ? AND primitive = ? AND owner_id = ?
                """,
                (local_row["id"], "reminder_task", owner_id),
            ).fetchone()

            if tombstone is not None:
                self._delete_reminder_task_conn(conn, local_row["id"])
                self._delete_sync_mapping_conn(
                    conn, local_row["id"], "reminder_task", owner_id
                )
                self._delete_tombstone_conn(
                    conn, local_row["id"], "reminder_task", owner_id
                )
            else:
                self._record_conflict_conn(
                    conn,
                    local_id=local_row["id"],
                    primitive="reminder_task",
                    owner_id=owner_id,
                    server_state={},
                    local_state={"record": dict(local_row)},
                )

    def _record_conflict_conn(
        self,
        conn: sqlite3.Connection,
        local_id: str,
        primitive: str,
        owner_id: str,
        server_state: dict[str, Any],
        local_state: dict[str, Any],
    ) -> str:
        conflict_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        conn.execute(
            """
            INSERT INTO sync_conflicts
            (id, local_id, primitive, owner_id, server_state, local_state,
             server_state_at, created_at, resolved_at, resolution, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0)
            """,
            (
                conflict_id,
                local_id,
                primitive,
                owner_id,
                self._to_json(server_state),
                self._to_json(local_state),
                self._to_utc_iso(server_state.get("updated_at") or now),
                self._to_utc_iso(now),
            ),
        )
        return conflict_id

    def _update_sync_state_conn(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        **kwargs: Any,
    ) -> None:
        if not kwargs:
            return
        self._validate_kwargs(kwargs, self._SYNC_STATE_COLUMNS, "sync state")
        fields: dict[str, Any] = {"owner_id": owner_id}
        for key, value in kwargs.items():
            if key == "sync_errors":
                fields[key] = self._to_json(value)
            elif key in self._DATETIME_FIELDS:
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value
        self._validate_sql_identifiers(list(fields.keys()))
        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))
        updates = [f"{key} = excluded.{key}" for key in fields if key != "owner_id"]
        self._validate_sql_identifiers([key.split(" ", 1)[0] for key in updates])
        conn.execute(
            f"""
            INSERT INTO sync_state ({columns}) VALUES ({placeholders})
            ON CONFLICT(owner_id) DO UPDATE SET {", ".join(updates)}
            """,
            list(fields.values()),
        )

    def _get_sync_state_conn(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch the sync state row for ``owner_id`` on an existing connection."""
        cursor = conn.execute(
            "SELECT * FROM sync_state WHERE owner_id = ?",
            (owner_id,),
        )
        return self._row_to_dict(cursor.fetchone(), json_fields={"sync_errors"})

    def _apply_pulled_reminders(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        server_items: list[dict[str, Any]],
        pending_local_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Insert or update reminder rows from a pulled server list.

        Rows with a pending local mutation become server-update conflicts instead of
        being overwritten. Returns the list of conflicts created.

        Must run inside an existing transaction (``conn`` is the open connection).
        """
        pending = pending_local_ids or set()
        conflicts: list[dict[str, Any]] = []
        for item in server_items:
            server_id = item.get("id")
            if not server_id:
                continue

            existing = self._get_reminder_task_by_server_id_conn(
                conn, owner_id, server_id
            )
            fields = {
                key: item[key]
                for key in self._REMINDER_TASK_COLUMNS
                if key in item and key not in {"id", "server_id", "owner_id"}
            }
            # §6.1 ruling 2, same as upsert_automation_definitions_from_
            # server's existing pop: transfer_state is a local-only marker
            # a server pull must never overwrite (a real server payload
            # never carries one, but nothing guarantees that forever).
            fields.pop("transfer_state", None)
            fields.setdefault("title", "Untitled reminder")
            if "schedule_kind" not in fields:
                fields["schedule_kind"] = "one_time"
            if "updated_at" not in fields:
                fields["updated_at"] = self._to_utc_iso(datetime.now(timezone.utc))

            if existing:
                local_id = existing["id"]
                if local_id in pending:
                    conflicts.append({
                        "local_id": local_id,
                        "server_state": dict(item),
                        "local_state": {"record": dict(existing)},
                    })
                    continue
                self._update_reminder_task_conn(conn, local_id, **fields)
            else:
                local_id = self._create_reminder_task_conn(
                    conn, owner_id, server_id=server_id, **fields
                )

            self._set_sync_mapping_conn(
                conn, local_id, server_id, "reminder_task", owner_id
            )
        return conflicts

    def _purge_pending_mutations(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        mutation_ids: list[int],
    ) -> None:
        """Delete pending mutations by their row ids inside an existing transaction."""
        if not mutation_ids:
            return
        placeholders = ", ".join("?" * len(mutation_ids))
        conn.execute(
            f"DELETE FROM pending_mutations WHERE id IN ({placeholders})",
            mutation_ids,
        )

    def _append_sync_error(self, owner_id: str, message: str) -> None:
        """Append a sync error, capping the history at 10 entries."""
        with self.transaction() as conn:
            state = self._get_sync_state_conn(conn, owner_id) or {}
            errors = list(state.get("sync_errors") or [])
            errors.append({"message": message, "timestamp": datetime.now(timezone.utc).isoformat()})
            errors = errors[-10:]
            self._update_sync_state_conn(conn, owner_id, sync_errors=errors)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_utc_iso(value: Any) -> Optional[str]:
        """Convert a datetime/date/string to a UTC ISO-8601 string.

        - ``datetime`` values are converted to UTC (naive datetimes are
          assumed to be UTC).
        - ``date`` values are treated as midnight UTC.
        - Strings are parsed as ISO-8601 and then converted to UTC.
        - ``None`` returns ``None``.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            dt = (
                value
                if value.tzinfo is not None
                else value.replace(tzinfo=timezone.utc)
            )
            return dt.astimezone(timezone.utc).isoformat()

        if isinstance(value, date):
            return datetime(
                value.year, value.month, value.day, tzinfo=timezone.utc
            ).isoformat()

        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid ISO-8601 datetime string: {value!r}"
                ) from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()

        raise TypeError(f"Expected datetime, date, or str, got {type(value).__name__}")

    @classmethod
    def _row_to_dict(
        cls,
        row: Optional[sqlite3.Row],
        json_fields: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Convert a sqlite3.Row to a plain dictionary.

        Booleans are restored for ``enabled`` columns and JSON fields listed in
        ``json_fields`` are parsed back into Python values.
        """
        if row is None:
            return None
        result: dict[str, Any] = dict(row)
        if "enabled" in result:
            result["enabled"] = bool(result["enabled"])
        for key in json_fields or set():
            if key in result and result[key] is not None:
                try:
                    result[key] = json.loads(result[key])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in field {key!r}: {result[key]!r}"
                    ) from exc
        return result

    @classmethod
    def _rows_to_dicts(
        cls,
        rows: list[sqlite3.Row],
        json_fields: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        """Convert a list of sqlite3.Row objects to plain dictionaries.

        ``rows`` is expected to come from ``cursor.fetchall()`` and therefore
        never contains ``None``. The cast reflects that guarantee while keeping
        ``_row_to_dict`` usable for ``fetchone()`` results.
        """
        return [
            cast(dict[str, Any], cls._row_to_dict(row, json_fields=json_fields))
            for row in rows
        ]

    @staticmethod
    def _to_json(value: Any) -> Optional[str]:
        """Serialize a value to a JSON string.``None`` returns ``None``."""
        if value is None:
            return None
        return json.dumps(value)

    @classmethod
    def _validate_kwargs(
        cls,
        kwargs: dict[str, Any],
        allowed_columns: set[str],
        label: str,
    ) -> None:
        """Validate kwargs against an allowed column set.

        Raises:
            ValueError: If a reserved field or an unknown field is present.
        """
        for key in kwargs:
            if key in cls._RESERVED_FIELDS:
                raise ValueError(
                    f"Field {key!r} is reserved and cannot be set via kwargs"
                )
            if key not in allowed_columns:
                raise ValueError(f"Unknown {label} field: {key!r}")

    @staticmethod
    def _validate_sql_identifiers(identifiers: list[str]) -> None:
        """Validate column/table identifiers before interpolating them into SQL."""
        for identifier in identifiers:
            if not validate_identifier(identifier):
                raise ValueError(f"Invalid SQL identifier: {identifier!r}")

    # ------------------------------------------------------------------
    # Reminder tasks
    # ------------------------------------------------------------------

    def create_reminder_task(self, owner_id: str, title: str, **kwargs: Any) -> str:
        """Create a reminder task and return its generated local UUID."""
        with self.transaction() as conn:
            task_id = self._create_reminder_task_conn(
                conn, owner_id, title, **kwargs
            )

        logger.debug(f"Created reminder task {task_id} for owner {owner_id}")
        return task_id

    def get_reminder_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Fetch a reminder task by local id."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM reminder_tasks WHERE id = ?", (task_id,)
            )
            return self._row_to_dict(
                cursor.fetchone(), json_fields=self._REMINDER_JSON_FIELDS
            )

    def get_reminder_task_by_server_id(
        self,
        owner_id: str,
        server_id: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch a reminder task by owner and server-side identifier."""
        with closing(self._get_connection()) as conn:
            return self._get_reminder_task_by_server_id_conn(
                conn, owner_id, server_id
            )

    def list_reminder_tasks(
        self,
        owner_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        status: Optional[str] = None,
        *,
        armable_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List reminder tasks with optional filters.

        ``armable_only=True`` adds the same dormant-transfer-state
        exclusion `list_armable_automation_definitions` applies (spec §6.1
        ruling 2: `DORMANT_TRANSFER_STATES` rows sit out): the DB-query
        half of `PriorityQueue.load`'s defense-in-depth pair. Callers that
        want every row for display (e.g. the workbench, which shows a
        dormant row's "waiting for server" state) leave it ``False``.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if owner_id is not None:
            conditions.append("owner_id = ?")
            params.append(owner_id)
        if enabled is not None:
            conditions.append("enabled = ?")
            params.append(1 if enabled else 0)
        if status is not None:
            conditions.append("last_status = ?")
            params.append(status)
        if armable_only:
            placeholders = ", ".join("?" for _ in DORMANT_TRANSFER_STATES)
            conditions.append(
                f"(transfer_state IS NULL OR transfer_state NOT IN ({placeholders}))"
            )
            params.extend(DORMANT_TRANSFER_STATES)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"SELECT * FROM reminder_tasks {where_clause} ORDER BY created_at",
                params,
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields=self._REMINDER_JSON_FIELDS
            )

    def update_reminder_task(self, task_id: str, **kwargs: Any) -> bool:
        """Update reminder task fields. Returns True if a row was changed."""
        with self.transaction() as conn:
            return self._update_reminder_task_conn(conn, task_id, **kwargs)

    def delete_reminder_task(self, task_id: str) -> bool:
        """Delete a reminder task by local id."""
        with self.transaction() as conn:
            return self._delete_reminder_task_conn(conn, task_id)

    def reminders_due_before(self, now: datetime) -> list[dict[str, Any]]:
        """Return enabled reminders whose next_run_at is at or before ``now``.

        Excludes `DORMANT_TRANSFER_STATES` rows unconditionally (spec §6.1
        ruling 2) -- this is the back-compat ``now=``-provided load path
        `PriorityQueue.load` uses, i.e. armable-only by construction; its
        sole caller is the queue, unlike `list_reminder_tasks` which also
        serves display listings that must keep dormant rows visible.
        """
        now_iso = self._to_utc_iso(now)
        dormant_placeholders = ", ".join("?" for _ in DORMANT_TRANSFER_STATES)
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM reminder_tasks
                WHERE enabled = 1
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                  AND (transfer_state IS NULL
                       OR transfer_state NOT IN ({dormant_placeholders}))
                ORDER BY next_run_at
                """,
                (now_iso, *DORMANT_TRANSFER_STATES),
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields=self._REMINDER_JSON_FIELDS
            )

    # -- TASK-26026: durable per-dispatch run ledger --------------------

    #: Retention default -- rows kept per task before pruning. A documented
    #: bound so the ledger cannot grow without limit (AC#3).
    DEFAULT_RUN_HISTORY_PER_TASK = 50

    #: The non-terminal status a reconcile sweep fails on next start (AC#4).
    _RUNNING_RUN_STATUS = "running"

    def begin_task_run(
        self, task_id: str, task_type: str, started_at: datetime
    ) -> int:
        """Record the start of one dispatch; returns the run row id.

        The row is left in ``running`` until ``finish_task_run`` writes a
        terminal status -- an app exit mid-dispatch therefore leaves a
        ``running`` row that startup reconciliation fails (AC#4).
        """
        now_iso = self._to_utc_iso(started_at)
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO scheduled_task_runs "
                "(task_id, task_type, status, started_at, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(task_id),
                    str(task_type),
                    self._RUNNING_RUN_STATUS,
                    now_iso,
                    now_iso,
                ),
            )
            return int(cursor.lastrowid)

    def finish_task_run(
        self,
        run_id: int,
        status: str,
        finished_at: datetime,
        *,
        error: Optional[str] = None,
    ) -> None:
        """Write a run's terminal status/finish/error."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE scheduled_task_runs "
                "SET status = ?, finished_at = ?, error_msg = ? "
                "WHERE id = ?",
                (
                    str(status),
                    self._to_utc_iso(finished_at),
                    (str(error)[:1000] if error is not None else None),
                    int(run_id),
                ),
            )

    def list_task_runs(
        self, task_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return a task's run history, newest first (AC#2)."""
        with closing(self._get_connection()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE task_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (str(task_id), int(limit)),
            ).fetchall()
            return [dict(row) for row in rows]

    def prune_task_runs(
        self, *, keep_per_task: int = DEFAULT_RUN_HISTORY_PER_TASK
    ) -> int:
        """Delete all but the newest ``keep_per_task`` runs per task (AC#3).

        Returns how many rows were removed.
        """
        keep = max(0, int(keep_per_task))
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM scheduled_task_runs WHERE id IN ("
                "  SELECT id FROM ("
                "    SELECT id, ROW_NUMBER() OVER ("
                "      PARTITION BY task_id ORDER BY id DESC"
                "    ) AS rn FROM scheduled_task_runs"
                "  ) WHERE rn > ?"
                ")",
                (keep,),
            )
            return int(cursor.rowcount)

    def fail_interrupted_task_runs(self, *, now: datetime) -> int:
        """Fail every ``running`` row on startup (AC#4).

        An unfinished run means the app exited mid-dispatch; a terminal
        ``failed`` with a finish time is more honest than a row stuck
        ``running`` forever. Finished history is untouched.
        """
        now_iso = self._to_utc_iso(now)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE scheduled_task_runs "
                "SET status = 'failed', "
                "    finished_at = COALESCE(finished_at, ?), "
                "    error_msg = COALESCE(error_msg, ?) "
                "WHERE status = ?",
                (
                    now_iso,
                    "interrupted by application exit",
                    self._RUNNING_RUN_STATUS,
                ),
            )
            return int(cursor.rowcount)

    # -- TASK-26027: durable failure incidents ---------------------------

    def record_task_failure(
        self, task_id: str, task_type: str, signature: str, now: datetime
    ) -> tuple[int, bool]:
        """Group a failure into an incident; return (incident_id, should_notify).

        A new (task_id, signature) opens an ``alerting`` incident and
        notifies (AC#1). A repeat of an already-open incident (alerting OR
        acknowledged) bumps its count and does NOT re-notify -- grouped
        (AC#1/#2). A different signature opens its own incident (AC#3).
        """
        now_iso = self._to_utc_iso(now)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM task_incidents "
                "WHERE task_id = ? AND signature = ? AND status != 'closed' "
                "LIMIT 1",
                (str(task_id), str(signature)),
            ).fetchone()
            if row is not None:
                incident_id = int(row[0])
                conn.execute(
                    "UPDATE task_incidents "
                    "SET occurrence_count = occurrence_count + 1, "
                    "    last_seen_at = ? WHERE id = ?",
                    (now_iso, incident_id),
                )
                return incident_id, False
            cursor = conn.execute(
                "INSERT INTO task_incidents "
                "(task_id, task_type, signature, status, occurrence_count, "
                " first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, 'alerting', 1, ?, ?)",
                (str(task_id), str(task_type), str(signature), now_iso, now_iso),
            )
            return int(cursor.lastrowid), True

    def record_task_success(self, task_id: str, now: datetime) -> int:
        """Close every open incident for a task; return how many closed (AC#4)."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE task_incidents "
                "SET status = 'closed', closed_at = ? "
                "WHERE task_id = ? AND status != 'closed'",
                (self._to_utc_iso(now), str(task_id)),
            )
            return int(cursor.rowcount)

    def acknowledge_incident(self, incident_id: int, now: datetime) -> None:
        """Acknowledge one incident: suppress further notifications only.

        Never disables the task or removes it from the queue (AC#7) -- this
        touches only the incident row.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE task_incidents "
                "SET status = 'acknowledged', acknowledged_at = ? "
                "WHERE id = ? AND status = 'alerting'",
                (self._to_utc_iso(now), int(incident_id)),
            )

    def list_task_incidents(
        self, task_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return a task's incidents, newest first (AC-visibility)."""
        with closing(self._get_connection()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM task_incidents WHERE task_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (str(task_id), int(limit)),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_reminder_dispatched(
        self,
        task_id: str,
        now: datetime,
        success: bool = True,
        *,
        grace_seconds: float = 0.0,
        timed_out: bool = False,
    ) -> None:
        """Update a reminder after dispatch so it is not immediately redispatched.

        For ``one_time`` reminders the task is disabled and ``next_run_at`` is
        cleared. For ``recurring`` reminders the next occurrence is computed from
        the stored cron expression and timezone.

        Missed-fire accounting (task-18937): when the scheduled time of this
        dispatch (the stored ``next_run_at``) is more than ``grace_seconds``
        before ``now``, the dispatch was late. The row then records
        ``missed_at`` = the earliest owed occurrence's scheduled time and, for
        recurring tasks, ``missed_count`` = occurrences that elapsed
        undispatched *before* this one (the dispatch itself covers exactly one
        occurrence; skipped ones are counted, never replayed --
        run-once-then-continue). An on-time dispatch clears both fields: the
        state describes the last dispatch and self-heals.

        What "late" does NOT tell you (task-19562): *why*. This docstring
        used to assert the cause -- "the scheduler was not running (or not
        aware of the task) at the scheduled time" -- and the UI repeated it
        as "Missed while away". Both were wrong for a real and ordinary
        case: ``SchedulerLoop.tick`` awaits every due handler serially and
        inline, so one slow handler delays every task behind it, easily past
        the grace, with the scheduler running throughout. These two fields
        record the lateness and the skipped occurrences, which are true
        regardless of cause; the cause itself is reported by the loop, which
        is the only place that knows it (``SchedulerLoop.
        _report_lateness_cause``).

        Timeout (task-18939): ``timed_out=True`` records the distinct
        terminal status ``"timed_out"`` -- the dispatch ran but was
        cancelled at its execution deadline. This stays separate from both
        ``"completed"`` (finished) and ``"missed"`` (ran and raised):
        ran-but-cancelled is its own honest outcome, and the schedule still
        advances so a wedged handler can never wedge the loop.
        """
        row = self.get_reminder_task(task_id)
        if row is None:
            return

        if timed_out:
            last_status = "timed_out"
        else:
            last_status = "completed" if success else "missed"
        fields: dict[str, Any] = {
            "last_run_at": now,
            "last_status": last_status,
            "updated_at": now,
        }

        scheduled_at = self._parse_utc_iso(row.get("next_run_at"))
        late_by = (
            (now - scheduled_at).total_seconds()
            if scheduled_at is not None
            else 0.0
        )
        if scheduled_at is not None and late_by > grace_seconds:
            fields["missed_at"] = scheduled_at
            schedule_kind = row.get("schedule_kind")
            if schedule_kind == "recurring":
                fields["missed_count"] = self._count_missed_occurrences(
                    row, scheduled_at, now
                )
            else:
                # one_time: fired late, nothing was skipped before it.
                fields["missed_count"] = 0
        else:
            fields["missed_at"] = None
            fields["missed_count"] = 0

        schedule_kind = row.get("schedule_kind")
        if schedule_kind == "one_time":
            fields["enabled"] = False
            fields["next_run_at"] = None
        elif schedule_kind == "recurring":
            cron_expr = row.get("cron")
            tz_name = row.get("timezone") or "UTC"
            next_run: datetime | None = None
            if cron_expr:
                try:
                    tz: tzinfo = ZoneInfo(tz_name)
                except Exception:
                    tz = timezone.utc
                base = now.astimezone(tz)
                next_run = croniter(cron_expr, base).get_next(datetime)
                next_run = next_run.astimezone(timezone.utc)
            fields["next_run_at"] = next_run

        self.update_reminder_task(task_id, **fields)

    @staticmethod
    def _parse_utc_iso(value: Any) -> Optional[datetime]:
        """Parse a stored UTC ISO-8601 string, tolerating junk as ``None``."""
        if not value or not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    #: Cap for skipped-occurrence counting: a hostile or hand-edited cron
    #: (every second) must not turn a long absence into an unbounded loop.
    #: The stored value uses the negative sentinel below when the true count
    #: exceeds the cap, so a capped count is never presented as exact
    #: (review finding: silent truncation).
    _MISSED_COUNT_CAP = 100_000
    _MISSED_COUNT_OVERFLOW = -1

    @classmethod
    def _count_missed_occurrences(
        cls, row: dict[str, Any], scheduled_at: datetime, now: datetime
    ) -> int:
        """Count cron occurrences in ``(scheduled_at, now)`` -- the skipped ones.

        The occurrence AT ``scheduled_at`` is the one this late dispatch
        covers, and an occurrence landing exactly at ``now`` coincides with
        the dispatch itself (the user is notified at that moment), so both
        endpoints are exclusive: "skipped" means scheduled strictly between
        the owed occurrence and when the dispatch actually fired. Returns 0
        when the cron expression is unusable -- an honest unknown is
        reported as "none skipped" rather than an exception.

        Returns ``_MISSED_COUNT_OVERFLOW`` (-1) when more than
        ``_MISSED_COUNT_CAP`` occurrences elapsed: the UI renders that as
        an explicit "more than N" rather than a false exact number.
        """
        cron_expr = row.get("cron")
        if not cron_expr or not isinstance(cron_expr, str):
            return 0
        tz_name = row.get("timezone") or "UTC"
        try:
            tz: tzinfo = ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc
        try:
            iterator = croniter(cron_expr, scheduled_at.astimezone(tz))
        except (ValueError, KeyError):
            return 0
        skipped = 0
        while skipped <= cls._MISSED_COUNT_CAP:
            occurrence = iterator.get_next(datetime)
            if occurrence is None or occurrence >= now:
                return skipped
            skipped += 1
        return cls._MISSED_COUNT_OVERFLOW

    # ------------------------------------------------------------------
    # Transfer state (spec §6) -- shared by both primitives
    # ------------------------------------------------------------------

    #: ``table_kind`` -> table name, for `set_transfer_state`/
    #: `clear_transfer_state`. Keys match the ``primitive`` string
    #: convention used across this module and `SyncEngine`/
    #: `SchedulingService` (``_REMINDER_PRIMITIVE``/``_DEFINITION_
    #: PRIMITIVE``) -- not user input, so building the ``UPDATE``'s table
    #: name from this dict carries no injection risk.
    _TRANSFER_STATE_TABLES = {
        "reminder_task": "reminder_tasks",
        "automation_definition": "automation_definitions",
    }

    def set_transfer_state(
        self,
        table_kind: str,
        row_id: str,
        state: Optional[str],
        *,
        expected: tuple[Optional[str], ...],
    ) -> bool:
        """Compare-and-set ``transfer_state`` on a reminder or definition row.

        The transfer machine's transitions (§6.1/§6.2) must be race-safe
        against a concurrent sync push and a concurrent UI cancel/retry
        both touching the same row's ``transfer_state`` -- so this is ONE
        guarded ``UPDATE ... WHERE id = ? AND (<expected-state guard>)``,
        never a separate SELECT-then-UPDATE: a plain ``SELECT`` takes no
        lock under Python's ``sqlite3`` default isolation (no implicit
        ``BEGIN`` before a read), so a read-then-write pair across two
        statements is NOT serialized against a second connection racing
        the same row -- two callers could each read the pre-transition
        state before either writes, and both would then "succeed",
        the second silently clobbering the first (fix-round-1 finding,
        reproduced by `test_set_transfer_state_concurrent_callers_do_not_
        both_succeed`). Folding the state check into the UPDATE's own
        WHERE clause makes the check-and-write a single statement, which
        SQLite's writer lock always runs atomically against other
        connections: whichever caller's UPDATE actually executes first
        wins and commits; a second caller's UPDATE then finds the WHERE
        no longer matches (the row already changed), so its
        ``rowcount`` is 0 and it correctly returns ``False``.

        Deliberately does NOT bump `automation_definitions.version` --
        same ``bump_version=False`` precedent `update_automation_
        definition` documents for machinery-driven updates (e.g. the
        scheduler's ``next_run_at`` advance): a transfer transition is not
        a user content edit, and bumping it on every transition would
        pollute optimistic-lock conflict detection the same way.

        Args:
            table_kind: ``"reminder_task"`` or ``"automation_definition"``.
            row_id: The row's local id.
            state: The new ``transfer_state`` value (``None`` clears it).
            expected: Current-state values that permit the transition.
                Refused (returns ``False``, no write) when the row's live
                ``transfer_state`` is not one of these -- including when
                the row does not exist at all.

        Returns:
            ``True`` if the row existed, its current state was in
            ``expected``, and the write happened; ``False`` otherwise.

        Raises:
            ValueError: ``table_kind`` is not a known primitive.
        """
        table = self._TRANSFER_STATE_TABLES.get(table_kind)
        if table is None:
            raise ValueError(f"Unknown table_kind for transfer_state: {table_kind!r}")

        # Build the expected-state guard as part of the UPDATE's WHERE,
        # not a separate read: `IN (...)` never matches NULL in SQL, so
        # a `None` in `expected` needs its own `IS NULL` branch alongside
        # the `IN (...)` branch for any non-NULL expected values.
        non_null_expected = [value for value in expected if value is not None]
        guard_clauses: list[str] = []
        params: list[Any] = [state, self._to_utc_iso(datetime.now(timezone.utc)), row_id]
        if None in expected:
            guard_clauses.append("transfer_state IS NULL")
        if non_null_expected:
            placeholders = ", ".join("?" for _ in non_null_expected)
            guard_clauses.append(f"transfer_state IN ({placeholders})")
            params.extend(non_null_expected)
        if not guard_clauses:
            # Nothing in `expected` -- no live state can ever satisfy an
            # empty set, so refuse without touching the DB.
            return False

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE {table} SET transfer_state = ?, updated_at = ? "
                f"WHERE id = ? AND ({' OR '.join(guard_clauses)})",
                params,
            )
            return cursor.rowcount > 0

    def clear_transfer_state(
        self,
        table_kind: str,
        row_id: str,
        *,
        expected: tuple[Optional[str], ...],
    ) -> bool:
        """Sugar for ``set_transfer_state(table_kind, row_id, None, ...)``."""
        return self.set_transfer_state(table_kind, row_id, None, expected=expected)

    def convert_row_to_server_mirror(
        self,
        table_kind: str,
        local_id: str,
        server_item: dict[str, Any],
        owner_id: str,
    ) -> str:
        """Convert a `transfer_to_server` row into its server-owned mirror.

        Called by `SyncEngine` right after a transfer's create call acks
        (spec §6.1.4). ``owner_id`` is the DESTINATION scope the transfer
        targets (the pending mutation's own ``owner_id``, e.g.
        ``"server:1"``) -- distinct from the transferring row's OWN
        ``owner_id`` column, which is still ``"local"`` (or whatever it
        was) until this call changes it. This is the same
        `target_owner`-vs-row-owner_id distinction
        `_apply_pulled_reminders`/`upsert_automation_definitions_from_
        server` already draw for a pull.

        Two outcomes, both inside ONE transaction:

        - A pulled mirror already exists for ``(owner_id, server_id)``
          (the §4 ``UNIQUE(owner_id, server_id)`` race -- a background
          pull landed the same server row between this transfer's send
          and its ack): server-wins, same rule every other pull-mirror
          write already follows -- keep the PULLED mirror, delete the
          local transferring row, and transplant its ``created_at`` onto
          the mirror (plus any `automation_audit_events` rows, definitions
          only) so the mirror's history/audit linkage doesn't silently
          reset.
        - Otherwise: convert the local row in place -- set ``server_id``,
          reassign ``owner_id`` to the destination scope, and clear
          ``transfer_state``. Reassigning ``owner_id`` (not just setting
          ``server_id``) is what actually excludes the row from local
          execution going forward: every armable filter and
          `is_server_scoped_owner` key off the ``owner_id`` prefix, not
          ``transfer_state`` alone (`transfer_state` only covers the
          in-flight window). Reminders also gain a `sync_mapping` row here
          (mirrors `_apply_pulled_reminders`'s own bookkeeping for every
          other server-known reminder); definitions have no equivalent
          table.

        Args:
            table_kind: ``"reminder_task"`` or ``"automation_definition"``.
            local_id: The local row that pushed the transfer.
            server_item: The server's create response (must carry ``id``).
            owner_id: The destination owner scope.

        Returns:
            ``"converted"``, ``"merged"``, or ``"vanished"`` (the local
            row was already gone by ack time -- same orphan shape as
            `adopt_server_definition_identity`; nothing left to convert).

        Raises:
            ValueError: unknown ``table_kind``, or ``server_item`` carries
                no ``id``.
        """
        table = self._TRANSFER_STATE_TABLES.get(table_kind)
        if table is None:
            raise ValueError(
                f"Unknown table_kind for convert_row_to_server_mirror: {table_kind!r}"
            )
        server_id = server_item.get("id")
        if not server_id:
            raise ValueError(
                "server_item must carry an 'id' to convert a transferred row"
            )

        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            row = conn.execute(
                f"SELECT created_at FROM {table} WHERE id = ?", (local_id,)
            ).fetchone()
            if row is None:
                return "vanished"

            existing_mirror = conn.execute(
                f"SELECT id FROM {table} WHERE owner_id = ? AND server_id = ? "
                "AND id != ?",
                (owner_id, server_id, local_id),
            ).fetchone()

            if existing_mirror is not None:
                mirror_id = existing_mirror["id"]
                conn.execute(
                    f"UPDATE {table} SET created_at = ? WHERE id = ?",
                    (row["created_at"], mirror_id),
                )
                if table_kind == "automation_definition":
                    conn.execute(
                        "UPDATE automation_audit_events SET definition_id = ? "
                        "WHERE definition_id = ?",
                        (mirror_id, local_id),
                    )
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (local_id,))
                return "merged"

            conn.execute(
                f"UPDATE {table} SET server_id = ?, owner_id = ?, "
                "transfer_state = NULL, updated_at = ? WHERE id = ?",
                (server_id, owner_id, now_iso, local_id),
            )
            if table_kind == "reminder_task":
                self._set_sync_mapping_conn(
                    conn, local_id, server_id, table_kind, owner_id
                )
            return "converted"

    def create_local_copy_from_mirror(self, table_kind: str, mirror_id: str) -> str:
        """Create a dormant local-owner copy of a server-owned mirror row (spec §6.2.1).

        Called when a `release_from_server` transfer starts on a mirror
        row (``owner_id`` != ``"local"``, ``server_id`` set): a NEW row is
        created with ``owner_id="local"``, a fresh local id, ``server_id=
        None``, and ``transfer_state="from_server_pending"`` -- a
        `DORMANT_TRANSFER_STATES` member, so the copy sits out of every
        armable-row query (Task 1) until the release replay's ack clears
        it (`SyncEngine._push_definition_release`/`_push_reminder_
        release`, Task 5). The mirror row itself is left completely
        untouched -- it keeps executing server-side until the release
        actually lands.

        A definition's ``schedule`` is stored in SERVER vocabulary on the
        mirror (pulled verbatim by `upsert_automation_definitions_from_
        server`, never translated on the way in) and is translated to
        CLIENT vocabulary here via `to_local_schedule` before ``next_run_
        at`` is computed -- Task 3's documented translation direction,
        and this is that function's first real caller. A reminder's
        schedule fields (`schedule_kind`/`run_at`/`cron`/`timezone`) use
        the SAME vocabulary on both sides (no rename table exists for
        them), so they are copied through unchanged.

        ``next_run_at`` is computed FRESH from the (translated) schedule
        at "now", the same way a brand-new local row's is
        (`SchedulingService._definition_db_fields_from_preview`'s
        `compute_next_run_at` call / `_compute_next_run_at`'s one_time-
        passthrough-or-next-cron-occurrence split) -- not copied from the
        mirror's own possibly-stale value, since the mirror may not have
        pulled the server's latest progress yet.

        ONE transaction: the mirror read and the copy's INSERT are atomic,
        so a concurrent mirror delete/pull-update can never leave the
        copy built from half-updated data.

        Args:
            table_kind: ``"reminder_task"`` or ``"automation_definition"``.
            mirror_id: The server-owned mirror row's local id.

        Returns:
            The new copy's local id.

        Raises:
            ValueError: unknown ``table_kind``, or no row exists at
                ``mirror_id``.
        """
        if table_kind not in self._TRANSFER_STATE_TABLES:
            raise ValueError(
                f"Unknown table_kind for create_local_copy_from_mirror: {table_kind!r}"
            )

        now = datetime.now(timezone.utc)
        with self.transaction() as conn:
            if table_kind == "reminder_task":
                mirror = self._row_to_dict(
                    conn.execute(
                        "SELECT * FROM reminder_tasks WHERE id = ?", (mirror_id,)
                    ).fetchone()
                )
                if mirror is None:
                    raise ValueError(f"No reminder_task mirror row at id {mirror_id!r}")

                schedule_kind = mirror.get("schedule_kind") or "one_time"
                if schedule_kind == "one_time":
                    next_run_at = self._parse_utc_iso(mirror.get("run_at"))
                elif schedule_kind == "recurring" and mirror.get("cron"):
                    try:
                        tz: tzinfo = ZoneInfo(mirror.get("timezone") or "UTC")
                    except Exception:
                        tz = timezone.utc
                    next_run_at = (
                        croniter(mirror["cron"], now.astimezone(tz))
                        .get_next(datetime)
                        .astimezone(timezone.utc)
                    )
                else:
                    next_run_at = None

                return self._create_reminder_task_conn(
                    conn,
                    "local",
                    mirror["title"],
                    body=mirror.get("body"),
                    schedule_kind=schedule_kind,
                    run_at=mirror.get("run_at"),
                    cron=mirror.get("cron"),
                    timezone=mirror.get("timezone"),
                    timeout_seconds=mirror.get("timeout_seconds"),
                    transfer_state="from_server_pending",
                    next_run_at=next_run_at,
                )

            mirror = self._row_to_dict(
                conn.execute(
                    "SELECT * FROM automation_definitions WHERE id = ?", (mirror_id,)
                ).fetchone(),
                json_fields=self._AUTOMATION_JSON_FIELDS,
            )
            if mirror is None:
                raise ValueError(
                    f"No automation_definition mirror row at id {mirror_id!r}"
                )

            local_schedule = to_local_schedule(mirror.get("schedule") or {})
            definition_id = str(uuid.uuid4())
            fields: dict[str, Any] = {
                "id": definition_id,
                "owner_id": "local",
                "family": mirror["family"],
                "name": mirror["name"],
                "description": mirror.get("description"),
                "lifecycle": mirror.get("lifecycle") or "configured",
                "health": "execution_unavailable",
                "schedule": local_schedule,
                "input": mirror.get("input") or {},
                "config": mirror.get("config") or {},
                "visibility_policy": mirror.get("visibility_policy") or {},
                "notification_policy": mirror.get("notification_policy") or {},
                "approval_policy": mirror.get("approval_policy") or {},
                "finding_policy": mirror.get("finding_policy") or {},
                "retention_policy": mirror.get("retention_policy") or {},
                "version": 1,
                "transfer_state": "from_server_pending",
                "next_run_at": compute_next_run_at(local_schedule, now=now),
                "created_at": now,
                "updated_at": now,
            }
            serialized = self._serialize_definition_fields(fields)
            self._validate_sql_identifiers(list(serialized.keys()))
            columns = ", ".join(serialized.keys())
            placeholders = ", ".join(["?"] * len(serialized))
            conn.execute(
                f"INSERT INTO automation_definitions ({columns}) VALUES ({placeholders})",
                list(serialized.values()),
            )
            return definition_id

    # ------------------------------------------------------------------
    # Automation definitions
    # ------------------------------------------------------------------

    def create_automation_definition(
        self,
        owner_id: str,
        family: str,
        name: str,
        *,
        pending_mutation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Create an automation definition and return its generated local UUID.

        Defaults ``lifecycle`` to ``configured`` and ``health`` to
        ``execution_unavailable`` when not provided. JSON fields are serialized
        and datetime fields are converted to UTC ISO-8601 strings.

        When ``pending_mutation`` is given (``{"primitive", "owner_id",
        "payload"}``), an ``automation_definition`` mutation is recorded in
        ``pending_mutations`` in the SAME transaction as the INSERT --
        mirrors ``update_result_review``'s ``pending_mutation`` precedent,
        so a crash between the two writes can never leave a local row
        without the outbox row that pushes it. Unlike that precedent,
        the dict carries no ``local_id``: this call generates the
        definition's id itself, so the mutation is always keyed by the id
        this call returns, not one the caller could have supplied ahead
        of time.
        """
        self._validate_kwargs(
            kwargs, self._AUTOMATION_DEFINITION_COLUMNS, "automation definition"
        )

        definition_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        now_iso = self._to_utc_iso(now)

        fields: dict[str, Any] = {
            "id": definition_id,
            "owner_id": owner_id,
            "family": family,
            "name": name,
            "lifecycle": "configured",
            "health": "execution_unavailable",
            "version": 1,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        for key, value in kwargs.items():
            if value is None and key in ("lifecycle", "health"):
                continue
            if key in self._AUTOMATION_JSON_FIELDS:
                fields[key] = self._to_json(value)
            elif key in self._DATETIME_FIELDS:
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value

        self._validate_sql_identifiers(list(fields.keys()))
        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))

        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO automation_definitions ({columns}) VALUES ({placeholders})",
                list(fields.values()),
            )
            if pending_mutation is not None:
                self._insert_pending_mutation_conn(
                    conn,
                    local_id=definition_id,
                    primitive=pending_mutation["primitive"],
                    owner_id=pending_mutation["owner_id"],
                    payload=pending_mutation["payload"],
                    now_iso=now_iso,
                )

        logger.debug(
            f"Created automation definition {definition_id} for owner {owner_id}"
        )
        return definition_id

    def get_automation_definition(self, definition_id: str) -> Optional[dict[str, Any]]:
        """Fetch an automation definition by local id."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM automation_definitions WHERE id = ?", (definition_id,)
            )
            return self._row_to_dict(
                cursor.fetchone(), json_fields=self._AUTOMATION_JSON_FIELDS
            )

    def get_automation_definition_by_server_id(
        self, owner_id: str, server_id: str
    ) -> Optional[dict[str, Any]]:
        """Fetch an automation definition by owner and server-side identifier.

        Mirrors ``get_reminder_task_by_server_id``. Used by the authoring
        facade (Task 4) to recover the local row an online create just
        mirrored via ``upsert_automation_definitions_from_server`` -- that
        upsert reports only insert/update counts, not the generated id.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM automation_definitions WHERE owner_id = ? AND server_id = ?",
                (owner_id, server_id),
            )
            return self._row_to_dict(
                cursor.fetchone(), json_fields=self._AUTOMATION_JSON_FIELDS
            )

    def list_automation_definitions(
        self,
        owner_id: Optional[str] = None,
        lifecycle: Optional[str] = None,
        family: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List automation definitions with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if owner_id is not None:
            conditions.append("owner_id = ?")
            params.append(owner_id)
        if lifecycle is not None:
            conditions.append("lifecycle = ?")
            params.append(lifecycle)
        if family is not None:
            conditions.append("family = ?")
            params.append(family)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"SELECT * FROM automation_definitions {where_clause} ORDER BY created_at",
                params,
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields=self._AUTOMATION_JSON_FIELDS
            )

    def list_armable_automation_definitions(
        self, owner_id: str = "local"
    ) -> list[dict[str, Any]]:
        """List local definitions ready to feed the scheduler queue (§7.2).

        A row arms only when all four hold: ``family='recurring_question'``
        (v1 -- the only executor registered), ``lifecycle='configured'``,
        a real ``next_run_at``, and ``transfer_state`` not in
        `DORMANT_TRANSFER_STATES` -- a definition that has actually been
        sent to the server (or is a dormant server-release copy) is not
        this side's to run; a merely-queued or failed transfer keeps
        arming (spec §6.1 ruling 2). ``owner_id`` defaults to ``"local"``:
        this is the accessor half of the defense-in-depth pairing with
        `PriorityQueue`'s own `is_server_scoped_owner` guard (slice 1) --
        neither alone is trusted to keep a server-scoped definition from
        arming locally.

        The result is bounded to `_ARMABLE_DEFINITIONS_CAP` rows, ordered by
        `next_run_at` ascending, so the soonest-due definitions are kept and
        any overflow is the latest-scheduled tail. A `logger.warning` fires
        when the cap is hit, since a truncated arm set must never fail
        silently.

        Args:
            owner_id: Owner scope to arm for. Defaults to ``"local"``.

        Returns:
            Armable definition rows (as dicts), ordered by ``next_run_at``
            ascending, capped at `_ARMABLE_DEFINITIONS_CAP` rows.
        """
        dormant_placeholders = ", ".join("?" for _ in DORMANT_TRANSFER_STATES)
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM automation_definitions
                WHERE family = 'recurring_question'
                  AND lifecycle = 'configured'
                  AND next_run_at IS NOT NULL
                  AND (transfer_state IS NULL
                       OR transfer_state NOT IN ({dormant_placeholders}))
                  AND owner_id = ?
                ORDER BY next_run_at
                LIMIT ?
                """,
                (*DORMANT_TRANSFER_STATES, owner_id, self._ARMABLE_DEFINITIONS_CAP),
            )
            rows = cursor.fetchall()
            if len(rows) == self._ARMABLE_DEFINITIONS_CAP:
                logger.warning(
                    "list_armable_automation_definitions: armable set truncated "
                    "at _ARMABLE_DEFINITIONS_CAP={} rows for owner_id={!r}",
                    self._ARMABLE_DEFINITIONS_CAP,
                    owner_id,
                )
            return self._rows_to_dicts(rows, json_fields=self._AUTOMATION_JSON_FIELDS)

    def update_automation_definition(
        self,
        definition_id: str,
        *,
        bump_version: bool = True,
        pending_mutation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update automation-definition fields. Returns True if a row changed.

        The ``version`` column is automatically incremented for optimistic
        locking; any ``version`` value supplied in kwargs is ignored. Pass
        ``bump_version=False`` for a non-edit update (e.g. the scheduler's
        `next_run_at` advance) so version churn doesn't pollute conflict
        detection -- PR-2 final-review parking note.

        When ``pending_mutation`` is given (``{"primitive", "owner_id",
        "payload"}``), an ``automation_definition`` mutation is recorded in
        ``pending_mutations`` -- keyed by ``definition_id`` -- in the SAME
        transaction as the UPDATE, same atomicity precedent as
        ``create_automation_definition``'s. Never recorded when the row
        didn't change (unknown ``definition_id``): there is nothing to
        push.
        """
        if not kwargs:
            return False

        self._validate_kwargs(
            kwargs, self._AUTOMATION_DEFINITION_COLUMNS, "automation definition"
        )

        updates: list[str] = []
        params: list[Any] = []

        for key, value in kwargs.items():
            if key == "version":
                # version is auto-incremented below; ignore user-supplied value
                continue
            if key in self._AUTOMATION_JSON_FIELDS:
                updates.append(f"{key} = ?")
                params.append(self._to_json(value))
            elif key in self._DATETIME_FIELDS:
                updates.append(f"{key} = ?")
                params.append(self._to_utc_iso(value))
            else:
                updates.append(f"{key} = ?")
                params.append(value)

        if not updates:
            return False

        self._validate_sql_identifiers([key.split(" ", 1)[0] for key in updates])
        if bump_version:
            updates.append("version = version + 1")
        updates.append("updated_at = ?")
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        params.append(now_iso)
        params.append(definition_id)

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE automation_definitions SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            changed = cursor.rowcount > 0
            if changed and pending_mutation is not None:
                self._insert_pending_mutation_conn(
                    conn,
                    local_id=definition_id,
                    primitive=pending_mutation["primitive"],
                    owner_id=pending_mutation["owner_id"],
                    payload=pending_mutation["payload"],
                    now_iso=now_iso,
                )
            return changed

    def delete_automation_definition(self, definition_id: str) -> bool:
        """Delete an automation definition by local id."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM automation_definitions WHERE id = ?", (definition_id,)
            )
            return cursor.rowcount > 0

    def log_automation_audit_event(
        self,
        definition_id: str,
        owner_id: str,
        event_type: str,
        actor: str,
        summary: str,
        **kwargs: Any,
    ) -> str:
        """Log an audit event for an automation definition.

        JSON fields (``before``, ``after``) are serialized; datetime fields are
        stored as UTC ISO-8601 strings.
        """
        self._validate_kwargs(
            kwargs, self._AUTOMATION_AUDIT_EVENT_COLUMNS, "automation audit event"
        )

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        fields: dict[str, Any] = {
            "id": event_id,
            "definition_id": definition_id,
            "owner_id": owner_id,
            "event_type": event_type,
            "actor": actor,
            "summary": summary,
            "created_at": self._to_utc_iso(now),
        }

        for key, value in kwargs.items():
            if key in self._AUDIT_JSON_FIELDS:
                fields[key] = self._to_json(value)
            elif key in self._DATETIME_FIELDS:
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value

        self._validate_sql_identifiers(list(fields.keys()))
        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))

        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO automation_audit_events ({columns}) VALUES ({placeholders})",
                list(fields.values()),
            )

        logger.debug(
            f"Created automation audit event {event_id} for definition {definition_id}"
        )
        return event_id

    # ------------------------------------------------------------------
    # Automation runs
    # ------------------------------------------------------------------

    def create_automation_run(
        self,
        owner_id: str,
        definition_id: str,
        definition_version: int,
        trigger_reason: str,
        **kwargs: Any,
    ) -> str | None:
        """Insert a run; return its id, or None when the slot deduped it.

        Also prunes the definition's runs to the newest
        ``_RUNS_RETAINED_PER_DEFINITION`` (spec §4.1): an
        every-15-minutes definition would otherwise write ~35k rows/year.
        """
        self._validate_kwargs(kwargs, self._AUTOMATION_RUN_COLUMNS, "automation run")
        run_id = str(uuid.uuid4())
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        fields: dict[str, Any] = {
            "id": run_id,
            "owner_id": owner_id,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "trigger_reason": trigger_reason,
            "status": "queued",
            "outcome": "none",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self._AUTOMATION_RUN_JSON_FIELDS:
                fields[key] = json.dumps(value)
            elif isinstance(value, datetime):
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value
        self._validate_sql_identifiers(list(fields.keys()))
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self.transaction() as conn:
            try:
                conn.execute(
                    f"INSERT INTO automation_runs ({columns}) VALUES ({placeholders})",
                    list(fields.values()),
                )
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" not in str(exc):
                    raise
                # The (definition, version, slot) UNIQUE fired: this slot
                # already ran. Dedupe is a result, not an error.
                return None
            conn.execute(
                """
                DELETE FROM automation_runs
                WHERE definition_id = ? AND id NOT IN (
                    SELECT id FROM automation_runs
                    WHERE definition_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (definition_id, definition_id, self._RUNS_RETAINED_PER_DEFINITION),
            )
        return run_id

    def update_automation_run(self, run_id: str, **kwargs: Any) -> bool:
        """Update automation-run fields. Returns True if a row changed."""
        if not kwargs:
            return False

        self._validate_kwargs(kwargs, self._AUTOMATION_RUN_COLUMNS, "automation run")

        updates: list[str] = []
        params: list[Any] = []

        for key, value in kwargs.items():
            if key in self._AUTOMATION_RUN_JSON_FIELDS:
                updates.append(f"{key} = ?")
                params.append(self._to_json(value))
            elif isinstance(value, datetime):
                updates.append(f"{key} = ?")
                params.append(self._to_utc_iso(value))
            else:
                updates.append(f"{key} = ?")
                params.append(value)

        self._validate_sql_identifiers([key.split(" ", 1)[0] for key in updates])
        if "updated_at" not in kwargs:
            # Auto-stamp only when the caller didn't supply one, mirroring
            # create_automation_run (caller's value wins) so sync code can
            # set server timestamps without them being clobbered.
            updates.append("updated_at = ?")
            params.append(self._to_utc_iso(datetime.now(timezone.utc)))
        params.append(run_id)

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE automation_runs SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            return cursor.rowcount > 0

    def list_automation_runs(
        self,
        owner_id: str,
        definition_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List automation runs for an owner, newest first.

        Optionally filtered to a single definition; paginated via
        ``limit``/``offset``.
        """
        conditions = ["owner_id = ?"]
        params: list[Any] = [owner_id]

        if definition_id is not None:
            conditions.append("definition_id = ?")
            params.append(definition_id)

        where_clause = f"WHERE {' AND '.join(conditions)}"
        params.extend([limit, offset])

        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"SELECT * FROM automation_runs {where_clause} "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                params,
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields=self._AUTOMATION_RUN_JSON_FIELDS
            )

    def reconcile_stale_automation_runs(self, older_than_seconds: float) -> int:
        """Mark queued/running runs older than the cutoff as interrupted.

        Called at scheduler start (spec §4.1): an app killed mid-run must
        not leave a phantom in-flight run.
        """
        cutoff = self._to_utc_iso(
            datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        )
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE automation_runs
                SET status = 'failed',
                    failure_reason = ?,
                    ended_at = ?,
                    updated_at = ?
                WHERE status IN ('queued', 'running') AND created_at < ?
                """,
                (json.dumps({"code": "interrupted"}), now_iso, now_iso, cutoff),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Automation results
    # ------------------------------------------------------------------

    def create_automation_result(
        self,
        owner_id: str,
        definition_id: str,
        run_id: str,
        kind: str,
        title: str,
        summary: str,
        dedupe_key: str,
        **kwargs: Any,
    ) -> str | None:
        """Insert a result; return its id, or None when the dedupe key fired.

        Mirrors ``create_automation_run``'s create shape: no pruning here
        (results are user-facing findings, not run bookkeeping).
        """
        self._validate_kwargs(kwargs, self._AUTOMATION_RESULT_COLUMNS, "automation result")
        result_id = str(uuid.uuid4())
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        fields: dict[str, Any] = {
            "id": result_id,
            "owner_id": owner_id,
            "definition_id": definition_id,
            "run_id": run_id,
            "kind": kind,
            "title": title,
            "summary": summary,
            "dedupe_key": dedupe_key,
            "review_state": "unread",
            "answer_mode": "none",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self._AUTOMATION_RESULT_JSON_FIELDS:
                fields[key] = json.dumps(value)
            elif isinstance(value, datetime):
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value
        self._validate_sql_identifiers(list(fields.keys()))
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self.transaction() as conn:
            try:
                conn.execute(
                    f"INSERT INTO automation_results ({columns}) VALUES ({placeholders})",
                    list(fields.values()),
                )
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" not in str(exc):
                    raise
                # The (owner_id, dedupe_key) UNIQUE fired: already reported.
                return None
        return result_id

    def list_automation_results(
        self,
        owner_id: str,
        review_state: str | None = None,
        definition_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List automation results for an owner, newest first.

        Optionally filtered by ``review_state`` and/or ``definition_id``;
        paginated via ``limit``/``offset``.
        """
        conditions = ["owner_id = ?"]
        params: list[Any] = [owner_id]

        if review_state is not None:
            conditions.append("review_state = ?")
            params.append(review_state)

        if definition_id is not None:
            conditions.append("definition_id = ?")
            params.append(definition_id)

        where_clause = f"WHERE {' AND '.join(conditions)}"
        params.extend([limit, offset])

        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"SELECT * FROM automation_results {where_clause} "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                params,
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields=self._AUTOMATION_RESULT_JSON_FIELDS
            )

    def count_unread_results(self, owner_id: str) -> int:
        """Count unread results for an owner (spec §4's inbox badge)."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM automation_results "
                "WHERE owner_id = ? AND review_state = 'unread'",
                (owner_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def get_automation_result(self, result_id: str) -> Optional[dict[str, Any]]:
        """Fetch an automation result by local id.

        Args:
            result_id: Local ``automation_results.id`` to look up.

        Returns:
            The result row as a dict (JSON fields already decoded), or
            ``None`` if no row matches ``result_id``.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM automation_results WHERE id = ?", (result_id,)
            )
            return self._row_to_dict(
                cursor.fetchone(), json_fields=self._AUTOMATION_RESULT_JSON_FIELDS
            )

    def update_result_review(
        self,
        result_id: str,
        review_state: str,
        review_note: str | None = None,
        reviewed_by: str | None = None,
        *,
        pending_mutation: dict[str, Any] | None = None,
    ) -> bool:
        """Set a result's review state; returns False for an unknown id.

        When ``pending_mutation`` is given, its
        ``automation_result_review`` mutation is inserted into
        ``pending_mutations`` in the SAME transaction as the review
        UPDATE, so a crash between the two can never leave a local
        review recorded without the outbox row that pushes it (or vice
        versa). The dict mirrors ``record_pending_mutation``'s
        parameters: ``local_id``, ``primitive``, ``owner_id``,
        ``payload`` -- an ``idempotency_key`` is generated into the
        payload if one isn't already present, same as the standalone
        method.
        """
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE automation_results
                SET review_state = ?, review_note = ?, reviewed_by = ?,
                    reviewed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (review_state, review_note, reviewed_by, now_iso, now_iso, result_id),
            )
            if cursor.rowcount == 0:
                return False

            if pending_mutation is not None:
                stored_payload = dict(pending_mutation["payload"])
                if not stored_payload.get("idempotency_key"):
                    stored_payload["idempotency_key"] = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pending_mutations
                    (local_id, primitive, owner_id, payload, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        pending_mutation["local_id"],
                        pending_mutation["primitive"],
                        pending_mutation["owner_id"],
                        self._to_json(stored_payload),
                        now_iso,
                    ),
                )
            return True

    # ------------------------------------------------------------------
    # Server-mirror upserts (schedules-handoff PR-3)
    # ------------------------------------------------------------------

    #: Primitive name pending `automation_result_review` mutations are
    #: stored under (matches the SyncEngine module constant of the same
    #: value -- see sync_engine.py's `_RESULT_REVIEW_PRIMITIVE`).
    _RESULT_REVIEW_PRIMITIVE = "automation_result_review"

    #: Result columns that may be copied verbatim from a server item on
    #: insert, beyond id/server_id/owner_id (handled separately).
    _AUTOMATION_RESULT_INSERT_FIELDS = _AUTOMATION_RESULT_COLUMNS | {
        "definition_id", "run_id", "kind", "title", "summary", "dedupe_key",
        "created_at",
    }

    #: Result fields a server-mirror update is allowed to touch on an
    #: existing row -- review state only (spec §5: "results sync down,
    #: review pushes up").
    _AUTOMATION_RESULT_REVIEW_FIELDS = {
        "review_state", "reviewed_at", "reviewed_by", "review_note", "updated_at",
    }

    def upsert_automation_definitions_from_server(
        self, owner_id: str, items: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Server-wins mirror of automation definitions pulled from the server.

        Matches local rows by ``(owner_id, server_id)``. Absent -> insert a
        new mirror row (server ``id`` becomes local ``server_id``; local
        ``id`` is a fresh UUID). Present -> every server-carried field is
        written EXCEPT ``transfer_state``: a server payload must never
        clear a local transfer marker (spec-2026-08-31-schedules-handoff-
        parity.md §6 parked finding). Archived lifecycle mirrors like any
        other field -- rows are never deleted here.

        Returns:
            ``{"inserted": n, "updated": n}``.
        """
        inserted = 0
        updated = 0
        with self.transaction() as conn:
            for item in items:
                server_id = item.get("id")
                if not server_id:
                    continue

                fields: dict[str, Any] = {
                    key: item[key]
                    for key in self._AUTOMATION_DEFINITION_COLUMNS
                    if key in item and key not in {"id", "server_id", "owner_id"}
                }
                # §6 parked finding: transfer_state is a local-only marker
                # a server mirror must never overwrite, even if a payload
                # somehow carried one.
                fields.pop("transfer_state", None)

                existing = conn.execute(
                    "SELECT id FROM automation_definitions "
                    "WHERE owner_id = ? AND server_id = ?",
                    (owner_id, server_id),
                ).fetchone()

                if existing is None:
                    local_id = str(uuid.uuid4())
                    now_iso = self._to_utc_iso(datetime.now(timezone.utc))
                    insert_fields = dict(fields)
                    insert_fields["id"] = local_id
                    insert_fields["server_id"] = server_id
                    insert_fields["owner_id"] = owner_id
                    insert_fields.setdefault("family", "recurring_question")
                    insert_fields.setdefault("name", "Untitled automation")
                    insert_fields.setdefault("lifecycle", "configured")
                    insert_fields.setdefault("health", "execution_unavailable")
                    insert_fields.setdefault("version", 1)
                    insert_fields.setdefault("created_at", now_iso)
                    insert_fields.setdefault("updated_at", now_iso)
                    serialized = self._serialize_definition_fields(insert_fields)
                    self._validate_sql_identifiers(list(serialized.keys()))
                    columns = ", ".join(serialized.keys())
                    placeholders = ", ".join(["?"] * len(serialized))
                    conn.execute(
                        f"INSERT INTO automation_definitions ({columns}) "
                        f"VALUES ({placeholders})",
                        list(serialized.values()),
                    )
                    inserted += 1
                else:
                    if not fields:
                        continue
                    serialized = self._serialize_definition_fields(fields)
                    self._validate_sql_identifiers(list(serialized.keys()))
                    updates = ", ".join(f"{key} = ?" for key in serialized)
                    conn.execute(
                        f"UPDATE automation_definitions SET {updates} WHERE id = ?",
                        [*serialized.values(), existing["id"]],
                    )
                    updated += 1
        return {"inserted": inserted, "updated": updated}

    def _serialize_definition_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Apply the same JSON/datetime conversion `update_automation_definition` uses.

        Also strips a `config.scope.resolved_sources` key when present
        (task 6 fix-round finding): both server-mirror write paths --
        `adopt_server_definition_identity` and
        `upsert_automation_definitions_from_server` -- route every
        definition write through here, so this is the single choke point
        that covers both. `normalize_recurring_question_scope`'s
        `"all_searchable_library"` branch (`recurring_question_scope.py`)
        computes `resolved_sources` fresh on every call as an OUTPUT
        projection, never an accepted input field -- persisting a
        server-echoed copy of it would make any later re-normalization of
        this row's scope (a scheduled dispatch, the sources-readable
        health check) report a spurious "unsupported field" error and
        degrade every run. Same bug, same fix shape as the local-authoring
        path's (`SchedulingService._definition_db_fields_from_preview`),
        which is a separate write path (`create_automation_definition`/
        `update_automation_definition` serialize inline, never through
        this method) and still needs its own strip.
        """
        config = fields.get("config")
        if isinstance(config, dict):
            scope = config.get("scope")
            if isinstance(scope, dict) and "resolved_sources" in scope:
                fields = dict(fields)
                fields["config"] = {
                    **config,
                    "scope": {k: v for k, v in scope.items() if k != "resolved_sources"},
                }

        serialized: dict[str, Any] = {}
        for key, value in fields.items():
            if key in self._AUTOMATION_JSON_FIELDS:
                serialized[key] = self._to_json(value)
            elif key in self._DATETIME_FIELDS:
                serialized[key] = self._to_utc_iso(value)
            else:
                serialized[key] = value
        return serialized

    def adopt_server_definition_identity(
        self, local_id: str, server_item: dict[str, Any]
    ) -> bool:
        """Adopt a server identity onto a local row after a create/update push.

        Called by `SyncEngine`'s definition push replay (Task 3) right
        after a `create`/`update` mutation succeeds: sets `server_id` and
        applies every server-wins field from `server_item` (the create/
        update response echo) in ONE transaction, same field set and
        `transfer_state` exclusion as `upsert_automation_definitions_from_
        server`'s existing-row branch -- but keyed by the local row
        directly (``id = local_id``) since the caller already knows which
        local row this mutation came from, rather than matching by
        ``(owner_id, server_id)``.

        Args:
            local_id: The local definition row that pushed the mutation.
            server_item: The definition row echoed back by the server's
                create/update response.

        Returns:
            ``True`` if a local row was found and updated, ``False``
            otherwise (e.g. the row was deleted locally in the meantime).
        """
        server_id = server_item.get("id")
        if not server_id:
            return False

        fields: dict[str, Any] = {
            key: server_item[key]
            for key in self._AUTOMATION_DEFINITION_COLUMNS
            if key in server_item and key not in {"id", "server_id", "owner_id"}
        }
        # §6 parked finding, same as the pull-mirror upsert: transfer_state
        # is a local-only marker a server echo must never overwrite.
        fields.pop("transfer_state", None)
        fields["server_id"] = server_id

        serialized = self._serialize_definition_fields(fields)
        self._validate_sql_identifiers(list(serialized.keys()))
        updates = ", ".join(f"{key} = ?" for key in serialized)
        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE automation_definitions SET {updates} WHERE id = ?",
                [*serialized.values(), local_id],
            )
            return cursor.rowcount > 0

    def upsert_automation_results_from_server(
        self,
        owner_id: str,
        items: list[dict[str, Any]],
        *,
        skip_review_server_ids: frozenset[str] = frozenset(),
    ) -> dict[str, int]:
        """Server-wins mirror of scheduled-task results pulled from the server.

        Matches local rows by ``(owner_id, server_id)``.

        Absent -> insert the full row (local ``id`` is a fresh UUID;
        ``definition_id`` and ``run_id`` are stored exactly as the server
        sent them -- plain TEXT, no local row to resolve to, same
        treatment the spec gives ``run_id``, spec §4.2). A ``dedupe_key``
        UNIQUE conflict against a locally-created row (not yet known to
        the server) is not an error: the insert is skipped and counted --
        the local row keeps ownership until its own push resolves the
        collision.

        Present -> update ONLY the review fields (``review_state``,
        ``reviewed_at``, ``reviewed_by``, ``review_note``, ``updated_at``).
        Two guard layers decide whether that update actually happens
        (Qodo TOCTOU/same-cycle-echo review):

        1. Pending-mutation guard (unpushed reviews): a per-row
           ``pending_mutations`` SELECT is run INSIDE this same write
           transaction, immediately before the row's own UPDATE -- not
           snapshotted once before the loop starts, which left a window
           for a concurrently-recorded review (the review service writes
           via ``to_thread`` while this upsert runs on the event loop) to
           land between the snapshot and this row's write and then get
           clobbered by a stale server payload despite its own mutation
           existing. An unpushed local review outranks the mirror until
           SyncEngine's pushback phase (which runs before this pull) has
           replayed it.
        2. Pushed-this-cycle guard (just-pushed reviews): ``server_id in
           skip_review_server_ids`` skips rows SyncEngine's pushback phase
           already replayed THIS sync cycle. Their pending mutation is
           already gone by the time this pull runs, so guard 1 can't see
           them -- without this second layer, a same-cycle results page
           that still echoes the pre-review server state (server write/
           read-path lag) would revert the review that was just pushed,
           and once the row ages out of the bounded newest-pages pull
           window, no later sync would ever correct it.

        The residual exposure after both layers is only a server that
        lies about its own committed writes (reports success on push,
        then immediately echoes different data back on pull) -- not
        something a client-side guard can detect.

        Returns:
            ``{"inserted": n, "updated": n, "skipped_dedupe": n}``.
        """
        inserted = 0
        updated = 0
        skipped_dedupe = 0
        with self.transaction() as conn:
            for item in items:
                server_id = item.get("id")
                if not server_id:
                    continue

                existing = conn.execute(
                    "SELECT id FROM automation_results "
                    "WHERE owner_id = ? AND server_id = ?",
                    (owner_id, server_id),
                ).fetchone()

                if existing is None:
                    local_id = str(uuid.uuid4())
                    now_iso = self._to_utc_iso(datetime.now(timezone.utc))
                    fields: dict[str, Any] = {
                        key: item[key]
                        for key in self._AUTOMATION_RESULT_INSERT_FIELDS
                        if key in item
                    }
                    fields["id"] = local_id
                    fields["server_id"] = server_id
                    fields["owner_id"] = owner_id
                    fields.setdefault("definition_id", "")
                    fields.setdefault("run_id", "")
                    fields.setdefault("kind", "finding")
                    fields.setdefault("title", "Untitled result")
                    fields.setdefault("summary", "")
                    fields.setdefault("dedupe_key", f"server:{server_id}")
                    fields.setdefault("review_state", "unread")
                    fields.setdefault("answer_mode", "none")
                    fields.setdefault("created_at", now_iso)
                    fields.setdefault("updated_at", now_iso)
                    serialized = self._serialize_result_fields(fields)
                    self._validate_sql_identifiers(list(serialized.keys()))
                    columns = ", ".join(serialized.keys())
                    placeholders = ", ".join(["?"] * len(serialized))
                    try:
                        conn.execute(
                            f"INSERT INTO automation_results ({columns}) "
                            f"VALUES ({placeholders})",
                            list(serialized.values()),
                        )
                    except sqlite3.IntegrityError as exc:
                        if "UNIQUE constraint failed" not in str(exc):
                            raise
                        # (owner_id, dedupe_key) collided with a
                        # locally-created row -- skip, don't overwrite it.
                        skipped_dedupe += 1
                        continue
                    inserted += 1
                else:
                    if server_id in skip_review_server_ids:
                        # Guard 2 (pushed-this-cycle): just replayed by
                        # this same sync's pushback phase -- see the
                        # design comment on this method.
                        continue
                    has_pending_review = conn.execute(
                        """
                        SELECT 1 FROM pending_mutations
                        WHERE local_id = ? AND primitive = ? AND owner_id = ?
                        LIMIT 1
                        """,
                        (existing["id"], self._RESULT_REVIEW_PRIMITIVE, owner_id),
                    ).fetchone()
                    if has_pending_review is not None:
                        # Guard 1 (pending-mutation): checked here, inside
                        # this row's own write transaction, not via a
                        # snapshot taken before the loop started -- see
                        # the design comment on this method.
                        continue
                    review_fields = {
                        key: item[key]
                        for key in self._AUTOMATION_RESULT_REVIEW_FIELDS
                        if key in item
                    }
                    if not review_fields:
                        continue
                    serialized = self._serialize_result_fields(review_fields)
                    self._validate_sql_identifiers(list(serialized.keys()))
                    updates = ", ".join(f"{key} = ?" for key in serialized)
                    conn.execute(
                        f"UPDATE automation_results SET {updates} WHERE id = ?",
                        [*serialized.values(), existing["id"]],
                    )
                    updated += 1
        return {
            "inserted": inserted,
            "updated": updated,
            "skipped_dedupe": skipped_dedupe,
        }

    def _serialize_result_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Apply the same JSON conversion `create_automation_result` uses.

        Datetime-shaped fields (``reviewed_at``, ``created_at``, ...) are
        passed through raw, mirroring `create_automation_result`'s own
        loop (which only special-cases actual ``datetime`` instances) --
        server items already carry UTC ISO-8601 strings.
        """
        serialized: dict[str, Any] = {}
        for key, value in fields.items():
            if key in self._AUTOMATION_RESULT_JSON_FIELDS:
                serialized[key] = json.dumps(value)
            elif isinstance(value, datetime):
                serialized[key] = self._to_utc_iso(value)
            else:
                serialized[key] = value
        return serialized

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def get_sync_mapping_by_server_id(
        self,
        server_id: str,
        primitive: str,
        owner_id: str,
    ) -> Optional[dict[str, Any]]:
        """Look up a sync mapping by server-side identifier.

        Returns the matching mapping row, or ``None`` if no mapping exists.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM sync_mapping
                WHERE server_id = ? AND primitive = ? AND owner_id = ?
                """,
                (server_id, primitive, owner_id),
            )
            return self._row_to_dict(cursor.fetchone())

    def get_sync_mapping_by_local_id(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> Optional[dict[str, Any]]:
        """Look up a sync mapping by local identifier.

        Returns the matching mapping row, or ``None`` if no mapping exists.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM sync_mapping
                WHERE local_id = ? AND primitive = ? AND owner_id = ?
                """,
                (local_id, primitive, owner_id),
            )
            return self._row_to_dict(cursor.fetchone())

    def set_sync_mapping(
        self,
        local_id: str,
        server_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        """Create or replace the mapping between a local and server record."""
        with self.transaction() as conn:
            self._set_sync_mapping_conn(
                conn, local_id, server_id, primitive, owner_id
            )

    def delete_sync_mapping(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        """Remove the sync mapping for a local record."""
        with self.transaction() as conn:
            self._delete_sync_mapping_conn(conn, local_id, primitive, owner_id)

    def get_sync_state(self, owner_id: str) -> Optional[dict[str, Any]]:
        """Fetch the sync state row for ``owner_id``, or ``None`` if absent."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM sync_state WHERE owner_id = ?",
                (owner_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(row, json_fields={"sync_errors"})

    def update_sync_state(self, owner_id: str, **kwargs: Any) -> None:
        """Upsert per-owner sync state.

        Supported fields: ``last_pull_at``, ``last_push_at``,
        ``last_conflict_at``, ``sync_errors``. The ``owner_id`` is always
        stored; other fields are updated if provided.
        """
        with self.transaction() as conn:
            self._update_sync_state_conn(conn, owner_id, **kwargs)

    # ------------------------------------------------------------------
    # Pending mutations
    # ------------------------------------------------------------------

    def _insert_pending_mutation_conn(
        self,
        conn: sqlite3.Connection,
        *,
        local_id: str,
        primitive: str,
        owner_id: str,
        payload: dict[str, Any],
        now_iso: str,
    ) -> None:
        """Insert one pending-mutation row on an already-open transaction.

        Shared by callers that record a mutation atomically alongside
        another write -- same ``INSERT OR REPLACE`` + auto-filled
        ``idempotency_key`` behavior as the standalone
        ``record_pending_mutation``, factored out of ``update_result_
        review``'s inline precedent so ``create_automation_definition``/
        ``update_automation_definition`` can reuse it (Task 4).
        """
        stored_payload = dict(payload)
        if not stored_payload.get("idempotency_key"):
            stored_payload["idempotency_key"] = str(uuid.uuid4())
        conn.execute(
            """
            INSERT OR REPLACE INTO pending_mutations
            (local_id, primitive, owner_id, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (local_id, primitive, owner_id, self._to_json(stored_payload), now_iso),
        )

    def record_pending_mutation(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Store a local mutation waiting to be pushed to the server.

        ``payload`` typically contains an ``action`` key (``create``,
        ``update``, or ``delete``) plus any fields required by the server
        client. An ``idempotency_key`` is generated and persisted in the
        payload if one is not already provided. Existing pending mutations
        for the same local id/primitive/owner are replaced.
        """
        stored_payload = dict(payload)
        if (
            "idempotency_key" not in stored_payload
            or not stored_payload["idempotency_key"]
        ):
            stored_payload["idempotency_key"] = str(uuid.uuid4())

        now = datetime.now(timezone.utc)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_mutations
                (local_id, primitive, owner_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    local_id,
                    primitive,
                    owner_id,
                    self._to_json(stored_payload),
                    self._to_utc_iso(now),
                ),
            )

    def get_pending_mutations(
        self,
        owner_id: str,
        primitive: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return pending mutations for ``owner_id``, optionally filtered by primitive."""
        conditions = ["owner_id = ?"]
        params: list[Any] = [owner_id]
        if primitive is not None:
            conditions.append("primitive = ?")
            params.append(primitive)

        where_clause = " AND ".join(conditions)
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM pending_mutations
                WHERE {where_clause}
                ORDER BY created_at
                """,
                params,
            )
            return self._rows_to_dicts(cursor.fetchall(), json_fields={"payload"})

    def get_pending_mutation_for_local_id(
        self, local_id: str, primitive: str
    ) -> Optional[dict[str, Any]]:
        """Return the pending mutation for ``local_id``/``primitive``, if
        any, regardless of which ``owner_id`` it is filed under.

        `get_pending_mutations` above requires the caller to already know
        `owner_id` -- fine for its own callers, which always read it off
        the row they already have. A `to_server_failed` row's UI display
        (schedules-handoff PR-5, Task 7 fix round finding 3) has no such
        row-derived owner_id to key off: the mutation was recorded under
        WHATEVER server was active at the time of the (failed) attempt,
        which is not necessarily the CURRENTLY active server, so guessing
        via "today's active server" silently misses the mutation after a
        server switch. Reading the row directly sidesteps the guess
        entirely -- its own `owner_id` column is the actual answer.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM pending_mutations
                WHERE local_id = ? AND primitive = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (local_id, primitive),
            )
            return self._row_to_dict(
                cursor.fetchone(), json_fields={"payload"}
            )

    def delete_pending_mutation(self, mutation_id: int) -> None:
        """Delete a pending mutation by its row id."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM pending_mutations WHERE id = ?", (mutation_id,))

    def delete_pending_mutation_for_record(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        """Delete any pending mutation matching a local record identifier."""
        with self.transaction() as conn:
            conn.execute(
                """
                DELETE FROM pending_mutations
                WHERE local_id = ? AND primitive = ? AND owner_id = ?
                """,
                (local_id, primitive, owner_id),
            )

    # ------------------------------------------------------------------
    # Tombstones
    # ------------------------------------------------------------------

    def record_tombstone(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        """Record that a local record was deleted and the delete must be pushed."""
        now = datetime.now(timezone.utc)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sync_tombstones
                (local_id, primitive, owner_id, deleted_at, pushed_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (local_id, primitive, owner_id, self._to_utc_iso(now)),
            )

    def get_tombstones(
        self,
        owner_id: str,
        primitive: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return tombstones for ``owner_id``, optionally filtered by primitive."""
        conditions = ["owner_id = ?"]
        params: list[Any] = [owner_id]
        if primitive is not None:
            conditions.append("primitive = ?")
            params.append(primitive)

        where_clause = " AND ".join(conditions)
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM sync_tombstones
                WHERE {where_clause}
                ORDER BY deleted_at
                """,
                params,
            )
            return self._rows_to_dicts(cursor.fetchall())

    def get_tombstone(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> Optional[dict[str, Any]]:
        """Return a single tombstone row if it exists."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM sync_tombstones
                WHERE local_id = ? AND primitive = ? AND owner_id = ?
                """,
                (local_id, primitive, owner_id),
            )
            return self._row_to_dict(cursor.fetchone())

    def delete_tombstone(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
    ) -> None:
        """Remove a tombstone after its delete has been pushed to the server."""
        with self.transaction() as conn:
            self._delete_tombstone_conn(conn, local_id, primitive, owner_id)

    # ------------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------------

    def record_conflict(
        self,
        local_id: str,
        primitive: str,
        owner_id: str,
        server_state: dict[str, Any],
        local_state: dict[str, Any],
    ) -> str:
        """Record a sync conflict between server and local state.

        Returns the generated conflict id.
        """
        with self.transaction() as conn:
            return self._record_conflict_conn(
                conn,
                local_id=local_id,
                primitive=primitive,
                owner_id=owner_id,
                server_state=server_state,
                local_state=local_state,
            )

    def get_conflicts(
        self,
        owner_id: str,
        primitive: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return unresolved conflicts for ``owner_id``, optionally filtered by primitive."""
        conditions = ["owner_id = ?", "resolved_at IS NULL"]
        params: list[Any] = [owner_id]
        if primitive is not None:
            conditions.append("primitive = ?")
            params.append(primitive)

        where_clause = " AND ".join(conditions)
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM sync_conflicts
                WHERE {where_clause}
                ORDER BY created_at
                """,
                params,
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields={"server_state", "local_state"}
            )

    def get_conflict_by_id(self, conflict_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single conflict row by id."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM sync_conflicts WHERE id = ?",
                (conflict_id,),
            )
            return self._row_to_dict(
                cursor.fetchone(), json_fields={"server_state", "local_state"}
            )

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
    ) -> bool:
        """Mark a conflict as resolved with the given resolution value.

        Returns ``True`` if a row was updated.
        """
        now = datetime.now(timezone.utc)
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE sync_conflicts
                SET resolved_at = ?, resolution = ?
                WHERE id = ? AND resolved_at IS NULL
                """,
                (self._to_utc_iso(now), resolution, conflict_id),
            )
            return cursor.rowcount > 0

    def increment_conflict_retry_count(self, conflict_id: str) -> bool:
        """Increment the retry count on a conflict."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                """
                UPDATE sync_conflicts
                SET retry_count = retry_count + 1
                WHERE id = ?
                """,
                (conflict_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
