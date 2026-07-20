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
from datetime import date, datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterator, Optional, Union, cast
from zoneinfo import ZoneInfo

from croniter import croniter
from loguru import logger

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.DB.sql_validation import validate_identifier


class ScheduledTasksDB(BaseDB):
    """Database operations for scheduled tasks and reminders."""

    _CURRENT_SCHEMA_VERSION = 1

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
    }

    _AUDIT_JSON_FIELDS = {
        "before",
        "after",
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

    def _initialize_schema(self) -> None:
        """Create tables, indexes, and schema version row."""
        from tldw_chatbook.Scheduling.db.migrations.v0_to_v1 import migrate

        migrate(self)

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
    ) -> list[dict[str, Any]]:
        """List reminder tasks with optional filters."""
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
        """Return enabled reminders whose next_run_at is at or before ``now``."""
        now_iso = self._to_utc_iso(now)
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reminder_tasks
                WHERE enabled = 1
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                ORDER BY next_run_at
                """,
                (now_iso,),
            )
            return self._rows_to_dicts(
                cursor.fetchall(), json_fields=self._REMINDER_JSON_FIELDS
            )

    def mark_reminder_dispatched(
        self,
        task_id: str,
        now: datetime,
        success: bool = True,
    ) -> None:
        """Update a reminder after dispatch so it is not immediately redispatched.

        For ``one_time`` reminders the task is disabled and ``next_run_at`` is
        cleared. For ``recurring`` reminders the next occurrence is computed from
        the stored cron expression and timezone.
        """
        row = self.get_reminder_task(task_id)
        if row is None:
            return

        fields: dict[str, Any] = {
            "last_run_at": now,
            "last_status": "completed" if success else "missed",
            "updated_at": now,
        }

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

    # ------------------------------------------------------------------
    # Automation definitions
    # ------------------------------------------------------------------

    def create_automation_definition(
        self, owner_id: str, family: str, name: str, **kwargs: Any
    ) -> str:
        """Create an automation definition and return its generated local UUID.

        Defaults ``lifecycle`` to ``configured`` and ``health`` to
        ``execution_unavailable`` when not provided. JSON fields are serialized
        and datetime fields are converted to UTC ISO-8601 strings.
        """
        self._validate_kwargs(
            kwargs, self._AUTOMATION_DEFINITION_COLUMNS, "automation definition"
        )

        definition_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        fields: dict[str, Any] = {
            "id": definition_id,
            "owner_id": owner_id,
            "family": family,
            "name": name,
            "lifecycle": "configured",
            "health": "execution_unavailable",
            "version": 1,
            "created_at": self._to_utc_iso(now),
            "updated_at": self._to_utc_iso(now),
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

    def update_automation_definition(self, definition_id: str, **kwargs: Any) -> bool:
        """Update automation-definition fields. Returns True if a row changed.

        The ``version`` column is automatically incremented for optimistic
        locking; any ``version`` value supplied in kwargs is ignored.
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
        updates.append("version = version + 1")
        updates.append("updated_at = ?")
        params.append(self._to_utc_iso(datetime.now(timezone.utc)))
        params.append(definition_id)

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE automation_definitions SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            return cursor.rowcount > 0

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
