"""Scheduled tasks database layer.

ScheduledTasksDB extends BaseDB and provides CRUD operations for reminder
tasks, plus the shared schema used by automation definitions, previews, audit
events, and sync helpers.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.Scheduling.db.schema import CREATE_SCHEMA_SQL


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

    _DATETIME_FIELDS = {
        "run_at",
        "next_run_at",
        "last_run_at",
        "missed_at",
        "created_at",
        "updated_at",
    }

    _RESERVED_FIELDS = {"id"}

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str = "default",
        check_integrity_on_startup: bool = False,
    ):
        super().__init__(db_path, client_id, check_integrity_on_startup)

    def _initialize_schema(self) -> None:
        """Create tables, indexes, and schema version row."""
        with closing(self._get_connection()) as conn:
            conn.executescript(CREATE_SCHEMA_SQL)
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self._CURRENT_SCHEMA_VERSION,),
            )
            conn.commit()

    def get_schema_version(self) -> int:
        """Return the currently recorded schema version."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

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
            dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()

        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()

        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"Invalid ISO-8601 datetime string: {value!r}") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()

        raise TypeError(
            f"Expected datetime, date, or str, got {type(value).__name__}"
        )

    @classmethod
    def _row_to_dict(cls, row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        """Convert a sqlite3.Row to a plain dictionary with booleans restored."""
        if row is None:
            return None
        result: dict[str, Any] = dict(row)
        if "enabled" in result:
            result["enabled"] = bool(result["enabled"])
        return result

    @classmethod
    def _validate_reminder_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Validate kwargs for create/update; raises ValueError on bad keys."""
        for key in kwargs:
            if key in cls._RESERVED_FIELDS:
                raise ValueError(f"Field {key!r} is reserved and cannot be set via kwargs")
            if key not in cls._REMINDER_TASK_COLUMNS:
                raise ValueError(f"Unknown reminder task field: {key!r}")

    # ------------------------------------------------------------------
    # Reminder tasks
    # ------------------------------------------------------------------

    def create_reminder_task(
        self, owner_id: str, title: str, **kwargs: Any
    ) -> str:
        """Create a reminder task and return its generated local UUID."""
        self._validate_reminder_kwargs(kwargs)

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

        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))

        with closing(self._get_connection()) as conn:
            conn.execute(
                f"INSERT INTO reminder_tasks ({columns}) VALUES ({placeholders})",
                list(fields.values()),
            )
            conn.commit()

        logger.debug(f"Created reminder task {task_id} for owner {owner_id}")
        return task_id

    def get_reminder_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Fetch a reminder task by local id."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "SELECT * FROM reminder_tasks WHERE id = ?", (task_id,)
            )
            return self._row_to_dict(cursor.fetchone())

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
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def update_reminder_task(self, task_id: str, **kwargs: Any) -> bool:
        """Update reminder task fields. Returns True if a row was changed."""
        if not kwargs:
            return False

        self._validate_reminder_kwargs(kwargs)

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

        updates.append("updated_at = ?")
        params.append(self._to_utc_iso(datetime.now(timezone.utc)))
        params.append(task_id)

        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                f"UPDATE reminder_tasks SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_reminder_task(self, task_id: str) -> bool:
        """Delete a reminder task by local id."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("DELETE FROM reminder_tasks WHERE id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0

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
            return [self._row_to_dict(row) for row in cursor.fetchall()]
