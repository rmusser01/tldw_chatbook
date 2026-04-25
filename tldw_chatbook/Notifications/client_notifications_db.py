"""SQLite-backed local notification inbox for Chatbook-owned events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..DB.base_db import BaseDB


class ClientNotificationsDB(BaseDB):
    """Dedicated local queue/inbox store for client notifications."""

    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path, client_id: str = "default"):
        self._memory_conn: sqlite3.Connection | None = None
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        if getattr(self, "is_memory_db", False):
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:")
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn
        return super()._get_connection()

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def _initialize_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

                CREATE TABLE IF NOT EXISTS client_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'information',
                    source_backend TEXT,
                    source_entity_kind TEXT,
                    source_entity_id TEXT,
                    payload TEXT NOT NULL DEFAULT '{}',
                    is_read INTEGER NOT NULL DEFAULT 0,
                    is_dismissed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    read_at TEXT,
                    dismissed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_client_notifications_inbox
                    ON client_notifications(is_dismissed, created_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_client_notifications_source
                    ON client_notifications(source_backend, source_entity_kind, source_entity_id);
                """
            )

    def insert_notification(
        self,
        *,
        category: str,
        title: str,
        message: str,
        severity: str = "information",
        source_backend: str | None = None,
        source_entity_kind: str | None = None,
        source_entity_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert a notification and return the normalized row."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO client_notifications (
                    category,
                    title,
                    message,
                    severity,
                    source_backend,
                    source_entity_kind,
                    source_entity_id,
                    payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    category,
                    title,
                    message,
                    severity,
                    source_backend,
                    source_entity_kind,
                    source_entity_id,
                    json.dumps(dict(payload or {}), sort_keys=True),
                ),
            )
            notification_id = int(cursor.lastrowid)
            conn.commit()
        return self.get_notification(notification_id)

    def get_notification(self, notification_id: int) -> dict[str, Any]:
        """Return a notification by id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM client_notifications WHERE id = ?",
                (notification_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Notification not found: {notification_id}")
        return self._normalize_row(row)

    def list_notifications(
        self,
        *,
        limit: int = 100,
        include_dismissed: bool = False,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """List inbox notifications newest-first."""
        clauses = []
        params: list[Any] = []
        if not include_dismissed:
            clauses.append("is_dismissed = 0")
        if category:
            clauses.append("category = ?")
            params.append(category)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(int(limit), 1))
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM client_notifications
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._normalize_row(row) for row in rows]

    def mark_read(self, notification_id: int, *, is_read: bool) -> bool:
        """Mark a notification read or unread."""
        read_at = self._now_iso() if is_read else None
        return self._update_flags(
            notification_id,
            is_read=int(bool(is_read)),
            read_at=read_at,
        )

    def dismiss_notification(self, notification_id: int, *, is_dismissed: bool) -> bool:
        """Dismiss or restore a notification from the inbox."""
        dismissed_at = self._now_iso() if is_dismissed else None
        return self._update_flags(
            notification_id,
            is_dismissed=int(bool(is_dismissed)),
            dismissed_at=dismissed_at,
        )

    def _update_flags(self, notification_id: int, **fields: Any) -> bool:
        assignments = ", ".join(f"{field} = ?" for field in fields)
        params = list(fields.values())
        params.append(notification_id)
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE client_notifications SET {assignments} WHERE id = ?",
                tuple(params),
            )
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_row(row: sqlite3.Row) -> dict[str, Any]:
        payload_text = row["payload"] or "{}"
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {}

        return {
            "id": row["id"],
            "category": row["category"],
            "title": row["title"],
            "message": row["message"],
            "severity": row["severity"],
            "source_backend": row["source_backend"],
            "source_entity_kind": row["source_entity_kind"],
            "source_entity_id": row["source_entity_id"],
            "payload": payload if isinstance(payload, dict) else {},
            "is_read": bool(row["is_read"]),
            "is_dismissed": bool(row["is_dismissed"]),
            "created_at": row["created_at"],
            "read_at": row["read_at"],
            "dismissed_at": row["dismissed_at"],
        }
