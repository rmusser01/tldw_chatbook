from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Union

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.config import get_notifications_db_path


class ClientNotificationsDB(BaseDB):
    def __init__(self, db_path: Union[str, Path, None] = None, client_id: str = "default") -> None:
        self._notifications_table = "client_notifications"
        super().__init__(db_path or get_notifications_db_path(), client_id)

    def _initialize_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS client_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'info',
                    source_backend TEXT NOT NULL,
                    source_entity_id TEXT,
                    source_entity_kind TEXT,
                    payload TEXT,
                    is_read INTEGER NOT NULL DEFAULT 0,
                    is_dismissed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    read_at TEXT,
                    dismissed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_client_notifications_created_at
                ON client_notifications(created_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_client_notifications_queue_state
                ON client_notifications(is_dismissed, is_read, created_at DESC);
                """
            )

    def insert(
        self,
        *,
        category: str,
        title: str,
        message: str,
        severity: str = "info",
        source_backend: str,
        source_entity_id: str | None = None,
        source_entity_kind: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload_json = json.dumps(payload, ensure_ascii=True) if payload is not None else None
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO {self._notifications_table} (
                    category,
                    title,
                    message,
                    severity,
                    source_backend,
                    source_entity_id,
                    source_entity_kind,
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
                    source_entity_id,
                    source_entity_kind,
                    payload_json,
                ),
            )
            notification_id = cursor.lastrowid
            row = conn.execute(
                f"SELECT * FROM {self._notifications_table} WHERE id = ?",
                (notification_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def list_notifications(self, *, limit: int = 100, include_dismissed: bool = False) -> list[dict[str, Any]]:
        query = f"SELECT * FROM {self._notifications_table}"
        if include_dismissed:
            query += " ORDER BY id DESC LIMIT ?"
        else:
            query += " WHERE is_dismissed = 0 ORDER BY id DESC LIMIT ?"
        params: list[Any] = [limit]

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def mark_read(self, notification_id: int, *, is_read: bool) -> bool:
        read_at = "CURRENT_TIMESTAMP" if is_read else None
        query = f"""
            UPDATE {self._notifications_table}
            SET is_read = ?, read_at = {read_at if is_read else "NULL"}
            WHERE id = ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (1 if is_read else 0, notification_id))
            conn.commit()
            return cursor.rowcount > 0

    def dismiss_notification(self, notification_id: int, *, is_dismissed: bool) -> bool:
        dismissed_at = "CURRENT_TIMESTAMP" if is_dismissed else None
        query = f"""
            UPDATE {self._notifications_table}
            SET is_dismissed = ?, dismissed_at = {dismissed_at if is_dismissed else "NULL"}
            WHERE id = ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (1 if is_dismissed else 0, notification_id))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            raise LookupError("Notification row not found")

        data = dict(row)
        data["is_read"] = bool(data["is_read"])
        data["is_dismissed"] = bool(data["is_dismissed"])
        if data.get("payload") is not None:
            data["payload"] = json.loads(data["payload"])
        return data

    def list(self, *, limit: int = 100, include_dismissed: bool = False) -> list[dict[str, Any]]:
        return self.list_notifications(limit=limit, include_dismissed=include_dismissed)

    def mark_read_notification(self, notification_id: int, *, is_read: bool) -> bool:
        return self.mark_read(notification_id, is_read=is_read)

    def dismiss(self, notification_id: int, *, is_dismissed: bool) -> bool:
        return self.dismiss_notification(notification_id, is_dismissed=is_dismissed)

    def insert_notification(
        self,
        *,
        category: str,
        title: str,
        message: str,
        severity: str = "info",
        source_backend: str,
        source_entity_id: str | None = None,
        source_entity_kind: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.insert(
            category=category,
            title=title,
            message=message,
            severity=severity,
            source_backend=source_backend,
            source_entity_id=source_entity_id,
            source_entity_kind=source_entity_kind,
            payload=payload,
        )
