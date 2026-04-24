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
        self._preferences_table = "client_notification_preferences"
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

                CREATE TABLE IF NOT EXISTS client_notification_preferences (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    delivery_enabled INTEGER NOT NULL DEFAULT 1,
                    muted_categories TEXT NOT NULL DEFAULT '[]',
                    muted_severities TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                INSERT OR IGNORE INTO client_notification_preferences (
                    id,
                    delivery_enabled,
                    muted_categories,
                    muted_severities
                )
                VALUES (1, 1, '[]', '[]');
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

    def list_notifications(
        self,
        *,
        limit: int = 100,
        include_dismissed: bool = False,
        category: str | None = None,
        severity: str | None = None,
        source_backend: str | None = None,
        source_entity_kind: str | None = None,
        source_entity_id: str | None = None,
        is_read: bool | None = None,
    ) -> list[dict[str, Any]]:
        query = f"SELECT * FROM {self._notifications_table}"
        clauses: list[str] = []
        params: list[Any] = []
        if not include_dismissed:
            clauses.append("is_dismissed = 0")
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        if source_backend is not None:
            clauses.append("source_backend = ?")
            params.append(source_backend)
        if source_entity_kind is not None:
            clauses.append("source_entity_kind = ?")
            params.append(source_entity_kind)
        if source_entity_id is not None:
            clauses.append("source_entity_id = ?")
            params.append(source_entity_id)
        if is_read is not None:
            clauses.append("is_read = ?")
            params.append(1 if is_read else 0)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))

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

    @staticmethod
    def _normalize_preference_values(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
        normalized: set[str] = set()
        for value in values or []:
            text = str(value).strip().lower()
            if text:
                normalized.add(text)
        return sorted(normalized)

    @staticmethod
    def _row_to_preferences(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {
                "delivery_enabled": True,
                "muted_categories": [],
                "muted_severities": [],
            }
        data = dict(row)
        return {
            "delivery_enabled": bool(data.get("delivery_enabled", 1)),
            "muted_categories": sorted(json.loads(data.get("muted_categories") or "[]")),
            "muted_severities": sorted(json.loads(data.get("muted_severities") or "[]")),
        }

    def get_preferences(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {self._preferences_table} WHERE id = 1"
            ).fetchone()
        return self._row_to_preferences(row)

    def update_preferences(
        self,
        *,
        delivery_enabled: bool | None = None,
        muted_categories: list[str] | tuple[str, ...] | set[str] | None = None,
        muted_severities: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any]:
        current = self.get_preferences()
        next_delivery_enabled = (
            current["delivery_enabled"] if delivery_enabled is None else bool(delivery_enabled)
        )
        next_muted_categories = (
            current["muted_categories"]
            if muted_categories is None
            else self._normalize_preference_values(muted_categories)
        )
        next_muted_severities = (
            current["muted_severities"]
            if muted_severities is None
            else self._normalize_preference_values(muted_severities)
        )
        with self._get_connection() as conn:
            conn.execute(
                f"""
                UPDATE {self._preferences_table}
                SET delivery_enabled = ?,
                    muted_categories = ?,
                    muted_severities = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (
                    1 if next_delivery_enabled else 0,
                    json.dumps(next_muted_categories, ensure_ascii=True),
                    json.dumps(next_muted_severities, ensure_ascii=True),
                ),
            )
            conn.commit()
        return self.get_preferences()

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            raise LookupError("Notification row not found")

        data = dict(row)
        data["is_read"] = bool(data["is_read"])
        data["is_dismissed"] = bool(data["is_dismissed"])
        if data.get("payload") is not None:
            data["payload"] = json.loads(data["payload"])
        return data

    def list(
        self,
        *,
        limit: int = 100,
        include_dismissed: bool = False,
        category: str | None = None,
        severity: str | None = None,
        source_backend: str | None = None,
        source_entity_kind: str | None = None,
        source_entity_id: str | None = None,
        is_read: bool | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_notifications(
            limit=limit,
            include_dismissed=include_dismissed,
            category=category,
            severity=severity,
            source_backend=source_backend,
            source_entity_kind=source_entity_kind,
            source_entity_id=source_entity_id,
            is_read=is_read,
        )

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
