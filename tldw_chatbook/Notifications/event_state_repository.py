"""SQLite-backed durable state for normalized event observation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.runtime_policy.server_parity_models import (
    EventCursor,
    EventDedupeKey,
    NotificationPresentationRecord,
    NormalizedEventRecord,
    SourceAuthority,
)

from .event_cursor_store import (
    CursorAdvanceResult,
    CursorAdvanceStatus,
    DedupeResult,
)


class _NoExpectedCursor:
    pass


_NO_EXPECTED_CURSOR = _NoExpectedCursor()


@dataclass(frozen=True, slots=True)
class EventStateRecordResult:
    event_key: str
    is_duplicate: bool
    cursor: EventCursor


@dataclass(frozen=True, slots=True)
class EventRetentionPolicy:
    source_authority: SourceAuthority
    server_profile_id: str | None
    authenticated_principal_id: str | None
    stream_name: str
    stream_instance_id: str
    max_age_days: int = 30
    max_count: int = 10_000


class EventStateRepository(BaseDB):
    """Durable event rows, dedupe records, cursors, and presentation watermarks."""

    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path, client_id: str = "default") -> None:
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

                CREATE TABLE IF NOT EXISTS event_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT,
                    authenticated_principal_id TEXT,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    entity_ref TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    event_id TEXT,
                    server_cursor TEXT,
                    emitted_at TEXT,
                    received_at TEXT,
                    transport_type TEXT NOT NULL,
                    payload_kind TEXT,
                    payload TEXT NOT NULL,
                    stored_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_event_records_scope
                    ON event_records(
                        source_authority,
                        server_profile_id,
                        authenticated_principal_id,
                        stream_name,
                        stream_instance_id,
                        id
                    );

                CREATE TABLE IF NOT EXISTS event_dedupe_records (
                    dedupe_key TEXT PRIMARY KEY,
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT,
                    authenticated_principal_id TEXT,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS event_processed_cursors (
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    cursor TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source_authority,
                        server_profile_id,
                        authenticated_principal_id,
                        stream_name,
                        stream_instance_id
                    )
                );

                CREATE TABLE IF NOT EXISTS event_presented_high_water (
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    cursor TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source_authority,
                        server_profile_id,
                        authenticated_principal_id,
                        stream_name,
                        stream_instance_id
                    )
                );

                CREATE TABLE IF NOT EXISTS event_presentations (
                    event_key TEXT PRIMARY KEY,
                    local_delivery_state TEXT NOT NULL,
                    server_read_state TEXT NOT NULL,
                    server_dismiss_state TEXT NOT NULL,
                    presented_at TEXT,
                    delivery_error TEXT
                );

                CREATE TABLE IF NOT EXISTS event_observer_status (
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    details TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source_authority,
                        server_profile_id,
                        authenticated_principal_id,
                        stream_name,
                        stream_instance_id
                    )
                );

                CREATE TABLE IF NOT EXISTS event_retention_policies (
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    max_age_days INTEGER NOT NULL,
                    max_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source_authority,
                        server_profile_id,
                        authenticated_principal_id,
                        stream_name,
                        stream_instance_id
                    )
                );
                """
            )

    def record_event_and_advance_processed_cursor(
        self,
        event: NormalizedEventRecord,
    ) -> EventStateRecordResult:
        """Atomically insert event/dedupe state and advance the processed cursor."""

        dedupe_key = self._dedupe_key(event)
        event_key = self._event_key(event, dedupe_key=dedupe_key)
        now = _utc_now()

        with self._get_connection() as conn:
            if self._dedupe_exists(conn, dedupe_key):
                return EventStateRecordResult(
                    event_key=event_key,
                    is_duplicate=True,
                    cursor=self._get_cursor_with_connection(conn, event, table="event_processed_cursors"),
                )

            conn.execute(
                """
                INSERT INTO event_records (
                    event_key,
                    dedupe_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    stream_name,
                    stream_instance_id,
                    event_kind,
                    entity_ref,
                    payload_hash,
                    event_id,
                    server_cursor,
                    emitted_at,
                    received_at,
                    transport_type,
                    payload_kind,
                    payload,
                    stored_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_key,
                    dedupe_key,
                    event.source_authority,
                    event.server_profile_id,
                    event.authenticated_principal_id,
                    event.stream_name,
                    event.stream_instance_id,
                    event.event_kind,
                    _json_dumps(event.entity_ref),
                    event.payload_hash,
                    event.event_id,
                    event.server_cursor,
                    event.emitted_at,
                    event.received_at,
                    event.transport_type,
                    event.payload_kind,
                    _json_dumps(event.payload),
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO event_dedupe_records (
                    dedupe_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    stream_name,
                    stream_instance_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dedupe_key,
                    event.source_authority,
                    event.server_profile_id,
                    event.authenticated_principal_id,
                    event.stream_name,
                    event.stream_instance_id,
                    now,
                ),
            )
            if event.server_cursor is not None:
                self._upsert_cursor(conn, event, table="event_processed_cursors", cursor=event.server_cursor, now=now)
            conn.commit()

            return EventStateRecordResult(
                event_key=event_key,
                is_duplicate=False,
                cursor=self._get_cursor_with_connection(conn, event, table="event_processed_cursors"),
            )

    def is_duplicate_event(self, event: NormalizedEventRecord) -> bool:
        with self._get_connection() as conn:
            return self._dedupe_exists(conn, self._dedupe_key(event))

    def remember_event(self, event: NormalizedEventRecord) -> DedupeResult:
        """Compatibility method for observer code paths that only track dedupe."""

        dedupe_key = self._dedupe_key(event)
        if self.is_duplicate_event(event):
            return DedupeResult(key=dedupe_key, is_duplicate=True)

        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO event_dedupe_records (
                    dedupe_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    stream_name,
                    stream_instance_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dedupe_key,
                    event.source_authority,
                    event.server_profile_id,
                    event.authenticated_principal_id,
                    event.stream_name,
                    event.stream_instance_id,
                    now,
                ),
            )
            conn.commit()
        return DedupeResult(key=dedupe_key, is_duplicate=False)

    def acknowledge_event(
        self,
        event: NormalizedEventRecord,
        *,
        expected_cursor: str | None | _NoExpectedCursor = _NO_EXPECTED_CURSOR,
    ) -> CursorAdvanceResult:
        current = self.get_processed_cursor(
            source_authority=event.source_authority,
            server_profile_id=event.server_profile_id,
            authenticated_principal_id=event.authenticated_principal_id,
            stream_name=event.stream_name,
            stream_instance_id=event.stream_instance_id,
        )
        if not isinstance(expected_cursor, _NoExpectedCursor) and current.cursor != expected_cursor:
            return self.reset_cursor(current, reason="cursor_mismatch")

        result = self.record_event_and_advance_processed_cursor(event)
        if event.server_cursor is None:
            return CursorAdvanceResult(
                status=CursorAdvanceStatus.IGNORED_NO_CURSOR,
                cursor=result.cursor,
                reason="missing_server_cursor",
            )
        return CursorAdvanceResult(status=CursorAdvanceStatus.ADVANCED, cursor=result.cursor)

    def reset_cursor(self, cursor: EventCursor, *, reason: str = "stale_cursor") -> CursorAdvanceResult:
        reset = EventCursor(
            source_authority=cursor.source_authority,
            server_profile_id=cursor.server_profile_id,
            authenticated_principal_id=cursor.authenticated_principal_id,
            stream_name=cursor.stream_name,
            stream_instance_id=cursor.stream_instance_id,
            cursor=None,
        )
        with self._get_connection() as conn:
            self._upsert_cursor(
                conn,
                reset,
                table="event_processed_cursors",
                cursor=None,
                now=_utc_now(),
            )
            self._upsert_observer_status(
                conn,
                reset,
                status="cursor_reset",
                reason=reason,
                details={},
                now=_utc_now(),
            )
            conn.commit()
        return CursorAdvanceResult(
            status=CursorAdvanceStatus.STALE_RESET,
            cursor=reset,
            reason=reason,
        )

    def reset_stream_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
        reason: str = "stale_cursor",
    ) -> CursorAdvanceResult:
        return self.reset_cursor(
            EventCursor(
                source_authority=source_authority,
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                stream_name=stream_name,
                stream_instance_id=stream_instance_id,
            ),
            reason=reason,
        )

    def get_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventCursor:
        return self.get_processed_cursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )

    def get_processed_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventCursor:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )
        with self._get_connection() as conn:
            return self._get_cursor_with_connection(conn, cursor, table="event_processed_cursors")

    def get_presented_high_water(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventCursor:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )
        with self._get_connection() as conn:
            return self._get_cursor_with_connection(conn, cursor, table="event_presented_high_water")

    def mark_event_presented_and_advance_high_water(
        self,
        *,
        event_key: str,
        cursor: str | None,
        presented_at: str | None = None,
    ) -> NotificationPresentationRecord:
        now = _utc_now()
        presented_at = presented_at or now
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT source_authority, server_profile_id, authenticated_principal_id, stream_name, stream_instance_id
                FROM event_records
                WHERE event_key = ?
                """,
                (event_key,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Event not found: {event_key}")

            conn.execute(
                """
                INSERT INTO event_presentations (
                    event_key,
                    local_delivery_state,
                    server_read_state,
                    server_dismiss_state,
                    presented_at,
                    delivery_error
                )
                VALUES (?, ?, ?, ?, ?, NULL)
                ON CONFLICT(event_key) DO UPDATE SET
                    local_delivery_state = excluded.local_delivery_state,
                    presented_at = excluded.presented_at,
                    delivery_error = NULL
                """,
                (event_key, "delivered", "unknown", "unknown", presented_at),
            )
            scope_cursor = EventCursor(
                source_authority=row["source_authority"],
                server_profile_id=row["server_profile_id"],
                authenticated_principal_id=row["authenticated_principal_id"],
                stream_name=row["stream_name"],
                stream_instance_id=row["stream_instance_id"],
                cursor=cursor,
            )
            self._upsert_cursor(
                conn,
                scope_cursor,
                table="event_presented_high_water",
                cursor=cursor,
                now=now,
            )
            conn.commit()

        return NotificationPresentationRecord(
            event_key=event_key,
            local_delivery_state="delivered",
            server_read_state="unknown",
            server_dismiss_state="unknown",
            presented_at=presented_at,
        )

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM event_records
                ORDER BY id ASC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def record_observer_status(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
        status: str,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )
        now = _utc_now()
        with self._get_connection() as conn:
            self._upsert_observer_status(
                conn,
                cursor,
                status=status,
                reason=reason,
                details=details or {},
                now=now,
            )
            conn.commit()
        return self.get_observer_status(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )

    def get_observer_status(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM event_observer_status
                WHERE source_authority = ?
                  AND server_profile_id = ?
                  AND authenticated_principal_id = ?
                  AND stream_name = ?
                  AND stream_instance_id = ?
                """,
                (
                    source_authority,
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    stream_name,
                    stream_instance_id,
                ),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["server_profile_id"] = _restore_scope_value(data["server_profile_id"])
        data["authenticated_principal_id"] = _restore_scope_value(data["authenticated_principal_id"])
        data["details"] = json.loads(data["details"])
        return data

    def prune_stream_state(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
        max_count: int | None = None,
        older_than: str | None = None,
    ) -> int:
        """Prune oldest normalized events for a stream while preserving cursors."""

        if max_count is None and older_than is None:
            raise ValueError("max_count or older_than is required")

        with self._get_connection() as conn:
            rows_by_id: dict[int, sqlite3.Row] = {}
            scope_params = (
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id,
            )
            if max_count is not None:
                count_rows = conn.execute(
                    """
                    SELECT id, event_key, dedupe_key
                    FROM event_records
                    WHERE source_authority = ?
                      AND server_profile_id IS ?
                      AND authenticated_principal_id IS ?
                      AND stream_name = ?
                      AND stream_instance_id = ?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (*scope_params, max(int(max_count), 0)),
                ).fetchall()
                rows_by_id.update({int(row["id"]): row for row in count_rows})
            if older_than is not None:
                age_rows = conn.execute(
                    """
                    SELECT id, event_key, dedupe_key
                    FROM event_records
                    WHERE source_authority = ?
                      AND server_profile_id IS ?
                      AND authenticated_principal_id IS ?
                      AND stream_name = ?
                      AND stream_instance_id = ?
                      AND stored_at < ?
                    ORDER BY id ASC
                    """,
                    (*scope_params, older_than),
                ).fetchall()
                rows_by_id.update({int(row["id"]): row for row in age_rows})
            rows = list(rows_by_id.values())
            if not rows:
                return 0

            event_ids = [int(row["id"]) for row in rows]
            event_keys = [str(row["event_key"]) for row in rows]
            dedupe_keys = [str(row["dedupe_key"]) for row in rows]

            conn.executemany("DELETE FROM event_presentations WHERE event_key = ?", [(key,) for key in event_keys])
            conn.executemany("DELETE FROM event_records WHERE id = ?", [(event_id,) for event_id in event_ids])
            conn.executemany("DELETE FROM event_dedupe_records WHERE dedupe_key = ?", [(key,) for key in dedupe_keys])
            conn.commit()
        return len(rows)

    def get_retention_policy(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventRetentionPolicy:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT max_age_days, max_count
                FROM event_retention_policies
                WHERE source_authority = ?
                  AND server_profile_id = ?
                  AND authenticated_principal_id = ?
                  AND stream_name = ?
                  AND stream_instance_id = ?
                """,
                (
                    source_authority,
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    stream_name,
                    stream_instance_id,
                ),
            ).fetchone()
        if row is None:
            return EventRetentionPolicy(
                source_authority=source_authority,
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                stream_name=stream_name,
                stream_instance_id=stream_instance_id,
            )
        return EventRetentionPolicy(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
            max_age_days=int(row["max_age_days"]),
            max_count=int(row["max_count"]),
        )

    def set_retention_policy(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
        max_age_days: int = 30,
        max_count: int = 10_000,
    ) -> EventRetentionPolicy:
        if max_age_days <= 0:
            raise ValueError("max_age_days must be positive")
        if max_count <= 0:
            raise ValueError("max_count must be positive")
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO event_retention_policies (
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    stream_name,
                    stream_instance_id,
                    max_age_days,
                    max_count,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    stream_name,
                    stream_instance_id
                )
                DO UPDATE SET
                    max_age_days = excluded.max_age_days,
                    max_count = excluded.max_count,
                    updated_at = excluded.updated_at
                """,
                (
                    source_authority,
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    stream_name,
                    stream_instance_id,
                    int(max_age_days),
                    int(max_count),
                    now,
                ),
            )
            conn.commit()
        return self.get_retention_policy(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )

    def clear_server_profile_state(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
    ) -> dict[str, int]:
        """Clear durable event state for logout, credential removal, or profile deletion."""

        if not server_profile_id:
            raise ValueError("server_profile_id is required")

        with self._get_connection() as conn:
            event_filter, params = self._server_profile_filter(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            rows = conn.execute(
                f"""
                SELECT event_key, dedupe_key
                FROM event_records
                WHERE {event_filter}
                """,
                params,
            ).fetchall()
            event_keys = [str(row["event_key"]) for row in rows]
            dedupe_keys = [str(row["dedupe_key"]) for row in rows]

            presentation_count = self._count_matching_presentations(conn, event_keys)
            event_count = len(event_keys)
            dedupe_count = len(dedupe_keys)
            processed_cursor_count = self._count_scoped_rows(
                conn,
                "event_processed_cursors",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            presented_high_water_count = self._count_scoped_rows(
                conn,
                "event_presented_high_water",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            observer_status_count = self._count_scoped_rows(
                conn,
                "event_observer_status",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            retention_policy_count = self._count_scoped_rows(
                conn,
                "event_retention_policies",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )

            conn.executemany("DELETE FROM event_presentations WHERE event_key = ?", [(key,) for key in event_keys])
            conn.executemany("DELETE FROM event_records WHERE event_key = ?", [(key,) for key in event_keys])
            conn.executemany("DELETE FROM event_dedupe_records WHERE dedupe_key = ?", [(key,) for key in dedupe_keys])
            self._delete_scoped_rows(
                conn,
                "event_processed_cursors",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            self._delete_scoped_rows(
                conn,
                "event_presented_high_water",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            self._delete_scoped_rows(
                conn,
                "event_observer_status",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            self._delete_scoped_rows(
                conn,
                "event_retention_policies",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )
            conn.commit()

        return {
            "events": event_count,
            "dedupe_records": dedupe_count,
            "presentations": presentation_count,
            "processed_cursors": processed_cursor_count,
            "presented_high_water": presented_high_water_count,
            "observer_status": observer_status_count,
            "retention_policies": retention_policy_count,
        }

    @staticmethod
    def _dedupe_key(event: NormalizedEventRecord) -> str:
        scope = [
            event.source_authority,
            event.server_profile_id,
            event.authenticated_principal_id,
            event.stream_name,
            event.stream_instance_id,
        ]
        if event.event_id:
            return _json_dumps([*scope, "event_id", event.event_id])
        if event.server_cursor:
            return _json_dumps([*scope, "server_cursor", event.server_cursor])
        fallback = EventDedupeKey.from_event(event)
        return _json_dumps(
            [
                *scope,
                "fallback",
                fallback.event_kind,
                fallback.entity_id,
                fallback.timestamp,
                fallback.payload_hash,
            ]
        )

    @staticmethod
    def _event_key(event: NormalizedEventRecord, *, dedupe_key: str) -> str:
        scope = [
            event.source_authority,
            event.server_profile_id or "none",
            event.authenticated_principal_id or "none",
            event.stream_name,
            event.stream_instance_id,
        ]
        event_identity = event.event_id or event.server_cursor or dedupe_key
        return ":".join([*scope, event_identity])

    @staticmethod
    def _dedupe_exists(conn: sqlite3.Connection, dedupe_key: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM event_dedupe_records WHERE dedupe_key = ?",
            (dedupe_key,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _get_cursor_with_connection(
        conn: sqlite3.Connection,
        cursor_like: EventCursor | NormalizedEventRecord,
        *,
        table: str,
    ) -> EventCursor:
        row = conn.execute(
            f"""
            SELECT cursor
            FROM {table}
            WHERE source_authority = ?
              AND server_profile_id = ?
              AND authenticated_principal_id = ?
              AND stream_name = ?
              AND stream_instance_id = ?
            """,
            (
                cursor_like.source_authority,
                _scope_value(cursor_like.server_profile_id),
                _scope_value(cursor_like.authenticated_principal_id),
                cursor_like.stream_name,
                cursor_like.stream_instance_id,
            ),
        ).fetchone()
        return EventCursor(
            source_authority=cursor_like.source_authority,
            server_profile_id=cursor_like.server_profile_id,
            authenticated_principal_id=cursor_like.authenticated_principal_id,
            stream_name=cursor_like.stream_name,
            stream_instance_id=cursor_like.stream_instance_id,
            cursor=row["cursor"] if row is not None else None,
        )

    @staticmethod
    def _upsert_cursor(
        conn: sqlite3.Connection,
        cursor_like: EventCursor | NormalizedEventRecord,
        *,
        table: str,
        cursor: str | None,
        now: str,
    ) -> None:
        conn.execute(
            f"""
            INSERT INTO {table} (
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id,
                cursor,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id
            )
            DO UPDATE SET cursor = excluded.cursor, updated_at = excluded.updated_at
            """,
            (
                cursor_like.source_authority,
                _scope_value(cursor_like.server_profile_id),
                _scope_value(cursor_like.authenticated_principal_id),
                cursor_like.stream_name,
                cursor_like.stream_instance_id,
                cursor,
                now,
            ),
        )

    @staticmethod
    def _upsert_observer_status(
        conn: sqlite3.Connection,
        cursor_like: EventCursor,
        *,
        status: str,
        reason: str | None,
        details: Mapping[str, Any],
        now: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO event_observer_status (
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id,
                status,
                reason,
                details,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id
            )
            DO UPDATE SET
                status = excluded.status,
                reason = excluded.reason,
                details = excluded.details,
                updated_at = excluded.updated_at
            """,
            (
                cursor_like.source_authority,
                _scope_value(cursor_like.server_profile_id),
                _scope_value(cursor_like.authenticated_principal_id),
                cursor_like.stream_name,
                cursor_like.stream_instance_id,
                status,
                reason,
                _json_dumps(details),
                now,
            ),
        )

    @staticmethod
    def _event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["entity_ref"] = json.loads(data["entity_ref"])
        data["payload"] = json.loads(data["payload"])
        return data

    @staticmethod
    def _server_profile_filter(
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        if authenticated_principal_id is None:
            return "source_authority = ? AND server_profile_id = ?", ("server", server_profile_id)
        return (
            "source_authority = ? AND server_profile_id = ? AND authenticated_principal_id = ?",
            ("server", server_profile_id, authenticated_principal_id),
        )

    @staticmethod
    def _scoped_table_filter(
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
    ) -> tuple[str, tuple[str, ...]]:
        if authenticated_principal_id is None:
            return "source_authority = ? AND server_profile_id = ?", ("server", _scope_value(server_profile_id))
        return (
            "source_authority = ? AND server_profile_id = ? AND authenticated_principal_id = ?",
            ("server", _scope_value(server_profile_id), _scope_value(authenticated_principal_id)),
        )

    @classmethod
    def _count_scoped_rows(
        cls,
        conn: sqlite3.Connection,
        table: str,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
    ) -> int:
        table_filter, params = cls._scoped_table_filter(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
        )
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table} WHERE {table_filter}", params).fetchone()
        return int(row["count"])

    @classmethod
    def _delete_scoped_rows(
        cls,
        conn: sqlite3.Connection,
        table: str,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
    ) -> None:
        table_filter, params = cls._scoped_table_filter(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
        )
        conn.execute(f"DELETE FROM {table} WHERE {table_filter}", params)

    @staticmethod
    def _count_matching_presentations(conn: sqlite3.Connection, event_keys: list[str]) -> int:
        if not event_keys:
            return 0
        count = 0
        for event_key in event_keys:
            row = conn.execute(
                "SELECT 1 FROM event_presentations WHERE event_key = ?",
                (event_key,),
            ).fetchone()
            if row is not None:
                count += 1
        return count


def _json_dumps(value: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _scope_value(value: str | None) -> str:
    return value if value is not None else "none"


def _restore_scope_value(value: str) -> str | None:
    return None if value == "none" else value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
