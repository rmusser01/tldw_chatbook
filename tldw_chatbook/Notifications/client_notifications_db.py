"""SQLite-backed local notification inbox for Chatbook-owned events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..DB.base_db import BaseDB
from ..DB.private_sqlite import connect_private_sqlite


DEFAULT_NOTIFICATION_SETTINGS = {
    "enabled": True,
    "toast_enabled": True,
    "persist_enabled": True,
    "category_preferences": {},
}

_CATEGORY_NOTIFICATION_SETTINGS = (
    "enabled",
    "toast_enabled",
    "persist_enabled",
)


class ClientNotificationsDB(BaseDB):
    """Dedicated local queue/inbox store for client notifications.

    task-15466: file-backed connections are held per thread (the
    ``Workspace_DB`` idiom). The previous shape opened a brand-new
    private-SQLite connection -- which re-verifies the database file and
    its three sidecars every time -- for every operation, including one
    per dispatched notification, and never closed any of them
    (``with conn`` is sqlite3's TRANSACTION context manager, not a closing
    one, so they leaked until GC).

    Thread safety: the inbox is read from the UI thread and written from
    dispatch worker threads, and sqlite3 refuses a connection used off its
    creating thread (``check_same_thread`` defaults to True). Thread-local
    storage is what makes the held connection safe -- each thread owns one.
    The ``:memory:`` branch is deliberately NOT thread-local: an in-memory
    database lives inside its connection, so per-thread connections would
    each see their own empty inbox. It keeps a single shared connection
    (which is why ``Home/active_work_adapter.py`` only moves inbox reads
    off-loop once it has confirmed the store is file-backed).
    """

    _CURRENT_SCHEMA_VERSION = 1

    #: Liveness-ping gate (mirrors `Workspace_DB`/`ChaChaNotes_DB`,
    #: task-261/3011): a recently-used held connection is known-good
    #: without spending a `SELECT 1` on every call.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: str | Path, client_id: str = "default"):
        # Both must precede super().__init__: _initialize_schema (run
        # eagerly for :memory: below) already needs a connection.
        self._memory_conn: sqlite3.Connection | None = None
        self._thread_local = threading.local()
        # TASK-21105: file-backed schema creation is deferred to the first
        # connection (initialize_schema=False below). Construction resolves
        # the path only; no file/WAL sidecars are created until first
        # feature use (an inbox read or a dispatched notification).
        # ``:memory:`` stays eager so its single shared connection is
        # created on the constructing thread, exactly as before.
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        super().__init__(db_path, client_id, initialize_schema=False)
        if self.is_memory_db:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the schema exactly once, on first connection (TASK-21105).

        Single-flight under a lock: the first touch can come from a
        dispatch worker thread while the UI thread reads the inbox. A
        failed attempt leaves ``_schema_ready`` False so the next
        operation retries.
        """
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self._initialize_schema()
            self._schema_ready = True

    def _get_connection(self) -> sqlite3.Connection:
        self._ensure_schema()
        return self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        """Open a raw connection without the first-use schema ensure.

        ``_initialize_schema`` must use this directly: it runs inside
        ``_ensure_schema``'s lock, and going through ``_get_connection``
        there would deadlock on the non-reentrant lock.
        """
        if getattr(self, "is_memory_db", False):
            if self._memory_conn is None:
                self._memory_conn = connect_private_sqlite(
                    "notifications.client",
                    ":memory:",
                )
                self._memory_conn.row_factory = sqlite3.Row
                # synchronous is harmless (and a no-op performance-wise) on an
                # in-memory database; set for uniformity with the file-backed
                # branch below rather than special-casing it away (task-15465).
                self._memory_conn.execute("PRAGMA synchronous = NORMAL")
                self._memory_conn.isolation_level = None
            return self._memory_conn
        conn = super()._get_connection()
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local notification inbox)
        # and avoids an fsync per commit. Unlike journal_mode, which is
        # persisted in the file, synchronous is per-connection and must be
        # re-applied on every NEW connection -- which is why this pairing
        # lives in the one place connections are created (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        # task-3012: a held (long-lived) connection needs true autocommit.
        # Python's default isolation mode auto-BEGINs on any DML; that
        # implicit transaction then makes the explicit BEGIN in
        # `transaction()` raise "cannot start a transaction within a
        # transaction", and silently ROLLS BACK bare DML on close.
        # Audited (task-15466) -- every site in this file: `_initialize_
        # schema` executescript (self-commits either way), single-statement
        # writes in insert_notification and _update_flags (each its own
        # autocommit transaction), the multi-statement settings loop in
        # update_settings (now wrapped in an explicit `transaction()`), and
        # read-only SELECTs elsewhere.
        conn.isolation_level = None
        return conn

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        In-memory stores share the single cached connection instead (see
        the class docstring). The liveness probe is a plain no-op
        statement; a connection another component closed (or that SQLite
        invalidated) is transparently replaced.
        """
        if getattr(self, "is_memory_db", False):
            return self._get_connection()
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            last_used = getattr(self._thread_local, "conn_last_used", None)
            if (
                last_used is None
                or (time.monotonic() - last_used)
                >= self._LIVENESS_PING_IDLE_SECONDS
            ):
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001 - already unusable
                        pass
                    conn = None
        if conn is None:
            conn = self._get_connection()
            self._thread_local.conn = conn
        self._thread_local.conn_last_used = time.monotonic()
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's held connection (no transaction opened).

        In autocommit mode a single statement is its own transaction, so
        reads and single-statement writes need nothing more than this.
        """
        yield self._held_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a write transaction.

        Required for any block whose statements must land (or not land)
        together: in autocommit mode each bare statement would otherwise
        commit on its own.

        Nesting: the explicit BEGIN runs on the ONE connection this thread
        holds (or, for ``:memory:``, the single shared one), so nesting a
        second ``transaction()`` inside one raises
        ``sqlite3.OperationalError: cannot start a transaction within a
        transaction``. Pre-port each block had its own connection and
        nesting silently "worked"; the outer block still rolls back
        cleanly, because the failure propagates through its ``except``.

        Raises:
            Exception: Re-raised after rolling back, on any error inside
                the ``with`` block. On clean exit the transaction commits.
        """
        conn = self._held_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        """Close the memory connection, or this thread's held connection."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _initialize_schema(self) -> None:
        # Raw connection: runs under _ensure_schema's lock (TASK-21105), so
        # it cannot use connection()/_held_connection (both re-enter
        # _get_connection). File-backed: one short-lived connection, closed
        # below; the held per-thread connection opens on the first real
        # operation. :memory:: the shared cached connection, never closed.
        conn = self._open_connection()
        try:
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

                CREATE TABLE IF NOT EXISTS client_notification_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
        finally:
            if not getattr(self, "is_memory_db", False):
                conn.close()

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
        with self.connection() as conn:
            # Single statement: autocommit already makes it its own
            # transaction, so no explicit commit is needed.
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
        return self.get_notification(notification_id)

    def insert(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for older inbox/controller call sites."""
        return self.insert_notification(**kwargs)

    def get_notification(self, notification_id: int) -> dict[str, Any]:
        """Return a notification by id."""
        with self.connection() as conn:
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
        severity: str | None = None,
        source_backend: str | None = None,
        source_entity_kind: str | None = None,
        source_entity_id: str | None = None,
        is_read: bool | None = None,
    ) -> list[dict[str, Any]]:
        """List inbox notifications newest-first."""
        clauses = []
        params: list[Any] = []
        if not include_dismissed:
            clauses.append("is_dismissed = 0")
        if category:
            clauses.append("category = ?")
            params.append(category)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if source_backend:
            clauses.append("source_backend = ?")
            params.append(source_backend)
        if source_entity_kind:
            clauses.append("source_entity_kind = ?")
            params.append(source_entity_kind)
        if source_entity_id:
            clauses.append("source_entity_id = ?")
            params.append(source_entity_id)
        if is_read is not None:
            clauses.append("is_read = ?")
            params.append(int(bool(is_read)))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(int(limit), 1))
        with self.connection() as conn:
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

    def list_notifications_after_id(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
        include_dismissed: bool = False,
    ) -> list[dict[str, Any]]:
        """List notifications newer than a known id for poll/observe flows."""
        clauses = ["id > ?"]
        params: list[Any] = [int(after_id)]
        if not include_dismissed:
            clauses.append("is_dismissed = 0")
        params.append(max(int(limit), 1))
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM client_notifications
                WHERE {" AND ".join(clauses)}
                ORDER BY id ASC
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

    def get_settings(self) -> dict[str, Any]:
        """Return local notification settings with defaults filled in."""
        settings = deepcopy(DEFAULT_NOTIFICATION_SETTINGS)
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM client_notification_settings"
            ).fetchall()
        for row in rows:
            try:
                value = json.loads(row["value"])
            except json.JSONDecodeError:
                continue
            if row["key"] == "category_preferences":
                try:
                    value = self._normalize_category_preferences(value)
                except ValueError:
                    continue
            settings[row["key"]] = value
        return settings

    def get_preferences(self) -> dict[str, Any]:
        """Compatibility preferences view used by the inbox controller."""
        settings = self.get_settings()
        category_preferences = settings.get("category_preferences") or {}
        muted_categories = [
            category
            for category, preferences in category_preferences.items()
            if isinstance(preferences, Mapping) and preferences.get("enabled") is False
        ]
        return {
            "delivery_enabled": bool(settings.get("enabled", True)),
            "muted_categories": sorted(muted_categories),
            "muted_severities": [],
        }

    def update_settings(self, **settings: Any) -> dict[str, Any]:
        """Persist known local notification settings and return the effective set."""
        unknown = set(settings) - set(DEFAULT_NOTIFICATION_SETTINGS)
        if unknown:
            raise ValueError(f"Unknown notification settings: {sorted(unknown)}")
        if not settings:
            # `update_preferences` reaches here with nothing to write when
            # its caller passed no changes. Returning early keeps a
            # zero-statement BEGIN IMMEDIATE -- which would still take the
            # write lock -- off a read-only path.
            return self.get_settings()
        now = self._now_iso()
        # One upsert per setting: under autocommit they would commit
        # independently, so an explicit transaction keeps a multi-key
        # update all-or-nothing.
        with self.transaction() as conn:
            for key, value in settings.items():
                if key == "category_preferences":
                    value = self._normalize_category_preferences(value)
                conn.execute(
                    """
                    INSERT INTO client_notification_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value, sort_keys=True), now),
                )
        return self.get_settings()

    def update_preferences(
        self,
        *,
        delivery_enabled: bool | None = None,
        muted_categories: list[str] | tuple[str, ...] | set[str] | None = None,
        muted_severities: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any]:
        """Compatibility preferences writer used by the inbox controller."""
        settings: dict[str, Any] = {}
        if delivery_enabled is not None:
            settings["enabled"] = bool(delivery_enabled)
        if muted_categories is not None:
            settings["category_preferences"] = {
                str(category): {"enabled": False}
                for category in muted_categories
                if str(category).strip()
            }
        self.update_settings(**settings)
        preferences = self.get_preferences()
        if muted_severities is not None:
            preferences["muted_severities"] = sorted(
                str(item) for item in muted_severities
            )
        return preferences

    @staticmethod
    def _normalize_category_preferences(value: Any) -> dict[str, dict[str, bool]]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("category_preferences must be a mapping.")

        normalized: dict[str, dict[str, bool]] = {}
        valid_settings = set(_CATEGORY_NOTIFICATION_SETTINGS)
        for raw_category, raw_preferences in value.items():
            category = str(raw_category).strip()
            if not category:
                raise ValueError("category_preferences contains an empty category.")
            if raw_preferences is None:
                continue
            if not isinstance(raw_preferences, Mapping):
                raise ValueError(
                    f"category_preferences[{category!r}] must be a mapping."
                )
            unknown = set(raw_preferences) - valid_settings
            if unknown:
                raise ValueError(
                    f"Unknown category notification settings for {category!r}: {sorted(unknown)}"
                )
            preferences = {
                setting: bool(raw_preferences[setting])
                for setting in _CATEGORY_NOTIFICATION_SETTINGS
                if setting in raw_preferences
            }
            if preferences:
                normalized[category] = preferences
        return dict(sorted(normalized.items()))

    def _update_flags(self, notification_id: int, **fields: Any) -> bool:
        assignments = ", ".join(f"{field} = ?" for field in fields)
        params = list(fields.values())
        params.append(notification_id)
        with self.connection() as conn:
            # Single statement: autocommit already makes it its own
            # transaction, so no explicit commit is needed.
            cursor = conn.execute(
                f"UPDATE client_notifications SET {assignments} WHERE id = ?",
                tuple(params),
            )
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
