"""SQLite-backed durable state for normalized event observation."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from loguru import logger

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
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


#: Key under which the single shared ``:memory:`` connection is held.
#: File-backed connections are keyed by thread ident (an ``int``), so a
#: string key can never collide with one.
_MEMORY_KEY = "memory"


@dataclass(slots=True)
class _HeldConnection:
    """One long-lived connection plus the bookkeeping ``close()`` needs."""

    conn: sqlite3.Connection
    last_used: float = field(default_factory=time.monotonic)
    #: Number of operations currently inside ``connection()``/``transaction()``
    #: on this connection. ``close()`` refuses to close a busy connection.
    depth: int = 0


class _NoExpectedCursor:
    pass


_NO_EXPECTED_CURSOR = _NoExpectedCursor()


class _FilterUnset:
    pass


_FILTER_UNSET = _FilterUnset()


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


@dataclass(frozen=True, slots=True)
class EventReplayWindow:
    source_authority: SourceAuthority
    server_profile_id: str | None
    authenticated_principal_id: str | None
    stream_name: str
    stream_instance_id: str
    state: str
    earliest_retained_cursor: str | None = None
    latest_retained_cursor: str | None = None
    last_pruned_cursor: str | None = None
    pruned_event_count: int = 0
    updated_at: str | None = None


class EventStateRepository(BaseDB):
    """Durable event rows, dedupe records, cursors, and presentation watermarks.

    TASK-21131: file-backed connections are HELD per thread (the sibling
    ``ClientNotificationsDB`` idiom). The previous shape opened a brand-new
    private-SQLite connection -- which re-validates the owner policy, the
    trusted directory and the artifact every time -- for every operation
    and never closed any of them (``with conn:`` is sqlite3's TRANSACTION
    context manager, not a closing one, so they leaked until GC). Measured
    on the shipped feed path, that open was 0.54 ms against 0.05 ms for the
    statement it was opened to run.

    Thread safety: the durable event ledger is written from the event loop
    (``EventObserver``) and read from ``asyncio.to_thread`` workers (Home's
    active-work cache), so each thread gets its OWN connection. They are
    keyed by thread ident in a plain dict rather than a ``threading.local``
    because ``close()`` has to reach connections it does not own -- which
    also requires ``check_same_thread=False`` on the file branch: with
    sqlite3's default guard, a cross-thread ``close()`` raises instead of
    closing, so a ``threading.local`` store can never release a worker
    pool's connections at all. Each connection still has exactly one user
    thread; ``close()`` is the only cross-thread toucher, and it refuses
    any connection whose thread is mid-operation.

    The ``:memory:`` branch is deliberately NOT per thread: an in-memory
    database lives inside its connection, so per-thread connections would
    each see their own empty ledger. It keeps a single shared connection --
    which is why closing it destroys the database, and why it keeps
    sqlite3's default same-thread guard rather than silently allowing two
    threads onto one unserialised handle.
    """

    _CURRENT_SCHEMA_VERSION = 1

    #: Liveness-ping gate (mirrors ``ClientNotificationsDB``): a recently
    #: used held connection is known-good without spending a ``SELECT 1``
    #: on every call.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: str | Path, client_id: str = "default") -> None:
        self._memory_conn: sqlite3.Connection | None = None
        self._held: dict[object, _HeldConnection] = {}
        self._held_lock = threading.RLock()
        # TASK-21105: file-backed schema creation (10 DDL statements) is
        # deferred to the first connection (initialize_schema=False below);
        # a local-only user whose event observation never runs pays nothing
        # at boot. ``:memory:`` (the app's parity-build fallback) stays
        # eager so its single cached connection binds to the constructing
        # thread, exactly as before.
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        super().__init__(db_path, client_id, initialize_schema=False)
        if self.is_memory_db:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the schema exactly once, on first connection (TASK-21105).

        Single-flight under a lock: feed reads can run from thread
        workers. A failed attempt leaves ``_schema_ready`` False so the
        next operation retries.
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
                    "notifications.event_state",
                    ":memory:",
                )
                self._memory_conn.row_factory = sqlite3.Row
                # synchronous is harmless (and a no-op performance-wise) on an
                # in-memory database; set for uniformity with the file-backed
                # branch below (task-15465).
                self._memory_conn.execute("PRAGMA synchronous = NORMAL")
                self._memory_conn.isolation_level = None
                # Consistent with the schema-creation script, which also
                # asserts it. Currently inert -- this schema declares no
                # FOREIGN KEY constraints -- but the pragma is per
                # connection, so asserting it in the ONE place connections
                # are created is what keeps it from drifting if one is
                # added (TASK-21131 AC #1).
                self._memory_conn.execute("PRAGMA foreign_keys = ON")
            return self._memory_conn
        # TASK-21131: this used to go through ``BaseDB._get_connection``,
        # which opens under the ``db.base`` owner. It now names this
        # module's OWN owner (whose registry entry was corrected to describe
        # the private file the app actually gives this store -- the enforced
        # target kinds are identical to ``db.base``'s), and passes
        # ``check_same_thread=False``, which BaseDB cannot. Held connections
        # are handed to ``close()`` on another thread; sqlite3's default
        # guard would refuse that and leave every worker-pool connection
        # open for the life of the process.
        conn = connect_private_sqlite(
            "notifications.event_state",
            self.db_path_str,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local event/notification
        # ledger) and avoids an fsync per commit (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        # TASK-21131: a HELD (long-lived) connection needs true autocommit.
        # Python's legacy isolation mode auto-BEGINs a DEFERRED transaction
        # on the first DML statement, which then makes the explicit
        # ``BEGIN IMMEDIATE`` in `transaction()` raise "cannot start a
        # transaction within a transaction", and silently rolls back bare
        # DML on close.
        conn.isolation_level = None
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held sqlite3 connection (see `_held_entry`)."""
        return self._held_entry().conn

    def _held_entry(self) -> _HeldConnection:
        """Return this thread's held connection entry, opening or reviving it.

        In-memory stores share the single cached connection instead (see
        the class docstring). The liveness probe is a plain no-op
        statement; a connection another component closed (or that SQLite
        invalidated) is transparently replaced.
        """
        if getattr(self, "is_memory_db", False):
            with self._held_lock:
                entry = self._held.get(_MEMORY_KEY)
                if entry is None or entry.conn is not self._memory_conn:
                    entry = _HeldConnection(conn=self._get_connection())
                    self._held[_MEMORY_KEY] = entry
                entry.last_used = time.monotonic()
                return entry

        key = threading.get_ident()
        with self._held_lock:
            entry = self._held.get(key)
        if entry is not None and (
            (time.monotonic() - entry.last_used) >= self._LIVENESS_PING_IDLE_SECONDS
        ):
            try:
                entry.conn.execute("SELECT 1")
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                try:
                    entry.conn.close()
                except Exception:  # noqa: BLE001 - already unusable
                    pass
                with self._held_lock:
                    if self._held.get(key) is entry:
                        del self._held[key]
                entry = None
        if entry is None:
            conn = self._get_connection()
            entry = _HeldConnection(conn=conn)
            with self._held_lock:
                self._held[key] = entry
        entry.last_used = time.monotonic()
        return entry

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's held connection (no transaction opened).

        In autocommit mode a single statement is its own transaction, so
        reads and single-statement writes need nothing more than this.
        """
        # Acquisition and registration share ONE lock hold: a `close()`
        # landing between them would close a connection this operation is
        # about to use, which is the exact failure the depth guard exists
        # to prevent. `_held_lock` is re-entrant, so `_held_entry` may take
        # it again.
        with self._held_lock:
            entry = self._held_entry()
            entry.depth += 1
        try:
            yield entry.conn
        finally:
            with self._held_lock:
                entry.depth -= 1

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside an IMMEDIATE write transaction.

        ``BEGIN IMMEDIATE`` (not the default deferred begin) is a
        prerequisite here, not polish: every write body in this repository
        reads before it writes (dedupe probes, scope lookups, replay-window
        bounds), and under ``isolation_level=None`` a deferred begin takes a
        read snapshot whose later write fails ``BUSY_SNAPSHOT`` -- which
        SQLite's busy handler does NOT retry. Taking the write lock up front
        also makes the read-modify-write bodies atomic against a concurrent
        writer on another thread's connection.

        Nesting: the explicit BEGIN runs on the ONE connection this thread
        holds (or, for ``:memory:``, the single shared one), so nesting a
        second ``transaction()`` inside one raises
        ``sqlite3.OperationalError: cannot start a transaction within a
        transaction``. No body in this file nests.

        Raises:
            Exception: Re-raised after rolling back, on any error inside
                the ``with`` block. On clean exit the transaction commits.
        """
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                # BaseException, not Exception: a cancelled or abandoned
                # body must not leave the write lock held on a connection
                # this thread will keep using.
                try:
                    conn.rollback()
                except Exception as rollback_error:  # noqa: BLE001
                    # Never let a failing rollback replace the original
                    # exception: type name only, no statement or payload.
                    logger.debug(
                        "Event state rollback failed: {}",
                        type(rollback_error).__name__,
                    )
                raise
            else:
                conn.commit()

    def close(self) -> None:
        """Close every held connection that is not mid-operation.

        A connection whose thread is still inside ``connection()``/
        ``transaction()`` is deliberately LEFT OPEN: closing it under live
        work raises ``ProgrammingError: Cannot operate on a closed
        database`` inside that operation (the TASK-21101/21125 shutdown
        class). Committed rows are durable under WAL either way, and an
        open transaction rolls back when the connection is finalized.

        The store re-arms: a later operation transparently opens a fresh
        connection. For ``:memory:`` that means a fresh -- and therefore
        empty -- database, so the schema flag is reset alongside it.
        """
        closable: list[sqlite3.Connection] = []
        with self._held_lock:
            memory_busy = False
            for key in list(self._held):
                entry = self._held[key]
                if entry.depth > 0:
                    if key == _MEMORY_KEY:
                        memory_busy = True
                    continue
                del self._held[key]
                closable.append(entry.conn)
            if self._memory_conn is not None and not memory_busy:
                if all(conn is not self._memory_conn for conn in closable):
                    closable.append(self._memory_conn)
                self._memory_conn = None
                # The in-memory database died with its connection; the next
                # operation must rebuild the schema rather than query tables
                # that no longer exist.
                self._schema_ready = False
        for conn in closable:
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

                CREATE TABLE IF NOT EXISTS event_replay_windows (
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    stream_instance_id TEXT NOT NULL,
                    earliest_retained_cursor TEXT,
                    latest_retained_cursor TEXT,
                    last_pruned_cursor TEXT,
                    pruned_event_count INTEGER NOT NULL DEFAULT 0,
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
        finally:
            if not getattr(self, "is_memory_db", False):
                conn.close()

    def record_event_and_advance_processed_cursor(
        self,
        event: NormalizedEventRecord,
    ) -> EventStateRecordResult:
        """Atomically insert event/dedupe state and advance the processed cursor."""

        dedupe_key = self._dedupe_key(event)
        event_key = self._event_key(event, dedupe_key=dedupe_key)
        now = _utc_now()

        # One IMMEDIATE transaction: the dedupe probe and the inserts it
        # gates must not straddle another writer's commit.
        with self.transaction() as conn:
            if self._dedupe_exists(conn, dedupe_key):
                return EventStateRecordResult(
                    event_key=event_key,
                    is_duplicate=True,
                    cursor=self._get_cursor_with_connection(
                        conn, event, table="event_processed_cursors"
                    ),
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
                self._upsert_cursor(
                    conn,
                    event,
                    table="event_processed_cursors",
                    cursor=event.server_cursor,
                    now=now,
                )
            self._sync_replay_window_bounds(conn, event, now=now)

            return EventStateRecordResult(
                event_key=event_key,
                is_duplicate=False,
                cursor=self._get_cursor_with_connection(
                    conn, event, table="event_processed_cursors"
                ),
            )

    def is_duplicate_event(self, event: NormalizedEventRecord) -> bool:
        with self.connection() as conn:
            return self._dedupe_exists(conn, self._dedupe_key(event))

    def remember_event(self, event: NormalizedEventRecord) -> DedupeResult:
        """Compatibility method for observer code paths that only track dedupe."""

        dedupe_key = self._dedupe_key(event)
        now = _utc_now()
        # The dedupe probe used to run on its own connection, so a second
        # writer could pass it before the first insert committed and then
        # fail the PRIMARY KEY. Probe and insert now share one IMMEDIATE
        # transaction.
        with self.transaction() as conn:
            if self._dedupe_exists(conn, dedupe_key):
                return DedupeResult(key=dedupe_key, is_duplicate=True)
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
        if (
            not isinstance(expected_cursor, _NoExpectedCursor)
            and current.cursor != expected_cursor
        ):
            return self.reset_cursor(current, reason="cursor_mismatch")

        result = self.record_event_and_advance_processed_cursor(event)
        if event.server_cursor is None:
            return CursorAdvanceResult(
                status=CursorAdvanceStatus.IGNORED_NO_CURSOR,
                cursor=result.cursor,
                reason="missing_server_cursor",
            )
        return CursorAdvanceResult(
            status=CursorAdvanceStatus.ADVANCED, cursor=result.cursor
        )

    def reset_cursor(
        self, cursor: EventCursor, *, reason: str = "stale_cursor"
    ) -> CursorAdvanceResult:
        reset = EventCursor(
            source_authority=cursor.source_authority,
            server_profile_id=cursor.server_profile_id,
            authenticated_principal_id=cursor.authenticated_principal_id,
            stream_name=cursor.stream_name,
            stream_instance_id=cursor.stream_instance_id,
            cursor=None,
        )
        with self.transaction() as conn:
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
        with self.connection() as conn:
            return self._get_cursor_with_connection(
                conn, cursor, table="event_processed_cursors"
            )

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
        with self.connection() as conn:
            return self._get_cursor_with_connection(
                conn, cursor, table="event_presented_high_water"
            )

    def mark_event_presented_and_advance_high_water(
        self,
        *,
        event_key: str,
        cursor: str | None,
        presented_at: str | None = None,
    ) -> NotificationPresentationRecord:
        now = _utc_now()
        presented_at = presented_at or now
        # Scope lookup + presentation upsert + high-water advance are one
        # read-modify-write; IMMEDIATE keeps them atomic.
        with self.transaction() as conn:
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

        return NotificationPresentationRecord(
            event_key=event_key,
            local_delivery_state="delivered",
            server_read_state="unknown",
            server_dismiss_state="unknown",
            presented_at=presented_at,
        )

    def list_events(
        self,
        *,
        source_authority: SourceAuthority | None | _FilterUnset = _FILTER_UNSET,
        server_profile_id: str | None | _FilterUnset = _FILTER_UNSET,
        authenticated_principal_id: str | None | _FilterUnset = _FILTER_UNSET,
        stream_name: str | None | _FilterUnset = _FILTER_UNSET,
        stream_instance_id: str | None | _FilterUnset = _FILTER_UNSET,
        payload_kind: str | None | _FilterUnset = _FILTER_UNSET,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []
        for field_name, value in (
            ("source_authority", source_authority),
            ("server_profile_id", server_profile_id),
            ("authenticated_principal_id", authenticated_principal_id),
            ("stream_name", stream_name),
            ("stream_instance_id", stream_instance_id),
            ("payload_kind", payload_kind),
        ):
            if isinstance(value, _FilterUnset):
                continue
            if value is None:
                where_clauses.append(f"{field_name} IS NULL")
                continue
            where_clauses.append(f"{field_name} = ?")
            params.append(value)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM event_records
                {where_sql}
                ORDER BY id ASC
                LIMIT ?
                """,
                (*params, max(int(limit), 1)),
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
        with self.transaction() as conn:
            self._upsert_observer_status(
                conn,
                cursor,
                status=status,
                reason=reason,
                details=details or {},
                now=now,
            )
        # Read back OUTSIDE the transaction: `transaction()` runs on the one
        # connection this thread holds, so a nested one would raise.
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
        with self.connection() as conn:
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
        data["authenticated_principal_id"] = _restore_scope_value(
            data["authenticated_principal_id"]
        )
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

        # Select-then-delete plus a replay-window recompute: one IMMEDIATE
        # transaction, or a concurrent writer's rows can slip between the
        # census and the DELETEs.
        with self.transaction() as conn:
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
                    SELECT id, event_key, dedupe_key, event_id, server_cursor
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
                    SELECT id, event_key, dedupe_key, event_id, server_cursor
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
            last_pruned_row = max(rows, key=lambda row: int(row["id"]))
            last_pruned_cursor = self._row_cursor_marker(last_pruned_row)
            scope_cursor = EventCursor(
                source_authority=source_authority,
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                stream_name=stream_name,
                stream_instance_id=stream_instance_id,
            )

            conn.executemany(
                "DELETE FROM event_presentations WHERE event_key = ?",
                [(key,) for key in event_keys],
            )
            conn.executemany(
                "DELETE FROM event_records WHERE id = ?",
                [(event_id,) for event_id in event_ids],
            )
            conn.executemany(
                "DELETE FROM event_dedupe_records WHERE dedupe_key = ?",
                [(key,) for key in dedupe_keys],
            )
            self._record_replay_prune(
                conn,
                scope_cursor,
                last_pruned_cursor=last_pruned_cursor,
                pruned_event_count=len(rows),
                now=_utc_now(),
            )
        return len(rows)

    def get_replay_window(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventReplayWindow:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )
        with self.connection() as conn:
            return self._get_replay_window_with_connection(conn, cursor)

    def get_replay_status(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
        requested_cursor: str | None = None,
    ) -> dict[str, Any]:
        cursor = EventCursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_name=stream_name,
            stream_instance_id=stream_instance_id,
        )
        with self.connection() as conn:
            window = self._get_replay_window_with_connection(conn, cursor)
            if requested_cursor is None:
                state = (
                    "available"
                    if window.earliest_retained_cursor is not None
                    else "empty"
                )
            elif self._is_cursor_retained(conn, cursor, requested_cursor):
                state = "available"
            elif window.last_pruned_cursor is not None and self._cursor_at_or_before(
                requested_cursor,
                window.last_pruned_cursor,
            ):
                state = "retention_gap"
            elif (
                window.earliest_retained_cursor is not None
                or window.pruned_event_count > 0
            ):
                state = "retention_gap"
            else:
                state = "empty"

        return {
            "state": state,
            "requested_cursor": requested_cursor,
            "earliest_retained_cursor": window.earliest_retained_cursor,
            "latest_retained_cursor": window.latest_retained_cursor,
            "last_pruned_cursor": window.last_pruned_cursor,
            "pruned_event_count": window.pruned_event_count,
            "server_refetch_required": state == "retention_gap",
            "updated_at": window.updated_at,
        }

    def get_retention_policy(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        stream_name: str,
        stream_instance_id: str,
        authenticated_principal_id: str | None = None,
    ) -> EventRetentionPolicy:
        with self.connection() as conn:
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
        with self.transaction() as conn:
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
        # Read back outside the transaction (see `record_observer_status`).
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

        # Census + deletes: the returned counts must describe the rows this
        # call actually removed, so they share one IMMEDIATE transaction.
        with self.transaction() as conn:
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
            replay_window_count = self._count_scoped_rows(
                conn,
                "event_replay_windows",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )

            conn.executemany(
                "DELETE FROM event_presentations WHERE event_key = ?",
                [(key,) for key in event_keys],
            )
            conn.executemany(
                "DELETE FROM event_records WHERE event_key = ?",
                [(key,) for key in event_keys],
            )
            conn.executemany(
                "DELETE FROM event_dedupe_records WHERE dedupe_key = ?",
                [(key,) for key in dedupe_keys],
            )
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
            self._delete_scoped_rows(
                conn,
                "event_replay_windows",
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
            )

        return {
            "events": event_count,
            "dedupe_records": dedupe_count,
            "presentations": presentation_count,
            "processed_cursors": processed_cursor_count,
            "presented_high_water": presented_high_water_count,
            "observer_status": observer_status_count,
            "retention_policies": retention_policy_count,
            "replay_windows": replay_window_count,
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

    @classmethod
    def _sync_replay_window_bounds(
        cls,
        conn: sqlite3.Connection,
        cursor_like: EventCursor | NormalizedEventRecord,
        *,
        now: str,
    ) -> None:
        earliest, latest = cls._retained_cursor_bounds(conn, cursor_like)
        row = cls._replay_window_row(conn, cursor_like)
        cls._upsert_replay_window(
            conn,
            cursor_like,
            earliest_retained_cursor=earliest,
            latest_retained_cursor=latest,
            last_pruned_cursor=row["last_pruned_cursor"] if row is not None else None,
            pruned_event_count=int(row["pruned_event_count"]) if row is not None else 0,
            now=now,
        )

    @classmethod
    def _record_replay_prune(
        cls,
        conn: sqlite3.Connection,
        cursor_like: EventCursor,
        *,
        last_pruned_cursor: str | None,
        pruned_event_count: int,
        now: str,
    ) -> None:
        earliest, latest = cls._retained_cursor_bounds(conn, cursor_like)
        row = cls._replay_window_row(conn, cursor_like)
        existing_count = int(row["pruned_event_count"]) if row is not None else 0
        cls._upsert_replay_window(
            conn,
            cursor_like,
            earliest_retained_cursor=earliest,
            latest_retained_cursor=latest,
            last_pruned_cursor=last_pruned_cursor,
            pruned_event_count=existing_count + int(pruned_event_count),
            now=now,
        )

    @classmethod
    def _get_replay_window_with_connection(
        cls,
        conn: sqlite3.Connection,
        cursor_like: EventCursor,
    ) -> EventReplayWindow:
        row = cls._replay_window_row(conn, cursor_like)
        if row is None:
            earliest, latest = cls._retained_cursor_bounds(conn, cursor_like)
            pruned_count = 0
            last_pruned_cursor = None
            updated_at = None
        else:
            earliest = row["earliest_retained_cursor"]
            latest = row["latest_retained_cursor"]
            pruned_count = int(row["pruned_event_count"])
            last_pruned_cursor = row["last_pruned_cursor"]
            updated_at = row["updated_at"]
        if earliest is not None:
            state = "available"
        elif pruned_count > 0:
            state = "retention_gap"
        else:
            state = "empty"
        return EventReplayWindow(
            source_authority=cursor_like.source_authority,
            server_profile_id=cursor_like.server_profile_id,
            authenticated_principal_id=cursor_like.authenticated_principal_id,
            stream_name=cursor_like.stream_name,
            stream_instance_id=cursor_like.stream_instance_id,
            state=state,
            earliest_retained_cursor=earliest,
            latest_retained_cursor=latest,
            last_pruned_cursor=last_pruned_cursor,
            pruned_event_count=pruned_count,
            updated_at=updated_at,
        )

    @staticmethod
    def _replay_window_row(
        conn: sqlite3.Connection,
        cursor_like: EventCursor | NormalizedEventRecord,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT *
            FROM event_replay_windows
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

    @staticmethod
    def _upsert_replay_window(
        conn: sqlite3.Connection,
        cursor_like: EventCursor | NormalizedEventRecord,
        *,
        earliest_retained_cursor: str | None,
        latest_retained_cursor: str | None,
        last_pruned_cursor: str | None,
        pruned_event_count: int,
        now: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO event_replay_windows (
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id,
                earliest_retained_cursor,
                latest_retained_cursor,
                last_pruned_cursor,
                pruned_event_count,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_authority,
                server_profile_id,
                authenticated_principal_id,
                stream_name,
                stream_instance_id
            )
            DO UPDATE SET
                earliest_retained_cursor = excluded.earliest_retained_cursor,
                latest_retained_cursor = excluded.latest_retained_cursor,
                last_pruned_cursor = excluded.last_pruned_cursor,
                pruned_event_count = excluded.pruned_event_count,
                updated_at = excluded.updated_at
            """,
            (
                cursor_like.source_authority,
                _scope_value(cursor_like.server_profile_id),
                _scope_value(cursor_like.authenticated_principal_id),
                cursor_like.stream_name,
                cursor_like.stream_instance_id,
                earliest_retained_cursor,
                latest_retained_cursor,
                last_pruned_cursor,
                int(pruned_event_count),
                now,
            ),
        )

    @staticmethod
    def _retained_cursor_bounds(
        conn: sqlite3.Connection,
        cursor_like: EventCursor | NormalizedEventRecord,
    ) -> tuple[str | None, str | None]:
        rows = conn.execute(
            """
            SELECT event_id, server_cursor
            FROM event_records
            WHERE source_authority = ?
              AND server_profile_id IS ?
              AND authenticated_principal_id IS ?
              AND stream_name = ?
              AND stream_instance_id = ?
            ORDER BY id ASC
            """,
            (
                cursor_like.source_authority,
                cursor_like.server_profile_id,
                cursor_like.authenticated_principal_id,
                cursor_like.stream_name,
                cursor_like.stream_instance_id,
            ),
        ).fetchall()
        if not rows:
            return None, None
        return EventStateRepository._row_cursor_marker(
            rows[0]
        ), EventStateRepository._row_cursor_marker(rows[-1])

    @staticmethod
    def _is_cursor_retained(
        conn: sqlite3.Connection,
        cursor_like: EventCursor,
        requested_cursor: str,
    ) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM event_records
            WHERE source_authority = ?
              AND server_profile_id IS ?
              AND authenticated_principal_id IS ?
              AND stream_name = ?
              AND stream_instance_id = ?
              AND (server_cursor = ? OR event_id = ?)
            """,
            (
                cursor_like.source_authority,
                cursor_like.server_profile_id,
                cursor_like.authenticated_principal_id,
                cursor_like.stream_name,
                cursor_like.stream_instance_id,
                requested_cursor,
                requested_cursor,
            ),
        ).fetchone()
        return row is not None

    @staticmethod
    def _cursor_at_or_before(requested_cursor: str, boundary_cursor: str) -> bool:
        if requested_cursor == boundary_cursor:
            return True
        if requested_cursor.isdigit() and boundary_cursor.isdigit():
            return int(requested_cursor) <= int(boundary_cursor)
        return False

    @staticmethod
    def _row_cursor_marker(row: sqlite3.Row) -> str | None:
        cursor = row["server_cursor"] or row["event_id"]
        if cursor is None:
            return None
        return str(cursor)

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
            return "source_authority = ? AND server_profile_id = ?", (
                "server",
                server_profile_id,
            )
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
            return "source_authority = ? AND server_profile_id = ?", (
                "server",
                _scope_value(server_profile_id),
            )
        return (
            "source_authority = ? AND server_profile_id = ? AND authenticated_principal_id = ?",
            (
                "server",
                _scope_value(server_profile_id),
                _scope_value(authenticated_principal_id),
            ),
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
        row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} WHERE {table_filter}", params
        ).fetchone()
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
    def _count_matching_presentations(
        conn: sqlite3.Connection, event_keys: list[str]
    ) -> int:
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
