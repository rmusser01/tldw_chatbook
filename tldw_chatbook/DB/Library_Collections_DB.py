"""SQLite persistence for local Library Collections."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterator, Union

from .base_db import BaseDB


class LibraryCollectionsDB(BaseDB):
    """Database wrapper for Library-owned local Collections.

    task-15466: connections are held per thread (the ``Workspace_DB``
    idiom, itself the ChaChaNotes ``_get_thread_connection`` shape). The
    old ``closing()``-per-use form paid a full private-SQLite open --
    which re-verifies the database file and its three ``-wal``/``-shm``/
    ``-journal`` sidecars every time -- for every single query on a screen
    that reads Collections repeatedly; task-3011 measured the same
    anti-pattern at 1,352 connections during one Console screen push.

    Thread safety: this DB is reached both from the UI thread and from
    ``asyncio.to_thread`` worker pools, and Python's sqlite3 refuses a
    connection used from a thread other than its creator
    (``check_same_thread`` defaults to True). Thread-local storage is what
    makes the held connection safe here -- each thread owns exactly one.
    """

    _CURRENT_SCHEMA_VERSION = 1

    #: Liveness-ping gate (mirrors `Workspace_DB`/`ChaChaNotes_DB`,
    #: task-261/3011): pinging on every call roughly doubles the statement
    #: count on query-heavy paths. A recently-used held connection is
    #: known-good without a ping.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        # Must precede super().__init__: BaseDB.__init__ calls
        # _initialize_schema(), which already needs the held connection.
        self._thread_local = threading.local()
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local collections cache)
        # and avoids an fsync per commit. Unlike journal_mode, which is
        # persisted in the file, synchronous is per-connection -- so it must
        # be re-applied here on every NEW connection, which is exactly why
        # this pairing lives in the one place connections are created
        # (task-15465).
        conn.execute("PRAGMA synchronous = NORMAL")
        # task-3012: a held (long-lived) connection needs true autocommit.
        # Python's default isolation mode auto-BEGINs on any DML, and an
        # implicit transaction accumulated outside `transaction()` makes the
        # explicit BEGIN there fail with "cannot start a transaction within a
        # transaction" -- and silently ROLLS BACK bare DML on close.
        # Audited (task-15466): every write in this file's own module and in
        # `Library/library_collections_service.py` goes through
        # `transaction()`; the only DML outside it is `_initialize_schema`'s
        # `executescript`, which self-commits under either mode.
        conn.isolation_level = None
        return conn

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        The liveness probe is a plain no-op statement; a connection another
        component closed (or that SQLite invalidated) is transparently
        replaced, mirroring `Workspace_DB._held_connection`.
        """
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
        """Yield the thread's held connection (row factory, foreign keys on).

        No transaction is opened: in autocommit mode each statement is its
        own implicit transaction, so single-statement reads cost nothing
        extra and never pin a WAL read snapshot between calls.
        """
        yield self._held_connection()

    @contextmanager
    def read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a read-only snapshot.

        ``BEGIN DEFERRED`` takes no lock until a statement needs one, so a
        block that only reads never acquires the write lock -- while still
        seeing one consistent snapshot across several SELECTs (a caller
        pairing a COUNT with its page needs exactly that). The block always
        ends with ROLLBACK, so a write placed in here would be DISCARDED;
        rather than lose it silently, the block raises when it sees one
        (see Raises). Rolling back an unwritten transaction is free.

        Nesting: this and ``transaction`` both issue an explicit BEGIN on
        the ONE connection this thread holds, so nesting either inside the
        other raises ``sqlite3.OperationalError: cannot start a transaction
        within a transaction``. Pre-port each block had its own connection
        and nesting silently "worked"; the outer block still rolls back
        cleanly, because the failure propagates through its ``except``.

        Yields:
            The thread's held ``sqlite3.Connection`` in a read snapshot.

        Raises:
            RuntimeError: If the block modified any row. The transaction is
                rolled back first, so the write is undone either way -- the
                error exists so the misuse cannot pass unnoticed.
        """
        conn = self._held_connection()
        # total_changes counts rows inserted/updated/deleted on this
        # connection since it was opened, so a difference across the block
        # means a write happened inside a snapshot that is about to be
        # thrown away. It cannot see a DML statement that matched zero rows
        # (nothing to lose in that case) or bare DDL, so this is a guard
        # against silent DATA LOSS, not a general read-only enforcement.
        changes_before = conn.total_changes
        conn.execute("BEGIN DEFERRED")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        wrote = conn.total_changes != changes_before
        conn.rollback()
        if wrote:
            raise RuntimeError(
                "read_transaction() is read-only: a write inside it is "
                "always rolled back. Use transaction() for writes."
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a write transaction.

        ``BEGIN IMMEDIATE`` takes the write lock up front. The MCP server
        opens this same database file in its own process, so a deferred
        transaction that read before writing could fail to upgrade; taking
        the lock immediately removes that class of failure. Pure readers
        must use ``read_transaction`` (or ``connection``) instead.

        Nesting: this and ``read_transaction`` both issue an explicit BEGIN
        on the ONE connection this thread holds, so nesting either inside
        the other raises ``sqlite3.OperationalError: cannot start a
        transaction within a transaction``. Pre-port each block had its own
        connection and nesting silently "worked"; the outer block still
        rolls back cleanly, because the failure propagates through its
        ``except``.

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
        """Close the current thread's held connection, if any."""
        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _initialize_schema(self) -> None:
        """Initialize the local Collections schema."""
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

                CREATE TABLE IF NOT EXISTS library_collections (
                    collection_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS library_collection_items (
                    membership_id TEXT PRIMARY KEY,
                    collection_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(collection_id)
                        REFERENCES library_collections(collection_id)
                        ON DELETE CASCADE,
                    UNIQUE(collection_id, source_type, source_id)
                );
                """
            )
            # No commit: executescript self-commits, and the held connection
            # is in autocommit mode, so there is no transaction to end.

    def get_schema_version(self) -> int:
        """Return the initialized schema version."""
        with self.connection() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0) if row is not None else 0
