"""The one SQLite store kernel: held connections and transactions (ADR-228).

Every store that hand-rolled the held thread-local connection + liveness-ping
idiom and the BEGIN IMMEDIATE ``transaction()`` context manager migrates onto
``SQLiteStoreKernel`` instead (TASK-33132). The semantics here are the hardened
Workspace/RAG lineage, whose lessons are load-bearing:

- a held connection needs true autocommit (``isolation_level = None``) -- Python's
  default isolation mode auto-BEGINs on any DML and then breaks the explicit
  BEGIN (task-15465/15466);
- per-connection PRAGMAs live in the one place connections are created
  (overridable ``_configure_connection``);
- ``transaction()`` rolls back on ``BaseException``, not ``Exception`` -- an
  escaped ``CancelledError``/``KeyboardInterrupt`` used to leave the transaction
  open and silently lose every later write on that thread (task-32801.5, proven
  on Prompts_DB);
- a connection another component closed is transparently replaced by the
  liveness probe.

Participants observability (Backup_Recovery/participants.py) is built in: the
getter and transaction entry points carry the ``_core_getter`` /
``_core_transaction`` decorations and register held connections, so a migrated
store cannot forget the wiring.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from tldw_chatbook.Backup_Recovery.participants import (
    _core_access,
    _core_cached_connection,
    _core_closing,
    _core_getter,
    _core_transaction,
    _register_core_connection,
)
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


class SQLiteStoreKernel:
    """Mixin providing the canonical held-connection and transaction machinery.

    Subclasses MUST set ``_kernel_sqlite_owner`` (their private-sqlite owner
    key, e.g. ``"db.rag_indexing"``) and the ``db_path_str`` /
    ``is_memory_db`` instance attributes, and MUST call
    ``_init_store_kernel()`` before any schema initialization that uses the
    held connection. Per-connection store policy (WAL, synchronous, foreign
    keys...) belongs in an overridden ``_configure_connection``.

    ``:memory:`` policy: uniformly thread-local, matching the Workspace
    template this kernel extracts. A store whose in-memory mode genuinely
    needs one shared connection overrides the accessor and says so in its
    docstring.
    """

    _LIVENESS_PING_IDLE_SECONDS = 30.0

    #: private-sqlite owner key for this store (subclasses must set).
    _kernel_sqlite_owner: str = ""
    #: resolved database path (subclasses must set before first connection).
    db_path_str: str = ""
    #: whether the store targets ``:memory:`` (subclasses must set).
    is_memory_db: bool = False

    def _init_store_kernel(self) -> None:
        """Initialize the per-thread connection state.

        Must run before any schema initialization that uses the held
        connection.
        """
        self._thread_local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """Open and configure a NEW database connection.

        This is the single place a connection is created, so every
        per-connection property lives here or in the store's
        ``_configure_connection`` override.
        """
        conn = connect_private_sqlite(self._kernel_sqlite_owner, self.db_path_str)
        _register_core_connection(self, conn)
        try:
            return self._configure_connection(conn)
        except BaseException:
            conn.close()
            raise

    def _configure_connection(self, conn: sqlite3.Connection) -> sqlite3.Connection:
        """Apply per-connection settings; override for store PRAGMA policy.

        The kernel defaults cover only what the held-connection contract
        itself requires: row access by column name and true autocommit.
        """
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        return conn

    @_core_getter
    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        The liveness probe is a plain no-op statement; a connection another
        component closed (or that SQLite invalidated) is transparently
        replaced.
        """
        _core_access(self)
        conn = getattr(self._thread_local, "conn", None)
        conn = _core_cached_connection(self, conn)
        if conn is not None:
            last_used = getattr(self._thread_local, "conn_last_used", None)
            if (
                last_used is None
                or (time.monotonic() - last_used) >= self._LIVENESS_PING_IDLE_SECONDS
            ):
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    try:
                        sqlite3.Connection.in_transaction.__get__(conn)
                    except sqlite3.ProgrammingError:
                        conn.close()
                    else:
                        raise
                    conn = None
        if conn is None:
            conn = self._get_connection()
            self._thread_local.conn = conn
        self._thread_local.conn_last_used = time.monotonic()
        return conn

    @_core_transaction
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the calling thread's held connection (no transaction).

        In autocommit mode a single statement is its own transaction, so
        reads and single-statement writes need nothing more than this.
        """
        yield self._held_connection()

    @_core_transaction
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the held connection inside a write transaction.

        Required for any block whose statements must land (or not land)
        together: in autocommit mode each bare statement would otherwise
        commit on its own.

        Nesting: the explicit BEGIN runs on the ONE connection this thread
        holds, so nesting a second ``transaction()`` inside one raises
        ``sqlite3.OperationalError: cannot start a transaction within a
        transaction``. The outer block still rolls back cleanly, because
        the failure propagates through its ``except``.

        Raises:
            BaseException: Re-raised after rolling back, on any error inside
                the ``with`` block (including ``CancelledError`` /
                ``KeyboardInterrupt`` -- task-32801.5). On clean exit the
                transaction commits.
        """
        conn = self._held_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        """Close the current thread's held connection, if any."""
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            with _core_closing(self, conn) as allowed:
                if not allowed:
                    return
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - preserve retryable native cache
                    return
                self._thread_local.conn = None
