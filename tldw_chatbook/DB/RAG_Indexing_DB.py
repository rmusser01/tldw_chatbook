# RAG_Indexing_DB.py
# Description: Database module for tracking RAG indexing state
#
"""
RAG_Indexing_DB.py
------------------

A SQLite-based module for tracking the state of RAG indexing operations.
This module provides functionality to:
- Track which items have been indexed and when
- Support incremental indexing by tracking last_modified timestamps
- Manage indexing state across different content types (media, conversations, notes)

The module uses a simple schema that tracks:
- Item ID and type
- Last indexed timestamp
- Last known modification timestamp
- Indexing status and metadata
"""

import sqlite3
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Optional, Any, Tuple, Union
from loguru import logger
from ..Metrics.metrics_logger import log_counter, log_histogram
from .private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.private_paths import lexical_path


#: Shared by the single-item and batch write paths so the two can never
#: drift into writing different columns.
_MARK_INDEXED_SQL = """
INSERT OR REPLACE INTO indexed_items
(item_id, item_type, last_indexed, last_modified, chunk_count, metadata)
VALUES (?, ?, ?, ?, ?, ?)
"""


class RAGIndexingDB:
    """
    Manages SQLite database for tracking RAG indexing state.

    This class provides methods to track which items have been indexed,
    when they were indexed, and their last known modification times.

    task-15466: connections are held per thread (the ``Workspace_DB``
    idiom). The previous shape opened a brand-new private-SQLite
    connection -- a file + three-sidecar verification each time -- for
    every operation, including once per item marked during a batch index,
    and never closed any of them (``with conn`` is sqlite3's TRANSACTION
    context manager, not a closing one, so they leaked until GC).

    Thread safety: indexing runs on worker threads while the UI may read
    stats from the loop thread, and sqlite3 refuses a connection used off
    its creating thread (``check_same_thread`` defaults to True).
    Thread-local storage is what makes a held connection safe here: each
    thread owns exactly one, so the live connection count is bounded by
    the number of threads that touch this DB rather than by call volume.
    """

    #: Liveness-ping gate (mirrors `Workspace_DB`/`ChaChaNotes_DB`,
    #: task-261/3011): a per-call ``SELECT 1`` would double the statement
    #: count on the per-item indexing path. A recently-used held
    #: connection is known-good without a ping.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: Union[str, Path], client_id: str = "default"):
        """
        Initialize the RAG indexing database.

        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier (for future multi-client support)
        """
        # Handle path types consistently
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = lexical_path(db_path)
        else:
            self.is_memory_db = db_path == ":memory:"
            self.db_path = (
                lexical_path(db_path) if not self.is_memory_db else Path(":memory:")
            )

        self.db_path_str = str(self.db_path) if not self.is_memory_db else ":memory:"
        self.client_id = client_id

        # Must precede _initialize_schema(): it already uses the held
        # connection.
        self._thread_local = threading.local()

        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Open and configure a NEW database connection with row factory.

        Callers wanting the thread's long-lived connection should use
        ``connection``/``transaction``; this is the single place a
        connection is created, so every per-connection property lives here.
        """
        conn = connect_private_sqlite("db.rag_indexing", self.db_path_str)
        conn.row_factory = sqlite3.Row
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local indexing-state
        # cache -- it is rebuilt from source content, never authoritative)
        # and avoids an fsync per commit. Unlike journal_mode, which is
        # persisted in the file, synchronous is per-connection, so it must be
        # re-applied on every NEW connection (task-15465) -- which is why
        # this pairing lives in the one place connections are created.
        conn.execute("PRAGMA synchronous = NORMAL")
        # task-3012: a held (long-lived) connection needs true autocommit.
        # Python's default isolation mode auto-BEGINs on any DML; that
        # implicit transaction then makes the explicit BEGIN in
        # `transaction()` raise "cannot start a transaction within a
        # transaction", and silently ROLLS BACK bare DML on close.
        # Audited (task-15466) -- every site in this file: `_initialize_
        # schema` executescript (self-commits either way), single-statement
        # writes in mark_item_indexed / remove_indexed_item /
        # update_collection_state (each its own autocommit transaction),
        # multi-statement writes in clear_all and mark_items_indexed (both
        # now wrapped in an explicit `transaction()`), and read-only SELECTs
        # elsewhere.
        conn.isolation_level = None
        # task-15465 left a WAL caution here: a lingering never-closed
        # reader pins the WAL and blocks checkpoint truncation, so the old
        # GC-only lifecycle risked unbounded -wal growth. This port is the
        # structural fix that comment pointed at -- connections are now
        # per-thread and finite, and because autocommit ends each
        # statement's implicit read transaction immediately, an idle held
        # connection holds no read snapshot and does not block checkpointing.
        return conn

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        The liveness probe is a plain no-op statement; a connection another
        component closed (or that SQLite invalidated) is transparently
        replaced, mirroring `Workspace_DB._held_connection`.

        ``:memory:`` asymmetry, deliberate: unlike ``ClientNotificationsDB``
        -- whose in-memory branch keeps ONE shared connection because the
        app really does fall back to an in-memory inbox -- this class stays
        uniformly thread-local, matching the ``Workspace_DB`` template it
        was ported from. The consequence is that with ``:memory:`` a SECOND
        thread gets its own connection and therefore its own schema-less
        database (an in-memory DB lives inside its connection, and
        ``_initialize_schema`` only ran on the constructing thread's).
        That is acceptable because production always constructs this DB
        from ``get_rag_indexing_db_path()``; ``:memory:`` exists here only
        for single-threaded tests. It is also not a regression -- before
        this port EVERY call opened a fresh, empty in-memory database, so
        no operation after construction could see the schema at all.
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
        """Yield the calling thread's held connection (no transaction).

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
        holds, so nesting a second ``transaction()`` inside one raises
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
        """Close the current thread's held connection, if any."""
        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _initialize_schema(self):
        """Initialize the database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS indexed_items (
            item_id TEXT NOT NULL,
            item_type TEXT NOT NULL,
            last_indexed DATETIME NOT NULL,
            last_modified DATETIME NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            metadata TEXT,
            PRIMARY KEY (item_id, item_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_indexed_items_type 
        ON indexed_items(item_type);
        
        CREATE INDEX IF NOT EXISTS idx_indexed_items_modified 
        ON indexed_items(last_modified);
        
        CREATE INDEX IF NOT EXISTS idx_indexed_items_indexed 
        ON indexed_items(last_indexed);
        
        -- Table for tracking collection states
        CREATE TABLE IF NOT EXISTS collection_state (
            collection_name TEXT PRIMARY KEY,
            last_full_index DATETIME,
            total_items INTEGER DEFAULT 0,
            indexed_items INTEGER DEFAULT 0,
            metadata TEXT
        );
        """

        with self.connection() as conn:
            # executescript self-commits under autocommit; no explicit
            # commit is needed (and none is possible outside a transaction).
            conn.executescript(schema)

    def mark_item_indexed(
        self,
        item_id: str,
        item_type: str,
        last_modified: datetime,
        chunk_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Mark an item as indexed.

        For a whole batch, prefer ``mark_items_indexed``: this method is a
        single autocommit statement, so N calls mean N commits.

        Args:
            item_id: Unique identifier for the item
            item_type: Type of item (media, conversation, note)
            last_modified: Last modification timestamp of the item
            chunk_count: Number of chunks created for this item
            metadata: Optional metadata about the indexing
        """
        start_time = time.time()

        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with self.connection() as conn:
                conn.execute(
                    _MARK_INDEXED_SQL,
                    (
                        item_id,
                        item_type,
                        now,
                        last_modified,
                        chunk_count,
                        metadata_json,
                    ),
                )

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "mark_indexed",
                    "item_type": item_type,
                    "chunk_count": str(chunk_count),
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "mark_indexed",
                    "item_type": item_type,
                    "status": "success",
                },
            )
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "mark_indexed",
                    "item_type": item_type,
                    "chunk_count": str(chunk_count),
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "mark_indexed",
                    "item_type": item_type,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Error marking item indexed (error_type={type(e).__name__})")
            raise

    def mark_items_indexed(
        self,
        items: Iterable[
            Tuple[str, str, datetime, int]
            | Tuple[str, str, datetime, int, Optional[Dict[str, Any]]]
        ],
    ) -> int:
        """Mark a whole batch of items as indexed in ONE transaction.

        The batch indexer used to call ``mark_item_indexed`` once per
        successful document, which meant one connection open and one
        commit (one fsync under the pre-task-15465 ``synchronous=FULL``)
        per item. This writes the batch with a single ``executemany``
        inside a single transaction, so the batch lands atomically: after
        an interruption, either every item in it is tracked as indexed or
        none is, and the untracked ones are simply re-indexed next run.

        Args:
            items: Tuples of ``(item_id, item_type, last_modified,
                chunk_count)`` with an optional fifth ``metadata`` mapping.

        Returns:
            The number of rows written.

        Raises:
            Exception: Re-raised after the transaction rolls back.
        """
        start_time = time.time()
        now = datetime.now(timezone.utc)
        rows: List[Tuple[Any, ...]] = []
        for item in items:
            item_id, item_type, last_modified, chunk_count = item[:4]
            metadata = item[4] if len(item) > 4 else None
            rows.append(
                (
                    item_id,
                    item_type,
                    now,
                    last_modified,
                    chunk_count,
                    json.dumps(metadata) if metadata else None,
                )
            )
        if not rows:
            return 0

        try:
            with self.transaction() as conn:
                conn.executemany(_MARK_INDEXED_SQL, rows)
        except Exception as e:
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "mark_indexed_batch",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Error marking {len(rows)} item(s) indexed "
                f"(error_type={type(e).__name__})"
            )
            raise

        log_histogram(
            "rag_indexing_db_operation_duration",
            time.time() - start_time,
            labels={"operation": "mark_indexed_batch"},
        )
        log_counter(
            "rag_indexing_db_operation_count",
            labels={
                "operation": "mark_indexed_batch",
                "status": "success",
                "batch_size": str(len(rows)),
            },
        )
        return len(rows)

    def get_items_to_index(
        self, item_type: str, modified_since: Optional[datetime] = None
    ) -> List[str]:
        """
        Get list of item IDs that need indexing.

        This method is used by the indexing service to determine which items
        are new or have been modified since last indexing.

        Args:
            item_type: Type of items to check
            modified_since: Only return items modified after this timestamp

        Returns:
            List of item IDs that need indexing
        """
        # This will be implemented by the indexing service
        # by comparing with the source database
        return []

    def get_indexed_item_info(
        self, item_id: str, item_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get indexing information for a specific item.

        Args:
            item_id: Item identifier
            item_type: Type of item

        Returns:
            Dictionary with indexing information or None if not indexed
        """
        start_time = time.time()

        query = """
        SELECT * FROM indexed_items 
        WHERE item_id = ? AND item_type = ?
        """

        try:
            with self.connection() as conn:
                cursor = conn.execute(query, (item_id, item_type))
                row = cursor.fetchone()

                result = None
                if row:
                    result = {
                        "item_id": row["item_id"],
                        "item_type": row["item_type"],
                        "last_indexed": row["last_indexed"],
                        "last_modified": row["last_modified"],
                        "chunk_count": row["chunk_count"],
                        "metadata": json.loads(row["metadata"])
                        if row["metadata"]
                        else None,
                    }

                # Log success metrics
                duration = time.time() - start_time
                log_histogram(
                    "rag_indexing_db_operation_duration",
                    duration,
                    labels={
                        "operation": "get_item_info",
                        "item_type": item_type,
                        "found": "true" if result else "false",
                    },
                )
                log_counter(
                    "rag_indexing_db_operation_count",
                    labels={
                        "operation": "get_item_info",
                        "item_type": item_type,
                        "status": "success",
                        "found": "true" if result else "false",
                    },
                )

                return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "get_item_info",
                    "item_type": item_type,
                    "found": "false",
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "get_item_info",
                    "item_type": item_type,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Error getting indexed item info: {e}")
            raise

    def get_indexed_items_by_type(self, item_type: str) -> Dict[str, datetime]:
        """
        Get all indexed items of a specific type with their last modified times.

        Args:
            item_type: Type of items to retrieve

        Returns:
            Dictionary mapping item_id to last_modified timestamp
        """
        query = """
        SELECT item_id, last_modified FROM indexed_items 
        WHERE item_type = ?
        """

        with self.connection() as conn:
            cursor = conn.execute(query, (item_type,))
            return {
                row["item_id"]: datetime.fromisoformat(row["last_modified"])
                for row in cursor
            }

    def remove_indexed_item(self, item_id: str, item_type: str):
        """
        Remove an item from the indexed items tracking.

        Args:
            item_id: Item identifier
            item_type: Type of item
        """
        start_time = time.time()

        query = "DELETE FROM indexed_items WHERE item_id = ? AND item_type = ?"

        try:
            with self.connection() as conn:
                # Single statement: autocommit already makes it its own
                # transaction, so no explicit commit is needed.
                cursor = conn.execute(query, (item_id, item_type))
                rows_affected = cursor.rowcount

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "remove_item",
                    "item_type": item_type,
                    "found": "true" if rows_affected > 0 else "false",
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "remove_item",
                    "item_type": item_type,
                    "status": "success",
                    "found": "true" if rows_affected > 0 else "false",
                },
            )
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "remove_item",
                    "item_type": item_type,
                    "found": "false",
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "remove_item",
                    "item_type": item_type,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Error removing indexed item: {e}")
            raise

    def update_collection_state(
        self,
        collection_name: str,
        total_items: int,
        indexed_items: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Update the state of a collection.

        Args:
            collection_name: Name of the collection (e.g., 'media_chunks')
            total_items: Total number of items in the source
            indexed_items: Number of items indexed
            metadata: Optional metadata about the collection
        """
        start_time = time.time()

        query = """
        INSERT OR REPLACE INTO collection_state 
        (collection_name, last_full_index, total_items, indexed_items, metadata)
        VALUES (?, ?, ?, ?, ?)
        """

        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with self.connection() as conn:
                conn.execute(
                    query,
                    (collection_name, now, total_items, indexed_items, metadata_json),
                )

            # Log success metrics
            duration = time.time() - start_time
            completion_rate = (
                (indexed_items / total_items * 100) if total_items > 0 else 0
            )
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "update_collection_state",
                    "collection": collection_name,
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "update_collection_state",
                    "collection": collection_name,
                    "status": "success",
                },
            )
            log_histogram(
                "rag_indexing_db_collection_completion_rate",
                completion_rate,
                labels={"collection": collection_name},
            )
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "rag_indexing_db_operation_duration",
                duration,
                labels={
                    "operation": "update_collection_state",
                    "collection": collection_name,
                },
            )
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "update_collection_state",
                    "collection": collection_name,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Error updating collection state: {e}")
            raise

    def get_collection_state(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection state or None
        """
        query = "SELECT * FROM collection_state WHERE collection_name = ?"

        with self.connection() as conn:
            cursor = conn.execute(query, (collection_name,))
            row = cursor.fetchone()

            if row:
                return {
                    "collection_name": row["collection_name"],
                    "last_full_index": row["last_full_index"],
                    "total_items": row["total_items"],
                    "indexed_items": row["indexed_items"],
                    "metadata": json.loads(row["metadata"])
                    if row["metadata"]
                    else None,
                }
            return None

    def get_indexing_stats(self) -> Dict[str, Any]:
        """
        Get overall indexing statistics.

        Returns:
            Dictionary with indexing statistics
        """
        stats = {"total_indexed": 0, "by_type": {}, "collections": {}}

        with self.connection() as conn:
            # Get counts by type
            cursor = conn.execute("""
                SELECT item_type, COUNT(*) as count 
                FROM indexed_items 
                GROUP BY item_type
            """)

            for row in cursor:
                stats["by_type"][row["item_type"]] = row["count"]
                stats["total_indexed"] += row["count"]

            # Get collection states
            cursor = conn.execute("SELECT * FROM collection_state")
            for row in cursor:
                stats["collections"][row["collection_name"]] = {
                    "last_full_index": row["last_full_index"],
                    "total_items": row["total_items"],
                    "indexed_items": row["indexed_items"],
                }

        return stats

    def clear_all(self):
        """Clear all indexing tracking data."""
        # Two statements: under autocommit they would commit
        # independently, so an explicit transaction keeps the wipe atomic.
        with self.transaction() as conn:
            conn.execute("DELETE FROM indexed_items")
            conn.execute("DELETE FROM collection_state")
        logger.warning("Cleared all RAG indexing tracking data")

    def is_item_indexed(self, item_id: str, item_type: str) -> bool:
        """
        Check if an item is indexed.

        Args:
            item_id: Item identifier
            item_type: Type of item

        Returns:
            True if item is indexed, False otherwise
        """
        info = self.get_indexed_item_info(item_id, item_type)
        return info is not None

    def needs_reindexing(
        self, item_id: str, item_type: str, current_modified: datetime
    ) -> bool:
        """
        Check if an item needs reindexing based on modification time.

        Args:
            item_id: Item identifier
            item_type: Type of item
            current_modified: Current modification timestamp of the item

        Returns:
            True if item needs reindexing, False otherwise
        """
        info = self.get_indexed_item_info(item_id, item_type)
        if not info:
            return True  # Not indexed yet

        # Compare timestamps
        last_modified = datetime.fromisoformat(info["last_modified"])
        return current_modified > last_modified

    def remove_item(self, item_id: str, item_type: str) -> bool:
        """
        Remove an item from indexing tracking.

        Args:
            item_id: Item identifier
            item_type: Type of item

        Returns:
            True if item was removed, False if it didn't exist
        """
        if not self.is_item_indexed(item_id, item_type):
            return False

        self.remove_indexed_item(item_id, item_type)
        return True
