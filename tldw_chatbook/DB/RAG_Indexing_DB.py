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
import sys
import threading
import time
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Iterable,
    Iterator,
    List,
    Dict,
    Optional,
    Any,
    Sequence,
    Tuple,
    Union,
)
from loguru import logger
from ..Metrics.metrics_logger import log_counter, log_histogram
from .private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.private_paths import lexical_path
from tldw_chatbook.Utils.timestamps import utc_now_iso
from tldw_chatbook.Backup_Recovery.participants import (
    _core_access,
    _core_cached_connection,
    _core_closing,
    _core_getter,
    _core_transaction,
    _register_core_connection,
)


#: Shared by the single-item and batch write paths so the two can never
#: drift into writing different columns.
_MARK_INDEXED_SQL = """
INSERT OR REPLACE INTO indexed_items
(item_id, item_type, last_indexed, last_modified, chunk_count, metadata)
VALUES (?, ?, ?, ?, ?, ?)
"""

#: ADR-223 / TASK-34420: hard ceiling on persisted embedding-cache rows.
#: 200k rows of a 384-dim float32 vector (~1.5 KB blob + ~140 B of
#: key/timestamp) is ~340 MB in the user data dir -- a defensible ceiling
#: for a rebuildable cache; 768-dim models double it, which is when the
#: cap is doing its job. Eviction is FIFO by ``(created_at, rowid)`` inside
#: the insert transaction; see backlog/decisions/223-*.md for the math and
#: why FIFO over LRU (no write amplification on the read path).
EMBEDDING_CACHE_MAX_ROWS = 200_000

#: IN-clause chunk for cache lookups: SQLite's default host-parameter
#: ceiling is 999, so 500 placeholders stays under it with headroom.
EMBEDDING_CACHE_LOOKUP_CHUNK = 500

# Tracking reads are bounded to the current ingestion batch, with one extra
# parameter for item_type inside SQLite's conservative 999-parameter limit.
INDEXED_ITEMS_LOOKUP_CHUNK = 500

#: Shared by the cache write path (one executemany, one transaction).
_STORE_EMBEDDINGS_SQL = """
INSERT OR REPLACE INTO embedding_cache
(model_id, content_hash, vector, created_at)
VALUES (?, ?, ?, ?)
"""


def _encode_embedding_vector(vector: Sequence[float]) -> bytes:
    """Serialize an embedding as little-endian raw float32 bytes (ADR-223).

    stdlib ``array`` (not numpy) keeps the DB layer dependency-free; the
    explicit byteswap makes big-endian hosts still write little-endian,
    which is the on-disk format this table commits to.
    """
    packed = array("f", vector)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _decode_embedding_vector(blob: bytes) -> Optional[List[float]]:
    """Decode a cached vector blob; ``None`` when the blob is corrupt.

    A row whose blob length is not a multiple of 4 cannot be float32 data;
    the caller treats it as absent (a corrupt cache row must never crash
    indexing).
    """
    if len(blob) % 4:
        return None
    packed = array("f")
    packed.frombytes(blob)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tolist()


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

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str = "default",
        embedding_cache_max_rows: int = EMBEDDING_CACHE_MAX_ROWS,
    ):
        """
        Initialize the RAG indexing database.

        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier (for future multi-client support)
            embedding_cache_max_rows: Row-count cap for the embedding
                cache table (ADR-223); tests shrink it to exercise
                eviction.
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
        self.embedding_cache_max_rows = int(embedding_cache_max_rows)

        # Must precede _initialize_schema(): it already uses the held
        # connection.
        self._thread_local = threading.local()

        self._initialize_schema()

    @_core_getter
    def _get_connection(self) -> sqlite3.Connection:
        """Open and configure a NEW database connection with row factory.

        Callers wanting the thread's long-lived connection should use
        ``connection``/``transaction``; this is the single place a
        connection is created, so every per-connection property lives here.
        """
        _core_access(self)
        conn = connect_private_sqlite("db.rag_indexing", self.db_path_str)
        _register_core_connection(self, conn)
        try:
            return self._configure_connection(conn)
        except BaseException:
            conn.close()
            raise

    def _configure_connection(self, conn):
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
        # task-19566 F11: no FKs are declared in this indexing-state cache
        # today, but SQLite enforces foreign keys per connection and defaults
        # to OFF -- enable it here (the one place connections are configured)
        # so a future schema change that declares one is enforced, not inert.
        conn.execute("PRAGMA foreign_keys = ON")
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

    @_core_getter
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
        # TASK-32801.5: BaseException, not Exception. A CancelledError or
        # KeyboardInterrupt raised inside the block is not an Exception, so
        # the rollback used to be skipped and the transaction stayed open --
        # and because `transaction()` treats an already-open transaction as
        # nested, every later write on that thread then rode it uncommitted
        # and was lost at close. Proven on Prompts_DB: a write after an
        # escaped KeyboardInterrupt did not survive close+reopen.
        # Library_Collections_DB and Workflows_DB already do this.
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

        -- ADR-223 / TASK-34420: persistent content-hash embedding cache.
        -- Keyed by (model_id, sha256(text)) so an unchanged re-embed after
        -- a restart is a table read instead of a provider/model embed call.
        -- This DB has no schema-version chain (no PRAGMA user_version, no
        -- DB/migrations entry); idempotent CREATE IF NOT EXISTS on open is
        -- its migration convention, and the table is purely additive -- old
        -- readers ignore it, new readers create it on open, and it is never
        -- authoritative (rebuildable from source content).
        CREATE TABLE IF NOT EXISTS embedding_cache (
            model_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (model_id, content_hash)
        );

        -- Supports the eviction prune (oldest-first by created_at).
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_created
        ON embedding_cache(created_at);
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

    def get_cached_embeddings(
        self, model_id: str, content_hashes: Sequence[str]
    ) -> Dict[str, List[float]]:
        """Look up cached embedding vectors by content hash (ADR-223).

        Reads are chunked into ``IN (...)`` queries of at most
        ``EMBEDDING_CACHE_LOOKUP_CHUNK`` placeholders (SQLite's default
        host-parameter ceiling is 999). Duplicate hashes are collapsed
        before querying. A row whose vector blob fails to decode is
        skipped (logged): a corrupt cache row must be treated as a miss,
        never crash the caller.

        Args:
            model_id: Embedding model identity the vectors belong to.
            content_hashes: sha256 hex digests of the texts being embedded.

        Returns:
            Mapping of ``content_hash -> vector`` for every requested hash
            that has a cached, decodable vector.
        """
        if not content_hashes:
            return {}

        unique_hashes = list(dict.fromkeys(content_hashes))
        found: Dict[str, List[float]] = {}
        start_time = time.time()
        try:
            with self.connection() as conn:
                for start in range(0, len(unique_hashes), EMBEDDING_CACHE_LOOKUP_CHUNK):
                    chunk = unique_hashes[
                        start : start + EMBEDDING_CACHE_LOOKUP_CHUNK
                    ]
                    placeholders = ",".join("?" * len(chunk))
                    cursor = conn.execute(
                        "SELECT content_hash, vector FROM embedding_cache "
                        f"WHERE model_id = ? AND content_hash IN ({placeholders})",
                        (model_id, *chunk),
                    )
                    for row in cursor:
                        vector = _decode_embedding_vector(row["vector"])
                        if vector is None:
                            logger.warning(
                                "embedding_cache: corrupt vector blob; treating as a miss"
                            )
                            continue
                        found[row["content_hash"]] = vector
        except Exception as e:
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "embedding_cache_lookup",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Error looking up cached embeddings (error_type={type(e).__name__})"
            )
            raise

        log_histogram(
            "rag_indexing_db_operation_duration",
            time.time() - start_time,
            labels={"operation": "embedding_cache_lookup"},
        )
        log_counter(
            "rag_indexing_db_operation_count",
            labels={
                "operation": "embedding_cache_lookup",
                "status": "success",
                "requested": str(len(unique_hashes)),
                "hits": str(len(found)),
            },
        )
        return found

    def store_cached_embeddings(
        self, model_id: str, rows: Sequence[Tuple[str, Sequence[float]]]
    ) -> None:
        """Persist embedding vectors keyed by content hash (ADR-223).

        One ``executemany`` of ``INSERT OR REPLACE`` inside one
        transaction, so a batch lands atomically; if the table is over
        ``embedding_cache_max_rows`` after the insert, the overflow is
        pruned oldest-first (``created_at`` then ``rowid`` for
        deterministic FIFO within a same-timestamp batch) **in the same
        transaction**. Duplicate hashes within one call collapse (last
        write wins). Vectors are stored as little-endian float32 bytes.

        Args:
            model_id: Embedding model identity the vectors belong to.
            rows: ``(content_hash, vector)`` pairs to persist.

        Raises:
            Exception: Re-raised after the transaction rolls back.
        """
        by_hash: Dict[str, Sequence[float]] = {}
        for content_hash, vector in rows:
            by_hash[content_hash] = vector
        if not by_hash:
            return

        now = utc_now_iso()
        payload = [
            (model_id, content_hash, _encode_embedding_vector(vector), now)
            for content_hash, vector in by_hash.items()
        ]
        start_time = time.time()
        try:
            with self.transaction() as conn:
                conn.executemany(_STORE_EMBEDDINGS_SQL, payload)
                overflow = conn.execute(
                    "SELECT COUNT(*) - ? FROM embedding_cache",
                    (self.embedding_cache_max_rows,),
                ).fetchone()[0]
                if overflow > 0:
                    conn.execute(
                        "DELETE FROM embedding_cache WHERE rowid IN ("
                        "SELECT rowid FROM embedding_cache "
                        "ORDER BY created_at ASC, rowid ASC LIMIT ?)",
                        (overflow,),
                    )
        except Exception as e:
            log_counter(
                "rag_indexing_db_operation_count",
                labels={
                    "operation": "embedding_cache_store",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Error storing {len(payload)} cached embedding(s) "
                f"(error_type={type(e).__name__})"
            )
            raise

        log_histogram(
            "rag_indexing_db_operation_duration",
            time.time() - start_time,
            labels={"operation": "embedding_cache_store"},
        )
        log_counter(
            "rag_indexing_db_operation_count",
            labels={
                "operation": "embedding_cache_store",
                "status": "success",
                "batch_size": str(len(payload)),
            },
        )

    def get_indexed_items_by_ids(
        self, item_type: str, item_ids: Sequence[str]
    ) -> Dict[str, datetime]:
        """Read modification timestamps for only the requested tracking IDs.

        Args:
            item_type: Type of the requested items.
            item_ids: Incoming batch IDs; duplicate IDs are read once.

        Returns:
            Existing IDs mapped to their stored timestamps, retaining offsets.
        """
        unique_ids = list(dict.fromkeys(item_ids))
        if not unique_ids:
            return {}
        found: Dict[str, datetime] = {}
        with self.connection() as conn:
            for start in range(0, len(unique_ids), INDEXED_ITEMS_LOOKUP_CHUNK):
                chunk = unique_ids[start : start + INDEXED_ITEMS_LOOKUP_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    "SELECT item_id, last_modified FROM indexed_items "
                    f"WHERE item_type = ? AND item_id IN ({placeholders})",
                    (item_type, *chunk),
                )
                for row in cursor:
                    found[row["item_id"]] = datetime.fromisoformat(row["last_modified"])
        return found

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
        # Multiple statements: under autocommit they would commit
        # independently, so an explicit transaction keeps the wipe atomic.
        # The embedding cache (ADR-223) is rebuildable tracking state too --
        # a cache reset that left half the cache behind would defeat itself.
        with self.transaction() as conn:
            conn.execute("DELETE FROM indexed_items")
            conn.execute("DELETE FROM collection_state")
            conn.execute("DELETE FROM embedding_cache")
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

        # Compare timestamps. The stored value is always tz-aware (the write path
        # in sqlite_datetime_fix stamps UTC onto naive input), so normalize a naive
        # caller value the same way instead of raising "can't compare offset-naive
        # and offset-aware datetimes".
        last_modified = datetime.fromisoformat(info["last_modified"])
        if current_modified.tzinfo is None:
            current_modified = current_modified.replace(tzinfo=timezone.utc)
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
