"""
Simple database connection pool for FTS5 searches.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple
from loguru import logger

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


# Global connection pools: one pooled `MediaDatabase` per db_path, reused
# across searches. `MediaDatabase` already manages a thread-local SQLite
# connection internally (see `_get_thread_connection`/`transaction()`), which
# is the pooling behavior callers (`RAGService._perform_fts5_search`, via its
# `pool.transaction()` call) need -- so the pool IS the `MediaDatabase`
# instance, not a bare path/size record.
#
# TASK-38714: each pool also owns a one-thread executor. The database is
# built on it and every search runs on it, so its one thread-local connection
# is reused by every search and `close_all_pools()` can close it on the
# thread that opened it (a close elsewhere reaches only that thread's own).
_connection_pools: Dict[str, Tuple[MediaDatabase, ThreadPoolExecutor]] = {}

# Qodo PR #1428 finding 3: the dict above was guarded only by an unlocked
# check-then-set (`if db_path not in _connection_pools: ... [construct] ...`).
# `MediaDatabase.__init__` runs schema init immediately, so two threads
# racing the first keyword search for the same path could both pass the
# `not in` check before either stored its instance, each construct their own
# `MediaDatabase`, and the loser's instance (and its open sqlite connection)
# is simply overwritten and leaked. This lock makes first-construction
# single-flight: one thread builds, the rest reuse it.
_connection_pools_lock = threading.Lock()


def get_connection_pool(db_path: str, pool_size: int = 3) -> MediaDatabase:
    """
    Get or create the pooled `MediaDatabase` connection for the given database.

    `pool_size` is accepted for backward API compatibility but unused --
    `MediaDatabase` hands out one connection per calling thread rather than
    capping a fixed pool.
    """
    # Fast path: no lock needed once the pool for this path already exists.
    entry = _connection_pools.get(db_path)
    if entry is not None:
        return entry[0]

    # Slow path: double-checked locking so only one thread ever constructs
    # the `MediaDatabase` for a given path, even under concurrent first
    # callers.
    with _connection_pools_lock:
        entry = _connection_pools.get(db_path)
        if entry is None:
            logger.debug(f"Opening pooled media database connection: {db_path}")
            executor = ThreadPoolExecutor(1, thread_name_prefix="rag-fts5-pool")
            try:
                pool = executor.submit(
                    MediaDatabase, db_path=db_path, client_id="rag_service_fts5"
                ).result()
            except BaseException:
                executor.shutdown(wait=False)
                raise
            entry = _connection_pools[db_path] = (pool, executor)
        return entry[0]


def pool_executor(db_path: str, pool: object) -> Optional[ThreadPoolExecutor]:
    """Return the one thread ``pool``'s searches must run on.

    ``None`` (the loop's default executor) for a pool this module does not
    hold: a test double, or a pool ``close_all_pools()`` already retired.
    """
    entry = _connection_pools.get(db_path)
    return entry[1] if entry is not None and entry[0] is pool else None


def close_all_pools():
    """Close each pool's connection on its own thread, then that thread."""
    with _connection_pools_lock:
        for db_path, (db, executor) in _connection_pools.items():
            try:
                executor.submit(db.close_connection).result()
            except Exception as e:
                logger.debug(f"Error closing pooled connection for {db_path}: {e}")
            executor.shutdown(wait=True)
        _connection_pools.clear()
        logger.debug("Closed and cleared all connection pool references")
