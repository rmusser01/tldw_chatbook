"""
Simple database connection pool for FTS5 searches.
"""

import threading
from typing import Dict
from loguru import logger

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


# Global connection pools: one pooled `MediaDatabase` per db_path, reused
# across searches. `MediaDatabase` already manages a thread-local SQLite
# connection internally (see `_get_thread_connection`/`transaction()`), which
# is the pooling behavior callers (`RAGService._perform_fts5_search`, via its
# `pool.transaction()` call) need -- so the pool IS the `MediaDatabase`
# instance, not a bare path/size record.
_connection_pools: Dict[str, MediaDatabase] = {}

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
    pool = _connection_pools.get(db_path)
    if pool is not None:
        return pool

    # Slow path: double-checked locking so only one thread ever constructs
    # the `MediaDatabase` for a given path, even under concurrent first
    # callers.
    with _connection_pools_lock:
        pool = _connection_pools.get(db_path)
        if pool is None:
            logger.debug(f"Opening pooled media database connection: {db_path}")
            pool = MediaDatabase(db_path=db_path, client_id="rag_service_fts5")
            _connection_pools[db_path] = pool
        return pool


def close_all_pools():
    """Close all pooled database connections."""
    with _connection_pools_lock:
        for db_path, db in _connection_pools.items():
            try:
                db.close_connection()
            except Exception as e:
                logger.debug(f"Error closing pooled connection for {db_path}: {e}")
        _connection_pools.clear()
        logger.debug("Closed and cleared all connection pool references")
