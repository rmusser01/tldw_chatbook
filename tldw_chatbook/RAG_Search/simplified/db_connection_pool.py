"""
Simple database connection pool for FTS5 searches.
"""

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


def get_connection_pool(db_path: str, pool_size: int = 3) -> MediaDatabase:
    """
    Get or create the pooled `MediaDatabase` connection for the given database.

    `pool_size` is accepted for backward API compatibility but unused --
    `MediaDatabase` hands out one connection per calling thread rather than
    capping a fixed pool.
    """
    if db_path not in _connection_pools:
        logger.debug(f"Opening pooled media database connection: {db_path}")
        _connection_pools[db_path] = MediaDatabase(
            db_path=db_path, client_id="rag_service_fts5"
        )

    return _connection_pools[db_path]


def close_all_pools():
    """Close all pooled database connections."""
    for db_path, db in _connection_pools.items():
        try:
            db.close_connection()
        except Exception as e:
            logger.debug(f"Error closing pooled connection for {db_path}: {e}")
    _connection_pools.clear()
    logger.debug("Closed and cleared all connection pool references")
