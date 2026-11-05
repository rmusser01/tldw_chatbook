"""
Simple database connection pool for FTS5 searches.
"""

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
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


class _PoolThread(ThreadPoolExecutor):
    """One pool's only thread (TASK-38714).

    The database is built here and every search runs here, so they reuse its
    one thread-local connection, and `close_all_pools()` closes it here: a
    close on any other thread reaches only that thread's own connection.
    Work queued behind the close is refused instead of reopening it.
    """

    def __init__(self) -> None:
        super().__init__(1, thread_name_prefix="rag-fts5-pool")
        self.retired = False  # written and read only on the pool thread

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(self._unless_retired, fn, *args, **kwargs)

    def _unless_retired(self, fn, *args, **kwargs):
        if self.retired:
            raise RuntimeError("media search pool closed")
        return fn(*args, **kwargs)

    def retire(self, db) -> None:
        try:
            db.close_connection()
        finally:
            self.retired = True


# Kept past `close_all_pools()`, so a search still holding a closed pool gets
# its shut-down thread (and an error to retry on), never the default executor.
_pool_threads: "weakref.WeakKeyDictionary[object, _PoolThread]" = (
    weakref.WeakKeyDictionary()
)


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
            thread = _PoolThread()
            try:
                pool = thread.submit(
                    MediaDatabase, db_path=db_path, client_id="rag_service_fts5"
                ).result()
            except BaseException:
                thread.shutdown(wait=False)
                raise
            _pool_threads[pool] = thread
            _connection_pools[db_path] = pool
        return pool


def pool_executor(pool: object) -> Optional[ThreadPoolExecutor]:
    """Return the one thread ``pool``'s searches must run on.

    Args:
        pool: A pool returned by ``get_connection_pool``.

    Returns:
        The pool's own one-thread executor, or ``None`` (the loop's default
        executor) for a pool this module never built, such as a test double.
    """
    try:
        return _pool_threads.get(pool)
    except TypeError:  # a double that takes no weak reference
        return None


def close_all_pools():
    """Close each pool's connection on its own thread, then that thread."""
    with _connection_pools_lock:
        for db_path, db in _connection_pools.items():
            thread = _pool_threads[db]
            try:
                thread.submit(thread.retire, db).result()
            except Exception as e:
                logger.debug(f"Error closing pooled connection for {db_path}: {e}")
            thread.shutdown(wait=True)
        _connection_pools.clear()
        logger.debug("Closed and cleared all connection pool references")
