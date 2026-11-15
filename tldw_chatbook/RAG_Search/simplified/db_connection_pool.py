"""
Simple database connection pool for FTS5 searches.
"""

import asyncio
import threading
import time
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
        if self.retired or maintenance._closed:
            raise RuntimeError("media search pool closed")
        return fn(*args, **kwargs)

    def retire(self, db) -> None:
        try:
            db.close_connection()
            if getattr(db._local, "conn", None) is not None:
                raise RuntimeError("media search pool close refused")
        finally:
            self.retired = True


# Kept past `close_all_pools()`, so a search still holding a closed pool gets
# its shut-down thread (and an error to retry on), never the default executor.
_pool_threads: "weakref.WeakKeyDictionary[object, _PoolThread]" = (
    weakref.WeakKeyDictionary()
)


# The services using each pool (TASK-38956). Pools are shared by db_path, so a
# service's close retires only the pools no other live service still uses.
_pool_users: "dict[str, weakref.WeakSet[object]]" = {}


def get_connection_pool(
    db_path: str, pool_size: int = 3, *, user: object = None
) -> MediaDatabase:
    """
    Get or create the pooled `MediaDatabase` connection for the given database.

    `pool_size` is accepted for backward API compatibility but unused --
    `MediaDatabase` hands out one connection per calling thread rather than
    capping a fixed pool. A `user` keeps the pool open until it calls
    `release_pools`.
    """
    if maintenance._closed:
        raise RuntimeError("media search pools paused for maintenance")
    # Fast path: no lock needed once the pool for this path already exists.
    pool = _connection_pools.get(db_path)
    if pool is not None and (user is None or user in _pool_users.get(db_path, ())):
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
        if user is not None:
            _pool_users.setdefault(db_path, weakref.WeakSet()).add(user)
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


def _retire_pool(db: MediaDatabase) -> None:
    """Close one native pool and its executor before unregistering it."""
    thread = _pool_threads[db]
    # Retirement remains admissible after searches have been refused.
    ThreadPoolExecutor.submit(thread, thread.retire, db).result()
    thread.shutdown(wait=True)


def release_pools(user: object) -> None:
    """Retire a last user's pool only after physical close succeeds."""
    with _connection_pools_lock:
        for db_path, users in list(_pool_users.items()):
            if user not in users:
                continue
            users.discard(user)
            if users:
                continue
            db = _connection_pools.get(db_path)
            if db is not None:
                _retire_pool(db)
                del _connection_pools[db_path]
            del _pool_users[db_path]


def close_all_pools():
    """Close each pool's connection on its own thread, then that thread."""
    with _connection_pools_lock:
        while _connection_pools:
            db_path, db = next(iter(_connection_pools.items()))
            _retire_pool(db)
            del _connection_pools[db_path]
            _pool_users.pop(db_path, None)
        _pool_users.clear()
        logger.debug("Closed and cleared all connection pool references")


class PoolMaintenance:
    """Retain physical SQL pool cleanup after all upstream producers settle."""

    def __init__(self) -> None:
        self._closed = False
        self._retirement: asyncio.Task[None] | None = None

    def _maintenance_close_admission(self) -> None:
        self._closed = True

    async def _maintenance_drain(self, deadline: float) -> bool:
        if not self._closed:
            raise RuntimeError("media search pool intake is open")
        if self._retirement is None:
            # Construction owns this lock before publishing its native handle.
            if _connection_pools_lock.acquire(blocking=False):
                try:
                    if not _connection_pools:
                        return True
                finally:
                    _connection_pools_lock.release()
            self._retirement = asyncio.create_task(asyncio.to_thread(close_all_pools))
        done, _ = await asyncio.wait(
            {self._retirement}, timeout=max(0.0, deadline - time.monotonic())
        )
        if not done:
            return False
        self._retirement.result()
        return True

    async def _maintenance_resume(self) -> None:
        if self._retirement is not None:
            await asyncio.gather(
                asyncio.shield(self._retirement), return_exceptions=True
            )
        self._retirement = None
        self._closed = False


maintenance = PoolMaintenance()
