"""Run synchronous ``SubscriptionsDB`` work off the event loop.

task-15463. A scheduled watchlist check used to do all of its sqlite
bookkeeping -- the subscription read, the run row, the item upserts, the
check result -- inline on the event loop, so an unattended check enabled by
default froze whatever tab the user was looking at for its whole duration.
Every one of those calls is a plain synchronous sqlite call inside an
``async def``, which is what this helper moves.

One helper rather than a bare ``asyncio.to_thread`` at each call site,
because the hop is not unconditionally safe:

* ``SubscriptionsDB`` keeps **thread-local** connections and builds its
  schema on the constructing thread (see ``SubscriptionsDB.
  _initialize_schema``). For a file-backed database that is exactly what
  makes the hop safe -- the worker opens its own connection to the same
  file, and one cached instance can therefore be shared by every thread.
* For a ``:memory:`` database it is the opposite: each connection is a
  *private, empty* database, so work executed on another thread would write
  into a database nobody can read and read from one nobody wrote to. Two
  live callers depend on in-memory instances (``WatchlistPreviewService``'s
  throwaway preview DB, and the in-memory service tests), so an in-memory
  database runs its work inline, on the calling thread, exactly as before.

The rest of the contract is deliberately unremarkable: one awaited hop per
call, so call ordering and exception propagation are identical to the direct
call this replaces.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_db_off_loop(
    db: Any, fn: Callable[..., T], /, *args: Any, **kwargs: Any
) -> T:
    """Await ``fn(*args, **kwargs)`` on a worker thread, unless ``db`` is in-memory.

    Args:
        db: The ``SubscriptionsDB`` whose connections the call will use. Read
            only for its ``is_memory_db`` flag; the callable is what actually
            touches it.
        fn: A synchronous callable doing the database work. It must not hold a
            ``db.transaction()`` open across the call boundary and must not
            need the event loop.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        Exception: Whatever ``fn`` raises, unchanged.
    """
    # `is True`, not truthiness: a `MagicMock` database (several handler tests
    # use one) answers every attribute with a truthy Mock, and treating those
    # as in-memory would quietly keep the very path this exists to move on the
    # event loop.
    if getattr(db, "is_memory_db", None) is True:
        return fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)
