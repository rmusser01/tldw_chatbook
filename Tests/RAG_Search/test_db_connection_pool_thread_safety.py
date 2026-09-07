"""Concurrent first keyword searches must not construct duplicate pools.

(Qodo PR #1428 finding 3) `get_connection_pool`'s global `_connection_pools`
dict was guarded only by an unlocked check-then-set: `if db_path not in
_connection_pools: ... _connection_pools[db_path] = MediaDatabase(...)`.
`MediaDatabase.__init__` runs schema init immediately (real I/O), so two
threads racing the first keyword search for the same db path could both
pass the `not in` check before either stored its instance -- each
constructs its own `MediaDatabase`, and the loser's instance (and its open
sqlite connection) is simply overwritten in the dict and leaked.
"""
import threading
import time

import pytest

from tldw_chatbook.RAG_Search.simplified import db_connection_pool as pool_mod

pytestmark = pytest.mark.unit


class _FakeMediaDatabase:
    """Stand-in for `MediaDatabase` that counts constructions and, like the
    real class, does non-trivial work in `__init__` (schema init) -- the
    `sleep` widens the race window so concurrent callers actually overlap
    inside the critical section under test instead of serializing by
    accident."""

    construction_count = 0
    _count_lock = threading.Lock()

    def __init__(self, db_path, client_id):
        with _FakeMediaDatabase._count_lock:
            _FakeMediaDatabase.construction_count += 1
        self.db_path = db_path
        self.client_id = client_id
        time.sleep(0.02)

    def close_connection(self):
        pass


@pytest.fixture(autouse=True)
def _isolated_pool_state(monkeypatch):
    """Give every test a clean pool dict and the fake constructor, and
    close whatever it opened afterward."""
    monkeypatch.setattr(pool_mod, "_connection_pools", {})
    monkeypatch.setattr(pool_mod, "MediaDatabase", _FakeMediaDatabase)
    _FakeMediaDatabase.construction_count = 0
    yield
    pool_mod.close_all_pools()


def test_concurrent_first_callers_construct_exactly_one_media_database():
    db_path = "/fake/shared/media.db"
    thread_count = 16
    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(thread_count)

    def _worker():
        barrier.wait(timeout=5)  # maximize overlap: all threads race together
        pool = pool_mod.get_connection_pool(db_path)
        with results_lock:
            results.append(pool)

    threads = [threading.Thread(target=_worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "worker thread did not complete in time"

    assert _FakeMediaDatabase.construction_count == 1, (
        "exactly one MediaDatabase must be constructed for concurrent "
        f"first callers of one db path, got {_FakeMediaDatabase.construction_count}"
    )
    assert len(results) == thread_count
    assert all(pool is results[0] for pool in results), (
        "every concurrent caller must receive the SAME pooled instance"
    )


def test_pool_size_parameter_still_accepted_and_unused():
    """`pool_size` stays API-compatible through the locking change."""
    db_path = "/fake/other/media.db"
    pool = pool_mod.get_connection_pool(db_path, pool_size=9)
    assert pool is pool_mod.get_connection_pool(db_path, pool_size=1)
    assert _FakeMediaDatabase.construction_count == 1


def test_close_all_pools_clears_state_under_the_same_lock():
    pool_mod.get_connection_pool("/fake/closeme/media.db")
    assert pool_mod._connection_pools
    pool_mod.close_all_pools()
    assert pool_mod._connection_pools == {}
