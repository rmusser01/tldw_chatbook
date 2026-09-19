"""A TTL prune must return the memory it frees, or the cache dies.

TASK-32811.6. `_prune_expired_async` and `prune_expired` deleted expired
entries from `self._cache` without recomputing `_current_memory_bytes`,
while the sync twin `_prune_expired_sync` called `_update_memory_sync`. The
counter therefore only ever grew: it ratcheted to the cap and every `put`
was rejected for lack of room, so the cache stopped accepting writes
permanently and every later search was a cold miss. The review reproduced
it dead at prune cycle 42 with zero entries cached and the counter pinned
just under a 1,024 KB cap.
"""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache


def _put(cache: SimpleRAGCache, query: str) -> None:
    cache.put(
        query=query,
        search_type="semantic",
        top_k=5,
        results=[{"id": query, "text": "x" * 2000}],
        context="ctx",
    )


def _make_cache() -> SimpleRAGCache:
    # Tiny TTL so a short sleep expires everything; a small memory cap so a
    # leak reaches it in a few cycles rather than thousands.
    return SimpleRAGCache(
        max_size=1000,
        ttl_seconds=0.05,
        max_memory_mb=1.0,
    )


def test_a_sync_prune_returns_the_memory_it_frees():
    cache = _make_cache()
    for i in range(20):
        _put(cache, f"q{i}")
    assert cache._current_memory_bytes > 0
    import time

    time.sleep(0.1)
    cache.prune_expired()
    assert len(cache._cache) == 0
    assert cache._current_memory_bytes == 0, (
        "prune deleted the entries but left their memory on the counter"
    )


@pytest.mark.asyncio
async def test_an_async_prune_returns_the_memory_it_frees():
    cache = _make_cache()
    for i in range(20):
        _put(cache, f"q{i}")
    await asyncio.sleep(0.1)
    await cache._prune_expired_async()
    assert len(cache._cache) == 0
    assert cache._current_memory_bytes == 0


def test_the_cache_still_accepts_writes_after_many_prune_cycles():
    """The user-visible consequence: the cache does not die."""
    import time

    cache = _make_cache()
    for cycle in range(60):
        for i in range(30):
            _put(cache, f"c{cycle}-q{i}")
        time.sleep(0.06)
        cache.prune_expired()

    # After 60 cycles the counter must reflect only what is currently held,
    # not the sum of everything ever pruned.
    assert cache._current_memory_bytes < cache.max_memory_bytes, (
        "the memory counter ratcheted past the cap; the cache is now dead"
    )
    # And a fresh put still lands.
    _put(cache, "final")
    assert any(k for k in cache._cache), "the cache stopped accepting writes"
