# Tests/Agents/test_tool_catalog_concurrency.py
"""Registry cache lock under concurrent/interleaved lookups (fleet PR 2a).

The registry's own comment (tool_catalog.py __init__) documents that
`_owner_cache`/`_name_to_id_cache`/`_source_cache` used to be rebuilt
without a lock, so two overlapping lookups -- in particular
`invoke_by_name()`'s old resolve_name() + _owner_and_id() pair -- could
observe different generations of the catalog. With N fleet children on
their own threads sharing the bridge's long-lived registry for a whole
session, that stopped being exotic.

The two tests below are DETERMINISTIC (no threads, no sleeps, no
Barrier/timing dependence): they monkeypatch `_ensure_catalog_cache` to
count calls / inject a `reset_catalog_cache()` right in the historical race
window, so they are red pre-fix and green post-fix on every run, not just
some fraction of runs under scheduler luck. A stochastic threaded test
follows as a deadlock canary only -- it does NOT reliably reproduce the
race (confirmed: 40/40 clean runs against the pre-fix code, including with
`sys.setswitchinterval(1e-6)`), so it must not be read as a race regression
guard.
"""

import threading

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry


def test_invoke_by_name_takes_exactly_one_catalog_snapshot():
    """invoke_by_name() must resolve name -> id -> provider from ONE
    locked snapshot, not from two independent _ensure_catalog_cache()
    calls (the old resolve_name() then _owner_and_id() pair). Pre-fix this
    counts 2; post-fix, 1."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry._ensure_catalog_cache()  # warm the cache first

    # Resolve the name BEFORE the counter is installed. `list_catalog()`
    # routes through `_ensure_catalog_cache()` itself, so taking the name
    # inside the counting window charges the test's own setup to the
    # subject and the assertion reads 2 no matter how `invoke_by_name`
    # behaves -- it stopped measuring anything the moment `list_catalog`
    # was refactored onto the shared snapshot helper.
    name = registry.list_catalog()[0].name

    real_ensure = registry._ensure_catalog_cache
    snapshots = []

    def counting():
        snapshots.append(1)
        return real_ensure()

    registry._ensure_catalog_cache = counting

    registry.invoke_by_name(name, {})

    assert len(snapshots) == 1  # pre-fix: 2


def test_reset_landing_in_the_window_cannot_break_a_lookup():
    """Simulate a `reset_catalog_cache()` from another thread landing in
    the exact window between a snapshot being taken and the caller using
    it. Pre-fix, resolve_name()'s own internal snapshot read happened
    AFTER this window (so a reset here could zero `_owner_cache` out from
    under `_owner_and_id()`, raising AttributeError on `.get()` against
    `None`); post-fix, the snapshot is the return value itself, immune to
    a reset that happens after it was captured."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    name = registry.list_catalog()[0].name
    tool_id = registry.resolve_name(name)

    real_ensure = registry._ensure_catalog_cache

    def ensure_then_reset():
        snapshot = real_ensure()
        registry.reset_catalog_cache()  # racing thread lands HERE
        return snapshot

    registry._ensure_catalog_cache = ensure_then_reset

    assert registry.resolve_name(name) == tool_id
    assert registry._owner_and_id(tool_id) is not None  # pre-fix: AttributeError
    assert registry._source_for(tool_id) == "builtin"
    assert registry.load_schema(tool_id).name == name


def test_concurrent_lookups_do_not_deadlock():
    """Deadlock canary, NOT a race-regression guard (see module docstring):
    hammer reset_catalog_cache()/resolve_name() from 8 threads and confirm
    the run completes (no deadlock) and every resolution succeeds. This
    passed 40/40 runs against the pre-fix code too, so a green result here
    proves absence of deadlock, not presence of the lock fix."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    known = [e.name for e in registry.list_catalog()]
    assert known, "catalog must be non-empty for this test to mean anything"

    errors = []
    barrier = threading.Barrier(8)

    def hammer():
        barrier.wait()
        for _ in range(200):
            registry.reset_catalog_cache()
            for name in known:
                tool_id = registry.resolve_name(name)
                if tool_id is None:
                    errors.append(f"resolve_name({name}) -> None")
                    return

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "deadlock: a thread never finished"
    assert errors == []
