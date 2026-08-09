---
id: TASK-3770
title: Fetch and robots caches share the unsynchronized eviction race
status: Done
assignee:
  - '@claude'
created_date: '2026-08-08 15:55'
updated_date: '2026-08-09 02:18'
labels:
  - web-tools
  - tech-debt
  - concurrency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-2832's review chain (opus reviewer + Qodo on PR #1444) established that module-level web-tool caches are mutated from concurrent tool-call worker threads, and the eviction scan (min() ITERATES the dict) can race a concurrent put/pop into "RuntimeError: dictionary changed size during iteration" — an intermittent tool failure. The NEW _search_cache got a lock in #1444; the two PRE-EXISTING caches in Tools/web_tool_impls.py — _fetch_cache (_cache_put + web_fetch's read/pop) and _robots_cache (_robots_cache_put + _robots_allows' read) — carry the identical race and were left as recorded debt (out of #1444's scope). Same fix shape: one lock per cache (or one shared) around the short cache ops only, never held across network calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 _fetch_cache and _robots_cache reads/puts/evictions are synchronized so concurrent tool threads cannot hit the iteration race
- [x] #2 No lock is held across any network or extraction call
- [x] #3 Existing cache tests stay green (behavior unchanged single-threaded)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added one threading.Lock per pre-existing cache in Tools/web_tool_impls.py, mirroring the
_search_cache_lock shape shipped in #1444: _fetch_cache_lock (guards _cache_put's body and
web_fetch's cache-hit dict .get()/.pop() only) and _robots_cache_lock (guards
_robots_cache_put's body and _robots_allows' cache read/expiry check only).

Lock scope carved precisely per the design doc (spec review, Important 1): in web_fetch's
cache-hit branch, only the dict .get(), expiry comparison, and .pop() run under
_fetch_cache_lock -- _validate_hop (live DNS), the robots re-check (_robots_allows, which can
trigger a full robots.txt fetch + client construction), and the `return` all run AFTER
release. In _robots_allows, only the cache read + expiry check run under _robots_cache_lock;
can_fetch() and a cache-miss's _fetch_robots_parser fetch run outside it -- the accepted
cross-call stampede (at most one duplicate robots fetch) is unchanged from before this task.
Updated _robots_allows' docstring from the now-false "no locks" to "cache bookkeeping locked,
fetch not" (Minor 6). _reset_state_for_tests()'s direct .clear() calls stay unguarded
(test-only, never concurrent with worker threads; Minor 7, deliberate).

Verification: added a barrier-based lock-usage test (parametrized across both locks) in
Tests/Tools/test_web_tool_impls.py -- a worker thread is parked inside the critical section
via a fake `time.monotonic()` that blocks on a threading.Event, and the test asserts
lock.acquire(blocking=False) fails while the worker is inside, then succeeds once released.
Manually confirmed the test discriminates: running the same barrier against an UNLOCKED
version of _cache_put let the non-blocking acquire succeed (the assertion would go red).
A deterministic race-reproduction test remains out of scope (flaky by nature, recorded
non-goal in the design doc).

Files: tldw_chatbook/Tools/web_tool_impls.py, Tests/Tools/test_web_tool_impls.py.
No deviations from the design doc.
<!-- SECTION:NOTES:END -->
