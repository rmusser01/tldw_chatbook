---
id: TASK-3503
title: config.load_settings cache-miss race can return None to worker threads
status: Done
assignee: []
created_date: '2026-08-07 20:36'
labels:
  - config
dependencies:
  - TASK-3170
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Diagnosed during TASK-3170's Task 6 while chasing an intermittent test failure (1/12 runs at commit ceba0f02b, cited in that task's report): config.load_settings() sets the module-level _SETTINGS_CACHE to None under its lock on a cache miss, then rebuilds the real settings object OUTSIDE that lock. A second thread (observed concretely as a Textual worker thread calling config.load_settings() while a cache rebuild is in flight) can read the cache in the None window and receive None instead of a settings mapping or the previous cached value. This is not scoped to RAG -- any code path that calls load_settings() from a worker thread while another thread is mid-rebuild can hit it, and the failure mode (a None where a settings dict is expected) tends to surface as a confusing downstream TypeError/AttributeError far from the real cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 config.load_settings() never returns None to a concurrent caller during a cache rebuild -- either the lock is held for the full miss-then-rebuild sequence, or a concurrent reader gets the previous cached value instead of None
- [x] #2 A concurrency regression test reproduces the race (multiple threads calling load_settings() while a cache invalidation/rebuild is forced) and fails against the current code before the fix, passes after
- [x] #3 No behavior change to the normal single-threaded / cache-hit path
<!-- AC:END -->

## Implementation Notes

`load_settings` is now a thin locking wrapper over the original body (renamed
`_load_settings_uncached`), holding a module-level RLock across
miss -> rebuild -> store with a double-check inside. Chose a wrapper over
re-indenting ~1400 lines into a `try/finally`.

The lock is REENTRANT deliberately: one rebuild makes several nested
configuration reads, so a plain `Lock` would deadlock the rebuilding thread
against itself.

**Measured** (8 threads, one invalidation, counting `_load_cli_config_
bootstrap`): 32 full rebuilds before, 4 after. A single uncontended rebuild
also costs 4, so cross-thread duplication is now zero and the remaining 4 is
one rebuild's own nested reads -- a separate, single-threaded inefficiency
this task does not address and which is worth its own follow-up. The
duplication was exactly linear in thread count, so the saving scales with
concurrency.

**AC #1 could not be reproduced as written.** It asks that `load_settings()`
never hand a concurrent caller `None`. It never did: `load_settings` does not
return the cache cell during the miss window, and no module outside
`config.py` reads `_SETTINGS_CACHE` directly (verified by grep). A thread
arriving mid-rebuild always took the miss branch and got a real mapping -- it
just paid for a redundant rebuild to get it. The real defect was the
duplicated work, which is what the regression test pins. `None`-freedom is
pinned anyway so a future refactor that starts returning the raw cell fails.

Files: `tldw_chatbook/config.py`,
`Tests/test_config_settings_cache_concurrency.py`.
