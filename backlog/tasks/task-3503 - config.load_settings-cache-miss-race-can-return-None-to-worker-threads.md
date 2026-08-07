---
id: TASK-3503
title: config.load_settings cache-miss race can return None to worker threads
status: To Do
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
- [ ] #1 config.load_settings() never returns None to a concurrent caller during a cache rebuild -- either the lock is held for the full miss-then-rebuild sequence, or a concurrent reader gets the previous cached value instead of None
- [ ] #2 A concurrency regression test reproduces the race (multiple threads calling load_settings() while a cache invalidation/rebuild is forced) and fails against the current code before the fix, passes after
- [ ] #3 No behavior change to the normal single-threaded / cache-hit path
<!-- AC:END -->
