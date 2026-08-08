---
id: TASK-3770
title: Fetch and robots caches share the unsynchronized eviction race
status: To Do
assignee: []
created_date: '2026-08-08 15:55'
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
- [ ] #1 _fetch_cache and _robots_cache reads/puts/evictions are synchronized so concurrent tool threads cannot hit the iteration race
- [ ] #2 No lock is held across any network or extraction call
- [ ] #3 Existing cache tests stay green (behavior unchanged single-threaded)
<!-- AC:END -->
