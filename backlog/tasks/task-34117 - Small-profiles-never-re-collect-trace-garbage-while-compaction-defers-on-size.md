---
id: TASK-34117
title: Small profiles never re-collect trace garbage while compaction defers on size
status: To Do
assignee: []
created_date: '2026-10-03 19:12'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console trace-maintenance loop keeps a logical GC result pending until physical compaction completes or fails with a non-retryable reason, and only runs a new collection when nothing is pending. Below `trace_compaction_min_database_bytes` (64 MiB by default) compaction defers with the retryable `database_threshold` reason, so after the first pass the loop never collects again: each GC interval is an epoch read plus a compaction retry. Unreachable trace rows from later graph epochs therefore stay until the database grows past the threshold, which a small profile may not reach for a long time. Found while building the TASK-33644 GC census (#2969), whose billed first pass is the only one that runs `collect` on a small database. Code: the `pending_gc_result` handling in `console_runtime`'s legacy trace-maintenance loop and `TRACE_PHYSICAL_MAINTENANCE_RETRYABLE_REASONS`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a database below the compaction size threshold, a GC interval that follows a graph-epoch advance runs a new logical collection (a test advances the epoch twice and observes two collections)
- [ ] #2 Size and freelist threshold deferrals no longer block logical collection, and compaction still runs only after a completed collection
- [ ] #3 Transient deferrals (provider activity, busy connections, retry backoff, maintenance busy) keep their current retry behaviour
<!-- AC:END -->
