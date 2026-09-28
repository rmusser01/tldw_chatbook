---
id: TASK-33405
title: App exit closes SSH sessions in parallel within a bounded time
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 20:30'
updated_date: '2026-09-28 16:46'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
App exit closes every live SSH session one after another, and each close can take seconds against a wedged host, so quitting with several remote bindings can stall for N times that. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Closing all sessions at app exit takes about as long as the slowest single close, not their sum
- [x] #2 App exit waits a bounded time for session closes and never hangs on a wedged host
<!-- AC:END -->

## Implementation Notes

Part of Task 3 work: `close_all()` uses new `_close_workers()` helper spawning daemon threads per worker (max 5 s join). Reap path uses fire-and-forget close. Timer-based test confirms parallel close is max ~2.4 s vs. serial ~3.0 s. Changes in `remote_session_registry.py` with 23 registry tests passing.
