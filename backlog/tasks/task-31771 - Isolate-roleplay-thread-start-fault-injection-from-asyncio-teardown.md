---
id: TASK-31771
title: Isolate roleplay thread-start fault injection from asyncio teardown
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:54'
updated_date: '2026-09-05 19:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve thread startup failure coverage without replacing the standard library Thread class used by the test runner during cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Constructor and start failures still release the fork transition without changing canonical state
- [x] #2 The standard library Thread identity remains unchanged during fault injection and pytest teardown emits no runner warning
- [x] #3 The full Console session settings file passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only isolation of an existing dependency fault injection; no production owner or thread contract changes.
1. Reproduce the asyncio teardown warning and trace roleplay serialization through asyncio.to_thread to concurrent.futures.thread.
2. Patch only the executor module threading namespace (Thread fault, real Lock/Semaphore) and assert stdlib Thread identity remains unchanged.
3. Run both fault variants with RuntimeWarning escalated and the complete settings file; run static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the global threading.Thread mutation with a namespace local to concurrent.futures.thread, the real asyncio.to_thread executor boundary. The namespace retains real Lock and Semaphore; assertions pin unchanged stdlib Thread identity before and after both constructor/start failure paths. Original RED with RuntimeWarning escalated: 2 passed plus 2 asyncio teardown errors. Focused GREEN: 2 passed; final complete session-settings file, including the independently tracked Subagents topology correction, passed 416 tests in 282.54s under -W error::RuntimeWarning with no runner warnings. Ruff lint and changed-region format checks passed; root reviewed the scoped diff. No production changes or ADR required; self-review complete.
<!-- SECTION:NOTES:END -->
