---
id: TASK-548
title: Close unscheduled gateway cleanup coroutines
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 20:42'
updated_date: '2026-07-24 20:42'
labels:
  - console
  - asyncio
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent cross-loop HTTP-client swaps from abandoning an aclose coroutine when the target event loop has already closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed run_coroutine_threadsafe submission closes the unscheduled coroutine object
- [ ] #2 Loop-swap cleanup remains best-effort and never raises into the caller
- [ ] #3 Gateway lifecycle and concurrency tests pass without unawaited-coroutine warnings
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a deterministic regression for scheduler rejection after aclose has produced its coroutine.
2. Close the unscheduled coroutine in the RuntimeError path without changing the best-effort public lifecycle contract.
3. Run focused lifecycle/concurrency tests with warnings promoted where practical, then resume the remaining suite.

ADR required: no
ADR path: N/A
Reason: This is a local resource-cleanup bug fix inside the existing Console gateway lifecycle and cross-loop ownership contract; it adds no interface or architectural boundary.
<!-- SECTION:PLAN:END -->
