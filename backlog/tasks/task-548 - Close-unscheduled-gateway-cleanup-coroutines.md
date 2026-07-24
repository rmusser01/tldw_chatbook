---
id: TASK-548
title: Close unscheduled gateway cleanup coroutines
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 20:42'
updated_date: '2026-07-24 20:47'
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
- [x] #1 A failed run_coroutine_threadsafe submission closes the unscheduled coroutine object
- [x] #2 Loop-swap cleanup remains best-effort and never raises into the caller
- [x] #3 Gateway lifecycle and concurrency tests pass without unawaited-coroutine warnings
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed the coroutine object produced by AsyncClient.aclose when run_coroutine_threadsafe rejects submission to a stopped or closed loop, preserving the gateway’s best-effort no-raise cleanup contract. Added a deterministic scheduler-rejection regression. The concurrency test now drains cleanup jobs scheduled on its private event loops before closing them, eliminating teardown-only pending-task and unawaited-coroutine noise without weakening the locking assertions. Verification: the targeted rejection/concurrency pair passed with RuntimeWarning promoted to error; the entire loopback-dependent gateway file passed 68 tests with RuntimeWarning promoted to error; the full non-gateway Chat suite passed 982 tests with 69 expected skips; Ruff check/format, py_compile, and git diff --check passed. ADR required: no. ADR path: N/A. Reason: local resource cleanup inside the existing cross-loop client ownership contract.
<!-- SECTION:NOTES:END -->
