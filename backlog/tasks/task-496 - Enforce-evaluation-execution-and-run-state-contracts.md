---
id: TASK-496
title: Enforce evaluation execution and run-state contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 15:45'
labels:
  - evals
  - reliability
  - workers
dependencies: []
references:
  - backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md
documentation:
  - Docs/superpowers/specs/2026-07-24-evaluation-execution-contracts-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make evaluation execution honor the synchronous provider boundary, configured request bounds, deterministic sample accounting, and truthful terminal run states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Synchronous production provider calls execute without blocking the event loop or being incorrectly awaited
- [ ] #2 Configured request timeouts and retry policy govern each provider attempt
- [ ] #3 Configured maximum concurrency bounds sample execution while returned results remain in input order
- [ ] #4 Each durably stored sample reports progress exactly once through the documented (completed, total, result) callback; synchronous and asynchronous callbacks are supported and same-loop UI updates do not use thread-only marshalling
- [ ] #5 Runs with one or more sample errors retain their results and finish failed with an actionable summary
- [ ] #6 The public run-ID cancellation API targets and drains the registered owning task, while public and direct cancellation persist cancelled state, unregister cleanly, and preserve CancelledError control flow
- [ ] #7 Regression tests reproduce the production dispatcher contract, timeout, concurrency, callback, partial-failure, public-cancellation, and cleanup behavior without unawaited coroutines or orphan sample tasks
- [ ] #8 Invalid concurrency, timeout, retry-count, and retry-delay bounds fail before evaluation begins
- [ ] #9 Callers receive the durable run ID before provider work through a documented sync-or-async run-start callback
- [ ] #10 Asynchronous orchestrator shutdown drains active evaluations before closing their database, while synchronous close fails without closing when a run is active
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md
Reason: TASK-496 implements the corrected callback, public-cancellation, provider, concurrency, and terminal-state contract already accepted in ADR-024.

1. Pin invalid bounds and the real synchronous dispatcher with red tests, then adapt provider attempts off-loop with configured timeout.
2. Normalize one timeout/retry policy across basic and specialized runners without double retry.
3. Bound sample tasks, preserve input order, await sync-or-async callbacks, and cancel/drain all children on failure or cancellation.
4. Store results before progress, make partial failures durable failed runs, and correct the stale integration patch target.
5. Register owning tasks, expose the real run ID before provider work, and implement public cancel-and-drain semantics.
6. Reconcile the UI callback and shutdown APIs without adding an application-state owner.
7. Run warning-strict focused, full Evals, integration, lint, compile, and diff verification before task reconciliation.

Detailed plan: Docs/superpowers/plans/2026-07-24-evaluation-execution-contracts.md
<!-- SECTION:PLAN:END -->
