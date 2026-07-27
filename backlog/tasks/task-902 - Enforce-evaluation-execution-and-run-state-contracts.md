---
id: TASK-902
title: Enforce evaluation execution and run-state contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 16:28'
labels:
  - evals
  - reliability
  - workers
dependencies: []
references:
  - backlog/decisions/031-bounded-evaluation-and-tool-worker-execution.md
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
- [x] #1 Synchronous production provider calls execute without blocking the event loop or being incorrectly awaited
- [x] #2 Configured request timeouts and retry policy govern each provider attempt
- [x] #3 Configured maximum concurrency bounds sample execution while returned results remain in input order
- [x] #4 Each durably stored sample reports progress exactly once through the documented (completed, total, result) callback; synchronous and asynchronous callbacks are supported
- [x] #5 Runs with one or more sample errors retain their results and finish failed with an actionable summary
- [x] #6 The public run-ID cancellation API targets and drains the registered owning task, while public and direct cancellation persist cancelled state, unregister cleanly, and preserve CancelledError control flow
- [x] #7 Regression tests reproduce the production dispatcher contract, timeout, concurrency, callback, partial-failure, public-cancellation, and cleanup behavior without unawaited coroutines or orphan sample tasks
- [x] #8 Invalid concurrency, timeout, retry-count, and retry-delay bounds fail before evaluation begins
- [x] #9 Callers receive the durable run ID before provider work through a documented sync-or-async run-start callback
- [x] #10 Asynchronous orchestrator shutdown drains active evaluations before closing their database, while synchronous close fails without closing when a run is active
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/031-bounded-evaluation-and-tool-worker-execution.md
Reason: TASK-902 implements the corrected callback, public-cancellation, provider, concurrency, and terminal-state contract already accepted in ADR-031.

1. Pin invalid bounds and the real synchronous dispatcher with red tests, then adapt provider attempts off-loop with configured timeout.
2. Normalize one timeout/retry policy across basic and specialized runners without double retry.
3. Bound sample tasks, preserve input order, await sync-or-async callbacks, and cancel/drain all children on failure or cancellation.
4. Store results before progress, make partial failures durable failed runs, and correct the stale integration patch target.
5. Register owning tasks, expose the real run ID before provider work, and implement public cancel-and-drain semantics.
6. Reconcile live callback consumers and shutdown APIs without restoring the retired gen-2 eval UI or adding an application-state owner.
7. Run warning-strict focused, full Evals, integration, lint, compile, and diff verification before task reconciliation.

Detailed plan: Docs/superpowers/plans/2026-07-24-evaluation-execution-contracts.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Adapted the synchronous production dispatcher with `asyncio.to_thread()` and
  per-attempt `wait_for()`, normalized and validated execution bounds, and
  applied one configured retry policy to basic and specialized runners.
- Added bounded indexed sample tasks, settlement-order sync/async callbacks,
  dataset-order results, and cancel-and-drain cleanup for callbacks, failures,
  and direct cancellation.
- Made durable storage precede progress, made retained sample errors finish
  failed with count summaries, registered real owner tasks, exposed the durable
  run ID before provider work, and implemented awaited public cancellation plus
  guarded synchronous/asynchronous shutdown.
- Reconciled the live A/B callback adapter, documented the public contracts,
  corrected stale patch targets, and canonicalized macOS temporary test paths.
- Current-dev reconciliation confirmed the gen-2 eval event/UI cluster was
  intentionally retired later; its stale test and owner-registry entries were
  removed rather than recreating a surrogate application path.
- Reused ADR-031; no new ADR was required.
- Verification: warning-strict focused gate `84 passed, 6 skipped`; full Evals
  suite `321 passed, 13 skipped`; Ruff, changed-source compileall, and
  `git diff --check` passed. The three suite warnings are an existing dependency
  version warning and two existing pytest collection warnings.
<!-- SECTION:NOTES:END -->
