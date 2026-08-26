---
id: TASK-22502
title: >-
  Harden the live-continuation retry against a cancel-during-CAS ledger/DB divergence
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - console
  - correctness
priority: medium
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Found by TASK-22205's adversarial reviewer (probe P2). If a cancel lands DURING the
pre-dispatch checkpoint CAS hop, the surviving thread commits `dispatch_started` while the
effect ledger records the transition as abandoned; `retry_dispatch_recovery` then fails
"Accepted turn is retained for recovery" on EVERY retry — an in-process wedge until restart
or session reopen (no data loss; the restart path's `reconcile_for_session` classifies it
correctly).

Not reachable by any production path today: only `close_session` (which retires the
continuation, and reopen recovers) and shutdown cancel that task; the Stop button cancels
only the stream task. Filed so it does not become reachable silently.

## Acceptance Criteria

- [ ] The live-continuation retry accepts an already-DISPATCH_STARTED checkpoint (or reconciles the ledger against the DB before re-claiming the effect)
- [ ] A test reproduces the cancel-during-CAS interleave and proves in-process recovery without a restart
- [ ] Coverage gaps noted by the same review are closed: a unit test for the `:memory:` offload gate, and the ordering probe's race-satisfiable `checkpoint_states` clause is tightened or documented as covered by the first-send atomicity tests
