---
id: TASK-1483
title: Cancel affordance for in-flight Evals bench runs
status: To Do
assignee: []
created_date: '2026-07-30 10:00'
updated_date: '2026-07-31 02:40'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from live UAT (2026-07-30). There is no way to cancel a running bench from the UI. The engine already has cooperative cancellation (`CancelToken`, checked per row/cell; cancelled cells persist and the grid renders partial per the spec's "Execution" section), and the sample-bench worker already holds a token — nothing exposes it. Matters more once task-1476 wires arbitrary benches and task-1482 allows larger grids.

Addendum (2026-07-30 whole-branch review, N6): the sample-bench worker and the bench-run worker were co-startable (independent guard flags, different worker groups) — data-safe but producing interleaved toasts and last-wins completion selects.

Addendum 2 (2026-07-30, PR #1113 review): mutual exclusion landed in that PR — `_on_sample_bench_requested` and `_on_primary_action_pressed` now each cross-check the other's running flag (`fix(evals): make sample and bench runs mutually exclusive`), so a press of one while the other is genuinely in flight is a no-op instead of starting a second, overlapping worker. The two workers can therefore never run concurrently; this task's cancel affordance only ever needs to cover ONE in-flight run at a time (whichever worker is currently running), not two simultaneous ones.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A visible Cancel affordance exists while a run is in flight
- [ ] #2 Cancelling preserves already-captured cells and the grid renders the partial run
- [ ] #3 The cancelled run group is labeled as cancelled in the rail and grid
<!-- AC:END -->
