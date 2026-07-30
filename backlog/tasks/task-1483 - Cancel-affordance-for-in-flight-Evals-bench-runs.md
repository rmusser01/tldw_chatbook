---
id: TASK-1483
title: >-
  Cancel affordance for in-flight Evals bench runs
status: To Do
assignee: []
created_date: '2026-07-30 10:00'
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A visible Cancel affordance exists while a run is in flight
- [ ] Cancelling preserves already-captured cells and the grid renders the partial run
- [ ] The cancelled run group is labeled as cancelled in the rail and grid
<!-- AC:END -->
