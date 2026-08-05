---
id: TASK-706
title: >-
  Word bench run rows never leave status='pending'
status: Done
assignee: []
created_date: '2026-07-26 14:30'
labels:
  - evals
  - word-bench
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

Nothing in `WordBenchRunner` calls `update_run_status`. `completed_samples` increments via `store_result`, but `status` and `end_time` are never set.

Consequences: `list_runs(status='completed')` can never return a word bench run, and neither `load_grid` nor any future UI can distinguish "cancelled, partial" from "still filling" from "finished". The design took real care to keep *failed* and *not-yet-run* distinguishable at the cell level; the same distinction is missing at run level, where the spec explicitly cares — "a cancelled run is a real, if incomplete, measurement and is never discarded".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Run rows move to `running` at launch, `completed` at the end, and `cancelled` on the cancel path
- [x] `end_time` is set on all terminal transitions
- [x] A test asserts a cancelled run group's rows read `cancelled`, not `pending`
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #924. Runs now transition `running` at launch, `completed` at the end, and `cancelled` on the cancel path, with `end_time` set on terminal transitions.

One gap remains and is deliberately not closed here: an uncaught exception mid-run leaves the row at `running` — there is no `failed` transition. Worth a follow-up.
<!-- SECTION:NOTES:END -->
