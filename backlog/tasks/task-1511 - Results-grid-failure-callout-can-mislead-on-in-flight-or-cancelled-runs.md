---
id: TASK-1511
title: Results-grid failure callout can mislead on in-flight or cancelled runs
status: Done
assignee: []
created_date: '2026-07-30 14:00'
updated_date: '2026-07-31 01:46'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review finding (F5) of the 2026-07-30 UAT fix batch. `_failure_summary` totals cells ATTEMPTED, not snippets×targets from the snapshot. A user navigating to an in-flight run whose first K cells all failed sees "All K cells failed — … then run the bench again" while the run is still going; a hard-cancelled run keeps that callout permanently with N below the real grid size. Self-corrects at completion via the unconditional select() recompose. Fix shape: gate the "All …/run again" sentence on group status, or total from the snapshot. Note (TASK-1480 amendment): the library rail now has its own ✓✗ all-cells-failed glyph for a COMPLETED group -- this callout's copy/gating for in-flight and cancelled runs should stay consistent with that scheme rather than introduce a second, divergent notion of "all failed".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An in-flight run group never renders the "All N cells failed … run the bench again" sentence
- [x] #2 A cancelled run's callout states cancellation and the true attempted/total proportions
- [x] #3 Completed-run callout behavior is unchanged
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

Approach: gate on group status (the task's own first suggested fix
shape), not snapshot totals. `load_grid` now also returns
`run_statuses` (the per-target runs' own `eval_runs.status`, in the
order `load_grid` already fetched them -- no extra DB read), and
`ResultsGrid.compose` rolls them up with
`_rolled_up_group_status` -- the EXACT precedence
`EvalsViewModel.run_groups()`'s pivot uses, re-implemented on the
grid's own data because a widget must not pivot every run in the
database (the same cost ruling `_register_grid_shortcuts` already
makes for the same classification). The two surfaces must stay in
lockstep -- that is the TASK-1480-amendment consistency requirement.

- **AC1**: an in-flight group with failures renders "Run in flight —
  F of A attempted cells failed so far — <reason>." -- never the
  "All N cells failed ... run the bench again" sentence (which was
  reachable because `_failure_summary` counts ATTEMPTED cells only).
- **AC2**: a cancelled group renders "Run cancelled — A of T cells
  attempted, F failed." against the snapshot's TRUE grid size
  (snippets x targets) -- even with zero failures, because a partial
  grid needs explaining. "Cancelled" covers run-level "failed" folded
  the same way the rail's ✗ glyph folds it.
- **AC3**: the completed branch keeps its TASK-1477 copy byte for byte;
  the existing exact-string test still passes, and a new companion
  pins it with statuses explicitly stamped "completed".

Tests: 4 new cases in `Tests/UI/test_evals_results_grid.py`
(in-flight, cancelled-with-failure, cancelled-clean,
completed-all-failed).

Modified files: `tldw_chatbook/Evals/word_bench/storage.py`,
`tldw_chatbook/UI/Evals/results_grid.py`,
`Tests/UI/test_evals_results_grid.py`.
