---
id: TASK-1511
title: Results-grid failure callout can mislead on in-flight or cancelled runs
status: To Do
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
- [ ] #1 An in-flight run group never renders the "All N cells failed … run the bench again" sentence
- [ ] #2 A cancelled run's callout states cancellation and the true attempted/total proportions
- [ ] #3 Completed-run callout behavior is unchanged
<!-- AC:END -->
