---
id: TASK-31721
title: Wait for the mounted Models pane before clock lifecycle assertions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:49'
updated_date: '2026-09-05 18:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the machine-memory lifecycle test precondition so remote selection happens after its actual pane exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unchanged injected-clock, failed-refresh and real-recompose assertions execute against the mounted Remote view and pass.
- [x] #2 The affected Models adoption test file passes with existing wait bounds and no runtime behavior changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the unchanged Remote mount failure and trace selection readiness: watch_active_view ran with is_mounted=False and zero remote pane targets, so no population was scheduled. 2. Require the real Models window to be mounted and own its remote pane before changing active_view; keep current wait budget and all clock/failed-refresh/recompose assertions. 3. Run focused regression and full Models adoption file, scoped static checks and self-review. ADR required: no. ADR path: N/A. Reason: test-only mount precondition correction, not a runtime behavior or lazy-view boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed only clock-test readiness: wait for the LLMManagementWindow to be mounted and its remote target composed before assigning active_view. The prior trace dispatched watch_active_view while is_mounted=False with no target, so it could not populate RemoteView. Clock/timestamp/failure/recompose assertions remain exact. Full142Models tests plus29Buddy tests passed308.43s after separately fixing the discovered empty-stack shutdown bug in31669. Full-file Ruff, changed-range formatting, diff whitespace and self-review pass. ADR not required for fixture readiness.
<!-- SECTION:NOTES:END -->
