---
id: TASK-1512
title: >-
  Stale Blocked badge and unrestored primary action after failure-path run end
status: To Do
assignee: []
created_date: '2026-07-30 14:00'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Scoped re-review finding on the 2026-07-30 batch's in-flight gating fix. When a mid-run recompose has composed the "A bench run is already in flight." Blocked badge + callout, a run ending on the FAILURE path (no select() recompose) leaves them stale — `_reset_bench_run_running_ui` restores only the button, yielding an enabled button under a "Blocked" badge; and after a failed SAMPLE run nothing restores the primary-action button at all (its finally resets only `#evals-create-sample-bench`), leaving a false "in flight" reason until the next rail click. Both states self-heal on any rail interaction and the error toast still fires. Fix shape: both workers' finally paths trigger one shared restore, or a cheap refresh(recompose=True) on the failure path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A bench run failing (exception path) restores the full primary-action block, not just the button
- [ ] A sample run failing while a bench is selected restores the primary action's true state
- [ ] Tests cover both failure paths with a mid-run recompose forced first
<!-- AC:END -->
