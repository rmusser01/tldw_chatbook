---
id: TASK-1512
title: >-
  Stale Blocked badge and unrestored primary action after failure-path run end
status: Done
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
- [x] A bench run failing (exception path) restores the full primary-action block, not just the button
- [x] A sample run failing while a bench is selected restores the primary action's true state
- [x] Tests cover both failure paths with a mid-run recompose forced first
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

Approach: the first suggested fix shape -- one shared restore.
`_reset_bench_run_running_ui` now restores the FULL primary-action
block from a fresh `_primary_action_state()`: the button
(label/disabled/tooltip), the TASK-1076 Blocked badge + callout pair
(updated in place when the settled state is legitimately blocked,
hidden when it is not), and task-1483's Cancel button (hidden). The
SAMPLE worker's finally calls the same restore in addition to its own
rail-button reset, closing AC2's "nothing restores the primary action
at all" gap (the sample worker previously reset only
`#evals-create-sample-bench`).

`Widget.remove()` is deferred/async, so the stale badge/callout are
hidden (`display = False`) rather than removed -- the next selection
change replaces the whole pane anyway, and tests assert on
`display`.

Tests: `test_a_failed_run_restores_the_full_primary_action_block`
(AC1) and `test_a_failed_sample_run_restores_the_primary_actions_
true_state` (AC2) -- both force a mid-run recompose first (a same-bench
reselection while the worker is gated), assert the badge/callout were
composed, then release the failure and assert the full settled block.
Both use a new `_GatedThenFailingCaptureClient` whose preflight blocks
on an event and then raises, giving the recompose a deterministic
in-flight window (AC3).

Modified files: `tldw_chatbook/UI/Screens/evals_screen.py`,
`Tests/UI/test_evals_screen.py`.
