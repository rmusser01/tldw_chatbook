---
id: TASK-1483
title: Cancel affordance for in-flight Evals bench runs
status: Done
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
- [x] #1 A visible Cancel affordance exists while a run is in flight
- [x] #2 Cancelling preserves already-captured cells and the grid renders the partial run
- [x] #3 The cancelled run group is labeled as cancelled in the rail and grid
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

- **Affordance (AC1)**: `#evals-cancel-bench-run` composes in the
  inspector pane directly after `#evals-primary-action` for every
  selection kind that composes the primary action (classic and
  skill-eval benches early-return; the skill-eval panel keeps its own
  Cancel). It is always in the DOM at a stable id and hidden
  (`display = False`) except while a run is in flight -- shown live by
  id from `_set_bench_run_running_ui`/`_set_sample_bench_running_ui`
  (a run starting does not recompose any pane), and re-derived from the
  running flags when a mid-run rail click recomposes the pane, the same
  fresh-instance seam `LibraryRail.sample_bench_running` closes.
- **Single-token dispatch (per Addendum 2)**: the handler cancels the
  first non-`None` of the three tokens -- the cross-worker guards prove
  only one can ever be live at once.
- **AC2**: cancellation is cooperative (`CancelToken.cancel()`); the
  word-bench engine stops scheduling further cells, an already-in-flight
  cell runs to completion and is kept, and the worker's completion path
  selects the partial run group so the grid renders it. Test
  (`test_cancelling_an_in_flight_run_keeps_captured_cells_and_labels_the_group`)
  pins: 2 of 3 cells persist (1 captured pre-cancel + 1 in-flight),
  selection lands on the run group.
- **AC3**: rail — the engine already stamps runs `"cancelled"` on
  cooperative cancel, so `run_groups()`'s pivot renders the `✗` glyph
  (pinned in the test). Grid — task-1511's cancelled callout ("Run
  cancelled — K of N cells attempted, M failed."). Character runs now
  stamp `"cancelled"` instead of `"completed"` when their token fired
  (previously a cancelled character run read as completed in the rail).
  Completion toasts say "Bench run cancelled." (word/character) or
  "Sample bench created; run cancelled." rather than "finished".

Modified files: `tldw_chatbook/UI/Screens/evals_screen.py`,
`Tests/UI/test_evals_screen.py` (shared commits with tasks
1510/1511/1512 over the same workers).
