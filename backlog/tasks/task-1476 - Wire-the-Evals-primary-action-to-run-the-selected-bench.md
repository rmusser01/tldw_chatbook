---
id: TASK-1476
title: >-
  Wire the Evals primary action to run the selected bench
status: Done
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30, dev 665ef1c01). `_primary_action_state()` in `UI/Screens/evals_screen.py` returns `disabled=True` in every branch and `evals_screen.py:333` documents that no press handler exists — the docstring says wiring was deferred to "PR 3b (the results grid)", but 3b merged without it. The shipped screen's Run Bench button can never run anything; the only executor is the one-click sample bench.

The consequence is a contradiction loop verified live: a failed sample run's blocked panel says "select a bench to start a new run", and selecting the bench says "Running a bench from this workbench isn't wired up yet; that lands with the results grid in a later PR." A user who fixes their server has no way to re-run.

The design spec (2026-07-25-evals-console-rebuild-design.md, "Screen IA and layout" and "Execution") already defines the behavior: the primary action reads "Run <bench name>" and is enabled when a bench is selected; execution is row-major in a worker, and the grid doubles as the progress view. `WordBenchRunner.run()` and the sample-bench worker pattern (`_create_sample_bench_worker`) provide the execution seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Selecting a word bench whose targets resolve enables the primary action, labeled "Run <bench name>"
- [x] Pressing it runs the bench in an exclusive worker and selects the resulting run group, so the grid shows results as they fill
- [x] Progress is visible while the run is in flight, and completion/failure each produce a notification
- [x] Blocked reasons remain for datasets, run groups, missing benches, and empty selection; the "isn't wired up yet / later PR" copy no longer exists in the codebase
- [x] After a run fails on a dead server, the user can start the server and re-run the same bench from the UI (the UAT recovery loop closes)
- [x] Tests cover the enabled-state gating, a successful run via a fake capture client, and re-run after a failed run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add run_existing_bench engine helper beside create_and_run_sample_bench
2. Wire #evals-primary-action press handler + exclusive worker mirroring the sample-bench pattern
3. Enable the found-bench branch of _primary_action_state; remove the "later PR" copy
4. Harden toasts/tooltips/labels against markup; gate buttons while a run is in flight
5. Fix the inspector CSS so the enabled button paints; live-verify the recovery loop
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commits 19ad9d43d, 46d56f371, da4967a7a, 7d16b5022, f2e37e6bf. `run_existing_bench(view_model, app_config, task_id, *, client_factory/progress/cancel_token)` loads the saved BenchConfig via storage.load_bench, resolves targets via db.get_model (RuntimeError naming any unresolvable id, no run group leaked), reuses the sample bench's client factory and orphan-run cleanup. The press handler pins the bench id at press time and runs an exclusive worker (group="evals-run-bench") mirroring _create_sample_bench_worker exactly; completion notifies and selects the new run group so the grid doubles as the progress/result view. All four bench-run toasts are markup=False (a reviewer reproduced an app crash from a `[/]`-bearing dataset name reaching notify); tooltip/label interpolations escape the bench name. A whole-branch finding added in-flight gating: recomposes during a run now render the action disabled with "A bench run is already in flight." Live verification then found the enabled button PAINTED NOWHERE at 235x52 — EvalsInspector (a Vertical, default height 1fr) starved its Button sibling inside the auto pane; fixed with `#evals-inspector-bench { height: auto; }` (the _lab.tcss:328 precedent one level deeper) plus painted-geometry tests at real size, and the five scroll_visible() test workarounds were removed. Recovery loop verified live end-to-end: dead server -> 4 failed -> server up -> Run from the bench row -> K 20, 0 failed, new run group auto-selected. Trade-off: the sample-bench and bench-run workers are co-startable (task-1483 addendum); failure-path badge staleness filed as task-1512.
<!-- SECTION:NOTES:END -->
