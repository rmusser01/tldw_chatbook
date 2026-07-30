---
id: TASK-1476
title: >-
  Wire the Evals primary action to run the selected bench
status: To Do
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
- [ ] Selecting a word bench whose targets resolve enables the primary action, labeled "Run <bench name>"
- [ ] Pressing it runs the bench in an exclusive worker and selects the resulting run group, so the grid shows results as they fill
- [ ] Progress is visible while the run is in flight, and completion/failure each produce a notification
- [ ] Blocked reasons remain for datasets, run groups, missing benches, and empty selection; the "isn't wired up yet / later PR" copy no longer exists in the codebase
- [ ] After a run fails on a dead server, the user can start the server and re-run the same bench from the UI (the UAT recovery loop closes)
- [ ] Tests cover the enabled-state gating, a successful run via a fake capture client, and re-run after a failed run
<!-- AC:END -->
