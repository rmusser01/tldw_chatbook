---
id: TASK-703
title: >-
  Return preflight results from the word bench runner so PR 3 can render readiness
status: Done
assignee: []
created_date: '2026-07-26 14:30'
labels:
  - evals
  - word-bench
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

PR 2's `WordBenchRunner.run()` computes a `PreflightResult` for every target and then discards everything except `.canary`. `state`, `k_returned`, `detail`, and `checked_at` are thrown away, and `run()` returns only the run-group id.

The design spec requires a per-target readiness badge (Ready / Unavailable / Blocked), a `.ds-recovery-callout` per failure mode, and a grid header stating the effective K. None of that is reachable from `run()`'s return value or from `load_grid`. PR 3's only options today are to re-run preflight — doubling the canary calls, and possibly getting a different verdict than the run actually used — or to reach past the engine's public API.

Related: `runner.py`'s own docstring says "fail-fast on a dead target is preflight's job", but an `unreachable` target still gets a run row and N failing cells. The state is computed and never acted on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] `run()` returns the per-target `PreflightResult` set alongside the run-group id, or exposes it through an `on_preflight` callback
- [x] The verdicts are persisted into the run snapshot, so a grid reloaded later can still explain why a column is empty
- [x] `load_grid` surfaces the stored verdicts to its caller
- [x] A test asserts a reloaded grid carries the readiness state of a target that was unreachable at run time
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Commit 4153644175 'feat(evals): return preflight results from the word bench runner' (TASK-703 in body): run() returns a RunOutcome and snapshots the per-target verdicts, and the results grid renders the warned column from the stored preflight snapshot (results_grid.py:85-87).
