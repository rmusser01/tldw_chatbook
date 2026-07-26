---
id: TASK-704
title: >-
  Remove the silent row limits in word bench load_grid and use the run_group_id index
status: To Do
assignee: []
created_date: '2026-07-26 14:30'
labels:
  - evals
  - word-bench
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

Two limits in `storage.load_grid`, both silent:

**Cells truncate at 1,000 per target.** `db.get_run_results(run["id"])` takes `Evals_DB`'s default `limit=1000`. A bench with more than 1,000 snippets renders a grid whose extra cells are simply absent — and this design made "absent means not yet run" a load-bearing invariant, so the truncation reads as unrun cells rather than missing data.

**Run lookup rescans up to 10,000 rows in Python.** `list_runs(limit=10_000)` plus a client-side filter on `run_group_id` decodes up to 10k rows (two JOINs, a `json.loads` each) on every grid open, and past 10k rows a group vanishes entirely. It fails loudly with `ValueError` rather than returning a partial grid, which is the safe failure mode, but it is still a wall.

`idx_eval_runs_group` was added by PR 2 for exactly this query and is currently used by nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `Evals_DB.list_runs` accepts a `run_group_id` filter that uses `idx_eval_runs_group`
- [ ] `load_grid` no longer scans-and-filters client-side
- [ ] Cell loading pages or passes an explicit limit, so a bench with more than 1,000 snippets renders every captured cell
- [ ] A test proves a >1,000-cell grid loads completely
<!-- AC:END -->
