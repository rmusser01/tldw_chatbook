---
id: TASK-1450
title: >-
  Test-suite measurement instrumentation: --durations in addopts + junit outcome-diff script
status: In Progress
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - performance
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full `pytest` runs take 1+ hour and nobody has per-test timing data — no `--durations` flag is configured anywhere, so every optimization argument is an estimate. The 2026-07-30 test-suite audit (`backlog/docs/test-suite-audit-2026-07-30.md`) launches a remediation program; every subsequent task must prove speedup and prove no coverage loss against a common baseline. This task adds the measurement plumbing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] `--durations=25 --durations-min=1.0` present in `addopts` so every run reports its slowest tests
- [ ] A committed script diffs two junit XML files into per-nodeid outcome deltas (new failures, disappeared tests, recovered tests, outcome flips) with a non-zero exit on regressions
- [ ] A serial full-suite baseline artifact (junit XML + wall time + pass/fail/skip counts at a recorded SHA) exists and is referenced from the audit doc

## Implementation Plan

1. Add duration flags to `[tool.pytest.ini_options]` addopts in pyproject.toml
2. Add `Tests/junit_outcome_diff.py` (stdlib-only argparse script; categories: pass→fail, pass→missing, fail→pass, new)
3. Run the serial baseline in a worktree at origin/dev; record artifacts and fill audit doc §8
