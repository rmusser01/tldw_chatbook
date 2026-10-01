---
id: TASK-19425
title: Core CI jobs now exceed their 120-minute ceiling and get canceled mid-suite
status: In Progress
assignee:
  - '@rmusser01'
created_date: '2026-08-21'
updated_date: '2026-10-01 01:03'
labels:
  - ci
  - testing
  - triage
dependencies: []
priority: high
---

## Description (the why)

With TASK-19160's fix in (PR #1858), the Core jobs no longer die on the
xdist INTERNALERROR — the workers survive the full run. What that exposed:
**both Core jobs now hit `timeout-minutes: 120`** (`test.yml` line 50) and
are canceled while still making progress, so the job reports no test
summary at all.

The growth trend predates 19160 and is visible in merged PRs' Core
durations (ubuntu): **#1840 79 min → #1835 87 min → #1823 121 min
(canceled) → #1858 120 min (canceled, both platforms)**. Dev gained ~17
PRs on 2026-08-19/20 (realtime voice, research runs, focus mode, latency
guardrails, …), and the all-but-UI suite was ~20.6k tests at the last
completed count.

Confounder to rule out at measurement time: the #1858 runs executed on
runners after a ~7-hour queue backlog (03:40–06:00 UTC), which may inflate
wall-time; #1835's runs at similar hours took 61–87 min, so backlog alone
does not explain the full jump.

## Acceptance Criteria (the what)

- [ ] The slowest test files/modules in a Core run are measured (pytest
      `--durations` or the json-report artifact), not guessed — naming
      whether the growth is a few pathological tests, a hang, or broad
      accretion
- [ ] A deliberate remedy ships: split/shard the Core job, raise the
      ceiling with the reason recorded, or fix the named slow tests —
      chosen from the measurement, with the owner consulted if the ceiling
      moves
- [ ] Both Core jobs complete (pass or fail) within their ceiling on a PR
      run, reporting a real test summary
- [ ] Any per-test timeout interaction is checked: `--timeout=300` with
      `timeout_method = "thread"` kills the whole worker process on a
      single hung test, which under `--max-worker-restart=3` can silently
      re-run large scopes and multiply wall-time

## Implementation Plan

1. Measure from real artifacts, not guesses: locate recent completed
   core-shard runs of `.github/workflows/test.yml` via `gh run list`,
   download their `core-test-results-<shard>` json-report artifacts, and
   rank slowest tests/files plus per-stage (setup/call/teardown) totals.
2. Classify the growth: pathological hangs (tests pinned at the
   `--timeout=300` cap), a few heavy soak/qualification tests, or broad
   accretion; verify each named pathology still reproduces at the current
   base before touching code.
3. Fix the pathologies the measurement names (test-code changes only;
   production changes only if a measured test exposes a real product
   hang), verifying each fix with a targeted local before/after run.
4. Check the `--timeout=300` / `timeout_method` / `--max-worker-restart=3`
   interaction analytically against the flags as they exist today.
5. Record the measurement table, the remedy chosen, and any ceiling or
   sharding decision that only the owner can make, in Implementation
   Notes; leave AC #3 (live CI proof within ceiling) explicitly pending
   unless a real run demonstrates it.
