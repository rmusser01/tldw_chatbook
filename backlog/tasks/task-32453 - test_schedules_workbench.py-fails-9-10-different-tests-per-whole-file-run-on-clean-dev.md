---
id: TASK-32453
title: >-
  test_schedules_workbench.py fails 9-10 different tests per whole-file run on
  clean dev
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - scheduling
  - tests
  - flake
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run as a whole file, `Tests/UI/test_schedules_workbench.py` fails a large and
UNSTABLE set of tests on clean `dev` -- the count and the names both change
between runs, so nobody reading that file can tell a regression from the
noise.

Measured at dev ff2dc03145 in a detached worktree, `pytest -q -p no:randomly`,
whole file:

- dev run 1: 4 failed / 177 passed
- dev run 2: 9 failed / 172 passed
- the two dev runs share only one failing name

The same file on a branch that differed only in a harness `CSS_PATH` pin
gave 8 and 10 failures, with a branch-only set in run 1 that had entirely
turned over by run 2 -- and three of those names, run as node ids in
isolation, behave IDENTICALLY on both trees (1 failed / 2 passed each). One
failure reads `'_SlowProbeServerClient' object has no attribute
'list_automation_results'` (a sync-engine error banner leaking into a header
assertion), which suggests shared background sync work outliving the test
that started it.

Found while sweeping harness stylesheet pins for task-32204; not caused by
that sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Three consecutive whole-file runs of `Tests/UI/test_schedules_workbench.py` on clean dev produce the SAME result, and that result is recorded
- [ ] #2 The cross-test leak is identified (the `list_automation_results` banner is the visible symptom) and the work that outlives its test is stopped or awaited
- [ ] #3 Whatever reds remain after that are either fixed or filed individually, so the file has an owned baseline
<!-- AC:END -->
