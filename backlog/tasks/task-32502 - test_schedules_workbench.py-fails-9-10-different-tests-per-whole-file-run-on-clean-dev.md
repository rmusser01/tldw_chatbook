---
id: TASK-32502
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landing measurement (2026-09-12, branch `fix/library-notes-w3-test-health`
vs a detached `origin/dev` 7159fc0b99, captures under
`wave3-caps/test-health/land3-*`): the whole-file chunks put five
`test_schedules_workbench.py` names in the branch-only FAILED set
(`test_committing_timezone_edit_preserves_cron`,
`test_header_paints_checking_not_a_false_unreachable_during_mount_probe`,
`test_queue_definition_pane_repaints_after_a_successful_in_pane_edit`,
`test_runs_on_cancel_button_cancels_the_dormant_copy_using_its_own_id`,
`test_runs_on_dropdown_confirm_dialog_lists_warnings`). Run as node ids,
serially: `test_header_paints_checking…` fails 2/2 on BOTH trees;
`test_queue_definition_pane_repaints…` fails 1/4 on the branch and 2/4 on dev
(`land3-sched-{branch,dev}-test_queue_definition…-{1..4}.out`);
`test_committing_timezone_edit…` 0/4 on the branch, 1/4 on dev; the two
`runs_on` names pass 2/2 on the branch (`land3-isolation-*.out`). So the
branch-only set is this file's instability, not the harness pin: every name
also fails on dev alone, and the same five appear across the earlier dev
whole-file runs (`sched-dev.txt`, `sched-dev-2.txt`).
<!-- SECTION:NOTES:END -->
