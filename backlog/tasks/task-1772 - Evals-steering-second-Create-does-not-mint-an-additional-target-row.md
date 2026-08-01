---
id: TASK-1772
title: 'Evals steering: second Create does not mint an additional target row'
status: To Do
assignee: []
created_date: '2026-08-01 21:10'
labels:
  - evals
  - test-failure
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_evals_steering_e2e.py::test_two_ui_authored_targets_one_steered_light_up_column_mode_delta fails on dev with 'AssertionError: second Create must mint an ADDITIONAL row'. Found while running the full Tests/UI sweep for TASK-596 Phase 1; verified pre-existing by reproducing it in a checkout containing none of that branch's code. Everything else in Tests/UI passes (2486 passed, 1 skipped), so this is the only red test in the directory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failing test passes, or is replaced by one that pins the intended behavior
- [ ] #2 Root cause is stated: whether the second Create genuinely fails to add a row, or the test's assumption about row identity is wrong
<!-- AC:END -->
