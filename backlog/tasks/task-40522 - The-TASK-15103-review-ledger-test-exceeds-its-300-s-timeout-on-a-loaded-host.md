---
id: TASK-40522
title: The TASK-15103 review-ledger test exceeds its 300 s timeout on a loaded host
status: To Do
assignee: []
created_date: '2026-11-12 14:40'
labels:
  - tests
  - flaky
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Architecture/test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_canonical_schema_and_arithmetic fails with `Failed: Timeout (>300.0s) from pytest-timeout`. Measured 2026-11-12 while rebasing TASK-36571.1, on a host at load1 40-58 from other sessions' pytest runs and load generators: it timed out in 2 of 2 runs on a detached dev 4b90eb860b worktree and 2 of 2 on the TASK-36571.1 head (once inside a whole `Tests/Architecture` run, once with four nodes selected). The branch does not touch the inventory or the ledger. Whether the test is slow in itself (a whole-repository parse or regeneration per run) or only on a loaded host is not yet known; TASK-39126 records a similar whole-repository parse cost in the Architecture guards job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's wall time is measured on a quiet host and under load, with where the time goes
- [ ] #2 The test finishes well inside its timeout on a host running two other pytest processes, 5 of 5 runs
- [ ] #3 The ledger checks it makes are unchanged (a mutant ledger row still fails it)
<!-- AC:END -->
