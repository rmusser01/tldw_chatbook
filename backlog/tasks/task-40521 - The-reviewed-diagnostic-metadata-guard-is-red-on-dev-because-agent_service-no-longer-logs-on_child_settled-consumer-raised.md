---
id: TASK-40521
title: >-
  The reviewed-diagnostic metadata guard is red on dev because agent_service no
  longer logs "on_child_settled consumer raised"
status: To Do
assignee: []
created_date: '2026-11-12 14:40'
labels:
  - tests
  - diagnostics
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only fails on dev 4b90eb860b with two failures. The first reads "tldw_chatbook/Agents/agent_service.py: expected one diagnostic containing 'on_child_settled consumer raised', found 0": the test's reviewed-metadata table still lists that label (with field type(exc).__name__), but agent_service.py has no diagnostic with that text; `git log -S` names 6125fbc5b8 ("Integrate native Console phases with current dev") as the last change to it. Either the diagnostic was removed or reworded and the reviewed table is stale, or the review-gated diagnostic was lost.

Measured 2026-11-12 while rebasing TASK-36571.1: it failed 2 of 2 runs on a detached dev 4b90eb860b worktree and 2 of 2 on the TASK-36571.1 head, which does not touch agent_service.py or the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both failures the test reports on dev are explained: each reviewed diagnostic is either present with its reviewed fields or removed from the reviewed table with the reason
- [ ] #2 The test passes on dev
- [ ] #3 A child-settled consumer that raises still leaves a metadata-only diagnostic (exception type, no traceback), or the task records why none is needed
<!-- AC:END -->
