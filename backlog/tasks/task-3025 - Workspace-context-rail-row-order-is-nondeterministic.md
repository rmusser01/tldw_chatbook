---
id: TASK-3025
title: Workspace context rail row order is nondeterministic
status: To Do
assignee: []
created_date: '2026-08-07 16:19'
labels:
  - bug
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_workspace_context_rail.py::test_conversation_status_row_label_and_value_are_separate_visual_runs fails nondeterministically IN ISOLATION -- measured 1 pass / 2 fail and 2 pass / 1 fail across two arms of three runs each, with and without an unrelated change, so it is not order-dependence across files. The assertion reads _composited_rows(scope_pair)[0] and intermittently gets the Model row where it expects Conversation, i.e. the composited rows arrive in a different order between runs. Worth investigating as a product nondeterminism in the rail rather than only as a flaky test: if row order varies, the rail can render its scope pairs in an unstable order for real users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail's composited row order is deterministic, or the test asserts on identity rather than index with the reason recorded
- [ ] #2 The test passes 10 consecutive isolated runs
<!-- AC:END -->
