---
id: TASK-3025
title: Workspace context rail row order is nondeterministic
status: Done
assignee: []
created_date: '2026-08-07 16:19'
updated_date: '2026-08-07 19:45'
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
- [x] #1 The rail's composited row order is deterministic, or the test asserts on identity rather than index with the reason recorded
- [x] #2 The test passes 10 consecutive isolated runs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the assertion read `_composited_rows(container)[0]` -- an INDEX
into whatever the compositor had painted at that instant. Sampled mid-layout it
returned a neighbouring pair's row (the observed failure got `'Model'` where
`'Conversation'` was expected).

Resolved via AC option 2: the row is now found from the label widget's own
identity, waiting until the paint agrees with the widget tree, so the rail's row
ORDER is no longer load-bearing for a test that is really about label/value
separation. 10 of 10 isolated runs pass at load average ~8, where it was roughly
1-in-3 before. Mutation-verified that the assertion still binds to the real
painted row (`'Conversation —'`), so determinism did not cost meaning.
<!-- SECTION:NOTES:END -->
