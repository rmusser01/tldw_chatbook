---
id: TASK-17656
title: 'Console: message-action focus-walk test red on dev after selection-feedback merge'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - console
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_native_chat_flow.py::test_console_message_action_keyboard_focus_stays_inside_action_row` fails on clean origin/dev (verified 2026-08-17 in a detached baseline worktree at `52cb09d46`, immediately after the console-selection-feedback merge landed): the focus walk expects to reach `console-message-action-edit-<id>` but stops at `console-message-action-speak-<id>`. The selection-feedback arc added a Comment/feedback action to the message action row, which likely changed the row's Tab-stop count or ordering without the walk test being updated. Found during task-17651's post-rebase sweep — that branch neither touches the action row nor changes the failure (837/838 green with this one red both with and without the branch).

Fix belongs with the selection-feedback feature: either the action-row focus order regressed for keyboard users (real defect) or the walk test needs the new stop encoded (contract update). Decide which by walking the row in a live probe first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The action-row keyboard focus walk is verified live: every action in a selected message's row is reachable by Tab in visual order, including the new feedback action
- [ ] #2 The test is green on dev, updated to the intended contract (or the focus-order defect it caught is fixed)
<!-- AC:END -->
