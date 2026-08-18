---
id: TASK-17656
title: 'Console: message-action focus-walk test red on dev after selection-feedback merge'
status: Done
assignee:
  - '@claude'
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
- [x] #1 The action-row keyboard focus walk is verified live: every action in a selected message's row is reachable by Tab in visual order, including the new feedback action
- [x] #2 The test is green on dev, updated to the intended contract (or the focus-order defect it caught is fixed)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce; identify what added the stop between Copy and Edit.
2. Walk the row live; decide regression vs stale pin; fix accordingly.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The filing's attribution was half right: the failure rode in with the selection-feedback merge train, but the mechanism was NOT the feedback action — commit `8ae87242a` ("move idle Speak into the selected-message action row", 2026-08-15, merged via that branch) deliberately moved Speak from the message header into the row between Copy and Edit, exactly where the on-screen guide has always listed it. Keyboard navigation was never broken; the walk test's expectations were stale. The walk gains the Speak stop, and — to honor AC#1 — now continues full circle through Regenerate, Continue, both feedback thumbs, and back to Delete, proving every stop of a completed assistant reply is Tab-reachable in visual order with focus contained, before re-focusing Save-as for the modal ending. Green with zero production changes.

Files: `Tests/UI/test_console_native_chat_flow.py`.
<!-- SECTION:NOTES:END -->
