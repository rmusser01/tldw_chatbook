---
id: TASK-1844
title: 'Approval deadline is invisible and its verdict is unlabelled'
status: To Do
assignee: []
created_date: '2026-08-01 19:30'
labels:
  - console
  - ux
  - agents
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ChatApprovalCard.set_batch` accepts `timeout_seconds` (`chat_approval_card.py:274`) and never reads it, while its own docstring claims the value is "surfaced on the card". The controller sets a 120s deadline. So a clock the user cannot see decides for them.

When it expires, `format_agent_step_marker` (`console_agent_bridge.py:354-369`) has no timeout branch -- the card simply disappears. The user cannot distinguish "the system gave up" from "I denied it" from "it never ran".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The remaining time is visible on the card whenever a deadline is armed
- [ ] #2 An expired approval produces a distinct transcript marker, worded to distinguish a system timeout from a user denial
- [ ] #3 If the deadline is not surfaced, the parameter and its docstring claim are removed instead
- [ ] #4 A test asserts the timeout path produces its own marker text
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prefer surfacing over deletion: a deadline that changes behaviour must be visible. Test alongside the existing timeout coverage in `Tests/UI/test_console_mcp_approval.py`, which already asserts undecided rows deny on timeout but never checks what the user sees.
<!-- SECTION:NOTES:END -->
