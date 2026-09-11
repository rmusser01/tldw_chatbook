---
id: TASK-32340
title: >-
  Prefill rows state the tools-skipped side effect
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review D4. While a prefill is armed, tool calling (including MCP) is silently skipped for that send; the docs disclose it but the Inspector's 'Prefill (next send only)' / 'Prefill (pinned)' rows do not. Append the consequence to the armed row so the state is self-describing.

Filed from the 2026-09-10 Console rail UX review (review item D4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Armed prefill rows indicate that tools will be skipped for the send
- [ ] #2 Unarmed/absent prefill rendering is unchanged
- [ ] #3 Row status class semantics unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): labels at chat_screen.py:14265-14292; prefill genuinely blocks agent/tools dispatch (console_chat_controller.py:17170-17178) and nothing surfaces it in the rows.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
