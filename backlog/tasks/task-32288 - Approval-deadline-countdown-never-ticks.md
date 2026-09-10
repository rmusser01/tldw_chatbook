---
id: TASK-32288
title: Approval deadline countdown never ticks
status: To Do
assignee: []
created_date: '2026-09-10 19:16'
labels:
  - console
  - approvals
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The deadline line is rendered once when the batch is set; with [mcp] approval_timeout_seconds configured the card shows a fixed 'Auto-denies in 2:00' for the whole window. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a finite timeout the countdown updates at least once per second and stops when the card hides.
- [ ] #2 With the default of 0 nothing is shown.
<!-- AC:END -->
