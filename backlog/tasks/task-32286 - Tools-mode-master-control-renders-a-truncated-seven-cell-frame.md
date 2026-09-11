---
id: TASK-32286
title: Tools-mode master control renders a truncated seven-cell frame
status: To Do
assignee: []
created_date: '2026-09-10 19:15'
labels:
  - mcp
  - tools-mode
  - ui
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 'Local workspace, web, and Watchlists tools' control at the top of Tools mode renders as a tiny bordered frame showing a truncated checkbox beside an 'Enabled' label at 250 columns. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The master control renders its checkbox and label fully at 80 to 250 columns.
- [ ] #2 A real-bundle CSS harness test pins the control's width.
<!-- AC:END -->
