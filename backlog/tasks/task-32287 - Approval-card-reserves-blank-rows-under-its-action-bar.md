---
id: TASK-32287
title: Approval card reserves blank rows under its action bar
status: To Do
assignee: []
created_date: '2026-09-10 19:16'
labels:
  - console
  - approvals
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A one-row batch renders a 17-row card at 50 rows, with roughly ten empty rows between the action bar and the bottom border; at 80x30 the action bar was not on screen. No CSS exists for the card container, its action bar or ChatTaskCards; cause not traced. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The card's height matches its content (title, optional deadline and summary, rows, action bar) plus padding.
- [ ] #2 At 80x24 a one-row card shows its action bar.
- [ ] #3 A mounted test pins the height for one-row and three-row batches.
<!-- AC:END -->
