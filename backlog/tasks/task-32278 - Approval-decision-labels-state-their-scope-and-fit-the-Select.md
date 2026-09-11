---
id: TASK-32278
title: Approval decision labels state their scope and fit the Select
status: To Do
assignee: []
created_date: '2026-09-10 19:11'
labels:
  - console
  - approvals
  - ux-copy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The card offers five decisions with no scope text; the 26-cell Select clips 'Approve for session' to 'Approve for' and wraps 'Always allow this exact input'. Users cannot tell how long a grant lasts or where to undo it, and the '(high risk)' badge explains itself only on hover with a reads-only sentence that is also used for mutating tools. User decision 2026-09-10: keep all five decisions on the card and add scope copy. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every decision label fits the closed Select without clipping at the card's minimum supported width.
- [ ] #2 One line under the controls states the selected decision's scope (this call only; until Chatbook exits; remembered for this tool with the place to change it; remembered for these arguments) and where to undo it.
- [ ] #3 The high-risk explanation is visible without hover and differs for reads and mutations.
- [ ] #4 The user guide lists the same five decisions with the same scopes.
<!-- AC:END -->
