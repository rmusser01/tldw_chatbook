---
id: TASK-32279
title: Transcript labels a user Deny as blocked and shows model-facing refusal text
status: To Do
assignee: []
created_date: '2026-09-10 19:12'
labels:
  - console
  - approvals
  - ux-copy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Deny, the tool marker reads 'blocked' and expanding it shows 'tool call denied by the user: ... Do not retry this call ...' under 'Full output', which is the instruction meant for the model. The card says Deny, the matrix says Off, the audit says denied, and the transcript says blocked. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user denial renders a user-facing status ('denied by you') distinct from a policy block.
- [ ] #2 The model-facing instruction stays in the collapsed detail under a label that says it was sent to the model.
- [ ] #3 One vocabulary table for allow, ask, off and deny states is used by the card, matrix, inspector, transcript and audit.
<!-- AC:END -->
