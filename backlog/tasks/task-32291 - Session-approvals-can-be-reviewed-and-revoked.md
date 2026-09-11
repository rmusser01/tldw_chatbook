---
id: TASK-32291
title: Session approvals can be reviewed and revoked
status: To Do
assignee: []
created_date: '2026-09-10 19:18'
labels:
  - mcp
  - approvals
  - permissions
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Approve for session' is held in memory with no caller of the clear path; the only way to revoke it is restarting the app, and nothing shows which tools are session-approved. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Session-approved tools are listed (inspector or Permissions) with a revoke action.
- [ ] #2 Revoking makes the next call to that tool ask again.
- [ ] #3 Docs state that a session approval lasts until Chatbook exits or is revoked.
<!-- AC:END -->
