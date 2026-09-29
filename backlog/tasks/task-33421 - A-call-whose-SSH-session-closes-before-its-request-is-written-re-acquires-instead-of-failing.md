---
id: TASK-33421
title: A call whose SSH session closes before its request is written re-acquires instead of failing
status: To Do
assignee: []
created_date: '2026-09-28 17:39'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33401
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
If the laptop closes a healthy session (idle reap, run end, app exit) after a call registered its request but before the REQUEST bytes were written, the call returns REMOTE_OP_FAILED although nothing was sent; TASK-33401 only covered calls not yet registered. Found by the final review of the TASK-33400..33408 branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A call whose request bytes never reached the session re-acquires (fresh session, or one-shot after run end or app exit)
- [ ] #2 A call whose request was written is never retried
<!-- AC:END -->
