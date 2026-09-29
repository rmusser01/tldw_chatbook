---
id: TASK-33420
title: SSH session start stays warm when the budget runs out mid bundle upload
status: To Do
assignee: []
created_date: '2026-09-28 17:39'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33400
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a cache miss, a call budget that expires while the bundle is being uploaded makes the session start fail protocol-class (`_WriteStalled`), switching the binding to one-shot calls for the rest of the run although the host is reachable. Found by the final review of the TASK-33400..33408 branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A budget expiry during the bundle upload fails only that call and does not switch the run to one-shot calls
- [ ] #2 The binding's status is unchanged by it
<!-- AC:END -->
