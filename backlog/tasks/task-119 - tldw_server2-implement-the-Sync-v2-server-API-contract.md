---
id: TASK-119
title: 'tldw_server2: implement the Sync v2 server API contract'
status: To Do
assignee: []
created_date: '2026-06-12 20:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Milestone blocker found in TASK-70.6 QA: the chatbook Sync v2 client requires /api/v1/sync/capabilities, /sync/devices/register, /sync/datasets/enroll, /sync/push, /sync/pull, /sync/conflicts(+resolve), /sync/keys/recovery-bundle, /sync/attachments — tldw_server2 currently implements only legacy /sync/send and /sync/get, so Sync v2 enrollment dies at capabilities (404) and manual Notes/Chat sync cannot run end-to-end against the approved local target. Server-side work (tldw_server2 repo), or point QA at a server branch that has the v2 endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sync v2 dry-run enrollment completes against the local server,Manual Notes+Chat sync push/pull round-trips end-to-end
<!-- AC:END -->
