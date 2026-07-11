---
id: TASK-161
title: Persistent ingest job history across restarts
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library ingest job registry is in-memory only; queued/failed jobs are lost on quit. Persist job history so users can review and retry across restarts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ingest job history survives an app restart,Failed/queued jobs can be retried after restart
<!-- AC:END -->
