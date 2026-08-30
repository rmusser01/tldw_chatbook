---
id: TASK-24602
title: Send-authority Run has no failed state so a failed run reads as Ready
status: To Do
assignee: []
created_date: '2026-08-30 00:53'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
project_console_send_authority derives Run from the branches: Inspector data incomplete, Recovery required, Waiting for approval, Blocked, Running, else Ready. There is no failed branch. A turn that returned HTTP 401 left the pinned authority block reading Run: Ready and Provider: ready while the transcript showed the failure, so the one surface pinned above the fold to answer what happens if I send now contradicted the transcript.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run that ended in provider or transport failure renders a distinct Run state, not Ready
- [ ] #2 The failed state names the failure and a specific next action
- [ ] #3 A test asserts the projection returns the failed state for a failed run and Ready only when no failure is outstanding
<!-- AC:END -->
