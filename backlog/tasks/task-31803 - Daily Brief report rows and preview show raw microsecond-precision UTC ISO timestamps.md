---
id: TASK-31803
title: Daily Brief report rows and preview show raw microsecond-precision UTC ISO timestamps
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ux
  - artifacts
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Report list rows and the preview header render timestamps like '2026-09-05T23:10:2...' raw. Format for humans (local time, minute precision) in list and preview surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Brief timestamps render in a human-readable local format in the list and preview.
<!-- AC:END -->
