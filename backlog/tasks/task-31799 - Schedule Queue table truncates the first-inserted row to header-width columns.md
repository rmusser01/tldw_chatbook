---
id: TASK-31799
title: Schedule Queue table truncates the first-inserted row to header-width columns
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - schedules
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Creating the first reminder renders the queue row with Title clipped to 5 chars and Details to 7 ('UAT f' / 'One-tim') despite ample free width; any filter keystroke re-renders with correct widths and it stays correct. Columns are auto-sized before the first content insert and not re-measured on row add.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first row added to an empty Schedule Queue renders with correctly measured column widths.
<!-- AC:END -->
