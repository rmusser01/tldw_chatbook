---
id: TASK-31800
title: Reminder form 'Run at' placeholder and examples hardcode a past datetime
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - schedules
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). The 'Run at' placeholder and helper/error copy use the literal '2026-08-28 09:00' (already in the past): UI/Screens/scheduling/forms/reminder_form.py:513, definition_detail.py:1266 and :1606. A user copying the example gets an already-due run time. Generate the example relative to now, or use an obviously-future fixed date.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The run-at example shown to users is always a future datetime (or clearly synthetic).
- [ ] #2 All three literal sites updated consistently.
<!-- AC:END -->
