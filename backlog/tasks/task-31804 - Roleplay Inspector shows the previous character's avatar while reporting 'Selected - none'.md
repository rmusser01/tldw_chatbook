---
id: TASK-31804
title: Roleplay Inspector shows the previous character's avatar while reporting 'Selected: none'
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - roleplay
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). After deselecting (or when selection clears), the Inspector's status line says 'Selected: none' but the previously selected character's avatar stays rendered - contradictory state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When selection is none, the Inspector shows no stale avatar.
<!-- AC:END -->
