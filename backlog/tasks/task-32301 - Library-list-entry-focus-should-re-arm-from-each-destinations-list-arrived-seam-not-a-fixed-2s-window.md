---
id: TASK-32301
title: >-
  Library list-entry focus should re-arm from each destination's list-arrived
  seam, not a fixed 2s window
status: To Do
assignee: []
created_date: '2026-09-10 21:30'
labels:
  - library
  - focus
  - keyboard
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Entry focus (task-2856) arms a fixed LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS window when a rail row is pressed, so a destination whose list takes longer than that to arrive lands focus nowhere and the keys that depend on focus being in the canvas go unadvertised. task-32260 measured Library at 12.6s to open on a seeded profile, which makes the miss routine on a session's first visit rather than rare. Measured on fix/library-crit9-shell: a cold first visit to Conversations misses the window and lands nowhere; a warm re-entry lands every time; a cold Prompts visit lands in its filter rather than row 0. Raised by the critique-9 shell re-review as the rider for task-32228's accepted ceiling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A destination's list arriving after the current window still receives entry focus, without widening the window for every canvas
- [ ] #2 The first visit of a session to Conversations on a seeded profile lands focus on its first row
- [ ] #3 A user interaction during the wait still cancels the pending focus, as the current settle window does
<!-- AC:END -->
