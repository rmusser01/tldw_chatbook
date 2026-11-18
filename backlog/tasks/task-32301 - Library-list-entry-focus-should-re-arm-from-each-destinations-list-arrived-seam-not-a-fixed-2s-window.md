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

TASK-41532 (2026-11-18): the cold Prompts visit lands in its filter because `_focus_library_list_entry` (`UI/Screens/library_screen.py`, the empty-list `fallback_selector`) takes the filter fallback when its attempt finds no rows yet; Prompts and Conversations have no wait for loading rows there (Notes got one in TASK-41129, and Skills has one). `Tests/UI/test_library_shell.py::test_slash_focuses_the_prompts_filter_not_the_rail_search` failed on every push-to-dev run from this until it focused the row itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A destination's list arriving after the current window still receives entry focus, without widening the window for every canvas
- [ ] #2 The first visit of a session to Conversations on a seeded profile lands focus on its first row
- [ ] #3 A user interaction during the wait still cancels the pending focus, as the current settle window does
- [ ] #4 A cold Prompts visit whose rows arrive after entry focus's first attempt lands on row 0, not the filter
<!-- AC:END -->
