---
id: TASK-247
title: Remove duplicate Console save_state on screen suspend
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, console]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
app.py:4611-4621 calls save_state() and stores the result before switch_screen; Textual's ScreenSuspend then fires ChatScreen.on_screen_suspend (chat_screen.py:10651-10655) which calls save_state() AGAIN and discards the result — a full O(sessions×messages) native-console serialization wasted on every tab switch away from Console. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console save_state runs exactly once per tab-switch-away (test counts invocations)
- [ ] #2 State restoration behavior unchanged (existing round-trip tests green)
<!-- AC:END -->
