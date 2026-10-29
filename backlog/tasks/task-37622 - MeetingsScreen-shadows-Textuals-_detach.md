---
id: TASK-37622
title: MeetingsScreen shadows Textual's _detach
status: To Do
assignee: []
created_date: '2026-10-29 12:32'
labels:
  - meetings
  - textual
dependencies:
  - TASK-37620
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MeetingsScreen defines a private _detach method (it unsubscribes from the live session). That name is Textual's hook for clearing a removed node's parent, which Textual calls when a screen is removed, and app_keep_alive also calls screen._detach() directly. The same shadowing in the Console row menus (TASK-37620) left closed menus reporting a parent and claiming the next Esc. Here a removed MeetingsScreen may keep its parent, and the session unsubscribe runs from Textual's teardown path rather than where the screen means it to.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MeetingsScreen no longer defines a method named _detach; its session unsubscribe keeps working where it is called today
- [ ] #2 A test removes or uninstalls a MeetingsScreen and asserts its parent is cleared and its session listener is unsubscribed exactly once
<!-- AC:END -->
