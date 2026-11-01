---
id: TASK-38202
title: >-
  Console: a case-only or trailing-space edit in the Conversations filter forces
  a full tray rebuild
status: To Do
assignee: []
created_date: '2026-10-30 06:06'
labels:
  - console
  - left-rail
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact.** Since TASK-33622.6 the conversation browser state keeps the filter's raw text, so a keystroke that changes only case or trailing whitespace (rows unchanged) still changes the state's query. The tray's in-place patch path then refuses the delta and the tray does a full rebuild of its children around the kept filter row. It is cheap with the kept row, but it is one avoidable rebuild per such debounced pass.

**Evidence.** TASK-33622.6 review: ConsoleConversationBrowserState.query now stores str(query); _browser_delta_may_patch in Widgets/Console/console_workspace_context.py treats a query change as unpatchable. Cost not measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A debounced pass whose rows and counts are unchanged and whose query differs only in case or surrounding whitespace updates the tray without a full rebuild
- [ ] #2 The filter input still shows exactly what the user typed while focused, and an unfocused filter still shows the state's query (TASK-33622.6 holds)
- [ ] #3 A test counts tray rebuilds across a case-only edit and fails on the pre-fix code
<!-- AC:END -->
