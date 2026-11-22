---
id: TASK-42247
title: "Console tray tests from TASK-38204 hand-roll startup settling that the shared rail-test helpers already do"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_workspace_context_rail.py
- backlog/tasks/task-38204 - Tests-a-state-pushed-straight-into-the-Console-Conversations-tray-is-overwritten-by-the-Consoles-own-startup-syncs.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the testsb stream (round 19). TASK-38204's tray tests push state into the Console before its own startup syncs finish. The testsb branch (`fu/r19-testsb`, TASK-42148) adds `_wait_for_console_startup_settled`, `_hold_workspace_state` and `_wait_for_selector` to `Tests/UI/test_console_workspace_context_rail.py`. Those helpers live in one test file, so the tray tests cannot use them without importing from another test module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The helpers sit in one shared `Tests/UI` support module and both the rail file and the tray tests use them
- [ ] #2 No tray test keeps its own copy of the settle-then-push wait
- [ ] #3 The affected files pass through the scratch wrapper at `-n 4`, with the load recorded
<!-- AC:END -->
