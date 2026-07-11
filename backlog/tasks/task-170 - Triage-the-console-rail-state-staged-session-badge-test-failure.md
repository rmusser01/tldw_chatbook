---
id: TASK-170
title: Triage the console-rail-state staged/session badge test failure
status: To Do
assignee: []
created_date: '2026-07-11 22:03'
labels:
  - follow-up
  - console
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Chat/test_console_rail_state.py::test_console_context_rail_badge_ignores_empty_staged_summary fails ('staged' == 'session') independent of recent work — reproduced at pre-sweep HEAD by two separate reviews. A badge-priority ordering bug in build_console_context_rail_badge. Triage and fix, or correct the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The badge test passes,The underlying staged-vs-session priority is correct or the test expectation is corrected
<!-- AC:END -->
