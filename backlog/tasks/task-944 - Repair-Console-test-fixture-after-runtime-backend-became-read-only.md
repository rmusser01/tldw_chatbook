---
id: TASK-944
title: Repair Console test fixture after runtime backend became read-only
status: To Do
assignee: []
created_date: '2026-07-27 16:39'
labels:
  - tests
  - console
  - baseline
dependencies: []
references:
  - Tests/UI/test_screen_navigation.py
  - Tests/UI/test_console_native_transcript.py
  - Tests/UI/test_console_tick_gating.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the shared Console navigation test harness after current_runtime_backend became a read-only derived property on dev. The fixture must configure runtime state through the supported owner instead of assigning the property directly, without changing production runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console native-transcript and tick-gating tests construct TldwCli without assigning current_runtime_backend
- [ ] #2 The affected tests pass against current dev without changing production runtime ownership
- [ ] #3 The repair remains isolated from TASK-553.16 citation implementation
<!-- AC:END -->
