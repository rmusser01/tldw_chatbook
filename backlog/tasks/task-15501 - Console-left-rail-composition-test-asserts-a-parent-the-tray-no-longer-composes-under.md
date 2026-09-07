---
id: TASK-15501
title: >-
  Console left-rail composition test asserts a parent the tray no longer
  composes under
status: To Do
assignee: []
created_date: '2026-08-11 19:48'
labels:
  - bug
  - console
  - tests
  - pre-existing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space asserts workspace_context.parent is session_body, but the tray now composes under console-rail-section-body-conversations. Either the composition changed without the test following, or the test encodes a layout intent the composition has drifted from; the test is red on this checkout and no longer guards the space-usage property it is named for.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space passes on a clean dev checkout
- [ ] #2 It is recorded which side was wrong: the composition drifted, or the test's expected parent was stale
- [ ] #3 The test still fails if a rail section stops using the available space, i.e. it is repaired rather than relaxed to vacuity
<!-- AC:END -->
