---
id: TASK-32599
title: Restore bounded Library queries during focus changes and same-band resizing
status: To Do
assignee: []
created_date: '2026-09-15 02:48'
labels:
  - library
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-23025 query-budget gates reproduce failures at baseline 2939afda63: 23 Library queries across three non-crossing resize frames (expected zero), and five per Tab (ceiling one). Nearest production call-site instrumentation attributes most work to _active_library_rail and _library_focusable, plus four ordinary-rail width queries and one on_resize query. This is measured excess query work; visible latency has not been established.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Non-crossing resize frames meet the established zero-Library-query gate while actual layout-band crossings still apply the correct reader and rail geometry.
- [ ] #2 Tab focus changes meet the established ceiling of one Library query and do not trigger whole-screen recomposition.
- [ ] #3 Keyboard traversal, visible focus, reader return and restored rail widths remain correct with production styles at 80 and 120 columns.
- [ ] #4 The targeted task-23025 gates pass without relaxing budgets merely to match the regression; any necessary contract change is supported by explicit behavior and measurement evidence.
<!-- AC:END -->
