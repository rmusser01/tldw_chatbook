---
id: TASK-21119
title: >-
  Every Chat-screen press walks the whole screen DOM twice for selection-menu dismissal
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21119).

`chat_screen.py:18939-18990`: `_dismiss_console_selection_menus_outside_transcript` runs
`self.query(ConsoleTranscript)` and `self.query(ConsoleSelectionMenu)` - two full-screen DOM
traversals - and is invoked on BOTH on_mouse_down and on_click of the same physical press
(~4 traversals per click) on the largest-DOM screen in the app. A direct contributor to the
click-lag symptom on every click.

## Acceptance Criteria

- [ ] Dismissal early-returns via a mounted-menu flag/registry (at most one menu is ever mounted) and a cached transcript reference - no full-screen queries when nothing is mounted
- [ ] Selection-menu dismissal behavior is unchanged (covered by existing selection tests)
