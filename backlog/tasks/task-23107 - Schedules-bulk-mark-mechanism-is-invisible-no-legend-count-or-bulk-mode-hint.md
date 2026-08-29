---
id: TASK-23107
title: >-
  Schedules bulk-mark mechanism is invisible: no legend, count, or bulk-mode
  hint
status: To Do
assignee: []
created_date: '2026-08-28 14:06'
labels:
  - ux
  - schedules
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing x prefixes a row with a filled-circle mark and missed-while-away rows carry a diamond glyph, but there is no legend, no marked-count, and no indication that space/d switch to bulk mode when marks exist. Cheap fix shape: reuse the existing #scheduling-pane-notice line ('2 marked - space toggles all, d deletes all, esc clears'). P2 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When one or more rows are marked, visible text states the marked count and which keys act on all marked rows and how to clear the marks
- [ ] #2 The missed-while-away glyph has a visible text explanation on screen
<!-- AC:END -->
