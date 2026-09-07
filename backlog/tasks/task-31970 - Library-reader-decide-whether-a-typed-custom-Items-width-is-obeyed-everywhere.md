---
id: TASK-31970
title: 'Library reader: decide whether a typed custom Items width is obeyed everywhere'
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - reader
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documented at task-31953's close. Two resolver clamps in `Utils/adaptive_reader_state.py` (the library-closed clamp and the priority-pane clamp) widen a typed custom Items width — typed 32 becomes 52 at terminal width 100, and an existing pin encodes typed 40 → 56. This is a product call, not a bug: either the typed width wins wherever it fits, or the clamps stay and the settings copy says so.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written decision (in this task or an ADR note) states whether a typed Items width wins over the two clamps
- [ ] #2 The resolver pins and the Settings copy match that decision
<!-- AC:END -->
