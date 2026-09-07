---
id: TASK-28233
title: CJK wide-character alignment in chat tables
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - ux
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C35, promoted by TASK-26041: table alignment breaks with wide CJK glyphs. wcwidth is now proven in-tree (Terminal/screen_model.py uses it), so the fix is applying the same measurement to chat/table rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tables containing CJK text align correctly (cell widths measured by display width, not len())
- [ ] #2 ASCII-only rendering is unchanged
<!-- AC:END -->
