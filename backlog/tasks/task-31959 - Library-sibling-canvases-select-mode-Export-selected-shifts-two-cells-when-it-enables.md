---
id: TASK-31959
title: >-
  Library sibling canvases - select-mode Export selected shifts two cells when
  it enables
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J Task 2 review I3: _apply_library_row_toggle patches the conversations, notes and prompts row buttons in place, so their select-mode 'Export selected' label moves two cells when the enabled state flips (library_conversations_canvas.py ~199, library_notes_canvas.py ~721 and ~1507). PR J padded only the Media select-mode row buttons, via library_disabled_action_label(align=True).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The sibling canvases' select-mode action labels hold their column across an enabled-state flip
- [ ] #2 Painted column pins cover each sibling canvas that carries the row toggle
<!-- AC:END -->
