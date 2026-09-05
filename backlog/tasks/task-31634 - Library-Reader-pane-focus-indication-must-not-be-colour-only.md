---
id: TASK-31634
title: Library Reader pane - focus indication must not be colour-only
status: To Do
assignee: []
created_date: '2026-09-05 06:18'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P2 (measured): the Reader content pane's focus is carried by its top border recolouring from 1.01:1 to 6.96:1 with byte-identical glyphs, so a monochrome or colour-vision-deficient user gets no signal that F6 moved focus into the Reader; buttons on the same screen get a heavy outline glyph change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Reader pane's focused state changes glyphs (heavy outline or equivalent), visible in a plain-text capture
- [ ] #2 A painted test diffs the plain capture between unfocused and focused states
<!-- AC:END -->
