---
id: TASK-25729
title: Toggle controls use three different state encodings across the app
status: To Do
assignee: []
created_date: '2026-08-31 05:10'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Radio and checkbox states are drawn three different ways: filled versus hollow glyphs in the first-run wizard, a blank inner cell for checkboxes, and an identical glyph in both states distinguished only by dot colour in Console modals. In that last form the off state renders at roughly 1.4 to 1, so selection is invisible in any text-based capture and carried by colour alone, which the project's own accessibility guidance forbids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selected and unselected states differ by glyph shape, not colour alone
- [ ] #2 One toggle encoding is used consistently across wizard, modal and settings surfaces
- [ ] #3 The unselected indicator meets at least 3 to 1 contrast
<!-- AC:END -->
