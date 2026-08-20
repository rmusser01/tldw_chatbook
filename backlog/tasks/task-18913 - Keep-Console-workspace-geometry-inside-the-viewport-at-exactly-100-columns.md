---
id: TASK-18913
title: Keep Console workspace geometry inside the viewport at exactly 100 columns
status: To Do
assignee: []
created_date: '2026-08-20 07:07'
labels:
  - console
  - ux
dependencies:
  - TASK-18912
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the reproduced exact-100-column Console rail state from expanding the grid far beyond the viewport so every supported persona retains a visible transcript and usable rail controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 At exactly 100x30, the default Context-only state and every explicitly supported rail state keep every displayed Console workspace child within the viewport; the transcript remains visible and usable.
- [ ] #2 Effective rail priority, compact overrides, stored preferences, and the 70/74-column usable-transcript floors remain consistent with ADR-043.
- [ ] #3 Existing geometry and access behavior at 80x24, 120x30, 160x45, and 235x52 do not regress.
- [ ] #4 A production-CSS Textual compositor regression proves the 100-column failure and is mutation-checked against the geometry correction.
- [ ] #5 The change is terminal-specific and does not alter TASK-18911's phone, pointer, hover, or served-browser ownership.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-20-task-18913-console-exact-100-column-containment-design.md`.
<!-- SECTION:DESIGN:END -->
