---
id: TASK-30043
title: Media items pane - legible action grammar at the narrow pane width
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-03 13:05'
updated_date: '2026-09-03 13:13'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique 2026-09-03 P1: at the items pane's 40-col floor the six-button toolbar chops to 't so E Tr R Se', select-mode actions render as bare disabled markers, and the armed-confirm safety copy clips mid-word; tooltips are mouse-only. User ruling on the critique's Q1: yes - the horizontal toolbar is the wrong grammar for this pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At the narrow pane width every list action shows a readable text label (no chopped fragments, no marker-only buttons)
- [ ] #2 Select-mode actions and the selected-count remain readable at the narrow width
- [ ] #3 The armed bulk-delete confirmation copy wraps instead of clipping
- [ ] #4 The wide layout keeps its current single-row presentation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Screen threads the measured items-pane width into the canvas (LibraryMediaRowGeometryChanged already delivers it) as narrow_actions bool\n2. Narrow grammar: toolbar splits into two auto-height rows with short REAL labels (type/sort/Export | Trash/Review/Select); select mode splits count+select-all+clear / three bulk actions; wide keeps one row\n3. Armed-confirm copy Static wraps (height auto) instead of clipping\n4. Live verify at the 40-col pane + wide
<!-- SECTION:PLAN:END -->
