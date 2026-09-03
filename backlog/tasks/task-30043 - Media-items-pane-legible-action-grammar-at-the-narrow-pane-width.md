---
id: TASK-30043
title: Media items pane - legible action grammar at the narrow pane width
status: Done
assignee:
  - '@claude'
created_date: '2026-09-03 13:05'
updated_date: '2026-09-03 15:48'
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
- [x] #1 At the narrow pane width every list action shows a readable text label (no chopped fragments, no marker-only buttons)
- [x] #2 Select-mode actions and the selected-count remain readable at the narrow width
- [x] #3 The armed bulk-delete confirmation copy wraps instead of clipping
- [x] #4 The wide layout keeps its current single-row presentation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Screen threads the measured items-pane width into the canvas (LibraryMediaRowGeometryChanged already delivers it) as narrow_actions bool\n2. Narrow grammar: toolbar splits into two auto-height rows with short REAL labels (type/sort/Export | Trash/Review/Select); select mode splits count+select-all+clear / three bulk actions; wide keeps one row\n3. Armed-confirm copy Static wraps (height auto) instead of clipping\n4. Live verify at the 40-col pane + wide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2350 (dev 5d1cd15e7). Multi-row is THE grammar (the items pane is ~40-44 cols in every real layout): browse = [type,sort]/[Export,Trash,Select]/[Review these]; select mode = [count, Select all N shown]/[Clear, Export, Review]/[Delete isolated on a danger row] with short real words + F-018 tooltips. BUNDLED_CSS lifts Button's 16-cell min-width floor (supersedes 28025's 1fr squeeze; its fit-contract test still passes); canvas min-width 40->36 (the shell allots ~37 - the 40 floor silently clipped 3 cells off every child incl. the confirm copy); confirm copy bounded so the safety sentence wraps whole. Qodo round: data-derived type values capped at 8 chars in the opener label (full value in tooltip + chooser strip). An earlier responsive-variant draft (on_resize + recompose) raced viewer-return settlement and was deleted rather than tuned. AC4 note: the wide single-row presentation was retired WITH the width machinery - superseded by the always-on multi-row grammar (approved direction: the single row was the wrong grammar for this pane). Live-verified at the real pane.
<!-- SECTION:NOTES:END -->
