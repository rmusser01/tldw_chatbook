---
id: TASK-31237
title: Reader uses its vertical space
status: To Do
assignee: []
created_date: '2026-09-04 01:50'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2: at 52 terminal rows the Reader's content box ends near row 39 (#library-media-viewer-content max-height: 75vh, _agentic_terminal.tcss) leaving ~10 blank rows below while long documents scroll inside a smaller box; the default-open "Search content…" find input spends 3 more rows on every fresh item (duplicating the Find action); a single-page document still renders two dead "○ Previous ○ Next" pager controls. This is the reading surface of a reading workflow idling a third of its pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Reader content area grows to fill the remaining pane height at tall terminal sizes (no stranded band below the box)
- [ ] #2 The content find input is collapsed until Find is invoked, and Escape re-collapses it
- [ ] #3 The pager row is hidden when there is only one page
- [ ] #4 The task-31222 regression (fixed 18-row cap under an unstyled 1fr band) stays fixed at small sizes
<!-- AC:END -->
