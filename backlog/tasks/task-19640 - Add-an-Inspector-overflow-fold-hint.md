---
id: TASK-19640
title: Add an Inspector overflow fold hint
status: To Do
assignee: []
created_date: '2026-08-20 07:10'
labels:
  - console
  - ux
dependencies:
  - TASK-18912
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make hidden Inspector content discoverable by showing the Console product's standard scroll-below hint only while more content exists below the visible fold.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 When the Inspector body has content below its visible viewport and is not scrolled to the end, a visible `▼ more — scroll` hint communicates that more content exists.
- [ ] #2 The hint is absent when the Inspector does not overflow or is scrolled to the end, and it reappears after the user scrolls upward while hidden content remains below.
- [ ] #3 The hint does not cover, reorder, duplicate, or change the semantics of Sources, run state, Tools, Approvals, Artifacts, live-work sources, or Session Settings.
- [ ] #4 Production-CSS Textual compositor tests cover representative 235x52, 120x30, and 80x24 states, including overflow and no-overflow cases.
- [ ] #5 Keyboard scrolling, pointer scrolling, focus order, rail badges, collapse/reopen behavior, and existing product-standard fold-hint consumers do not regress.
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-18915, colliding with the older
"Add-an-Inspector-overflow-fold-hint" task that arrived on dev first.
Per the owner rule decided 2026-08-21 in TASK-19601 (**older id keeps it;
the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-19640. Citations to TASK-18915
in already-merged commit messages, ADRs, or code comments written before
2026-08-21 refer to THIS task; the other TASK-18915 holder is the
older arrival and keeps the id.
