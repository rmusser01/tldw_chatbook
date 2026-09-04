---
id: TASK-31224
title: 'Media filter-miss recovery - suggest Clear filter, not Import media'
status: Done
assignee: []
created_date: '2026-09-03 22:31'
updated_date: '2026-09-04 00:31'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P2: fresh_zero never checks canvas.query so a filter miss pins 'Import media' as recovery; meanwhile #library-media-filter-clear never rendered live (unbounded Input swallows its row; no shared action class).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A zero-match ACTIVE-query page offers clearing the filter, not importing
- [x] #2 The Clear filter control is visible whenever a query is applied
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2359. fresh_zero gates Import/Show-all on an empty query; the filter Input was width:100% and consumed its row so Clear never rendered - now 1fr share with auto-width Clear (BUNDLED_CSS).
<!-- SECTION:NOTES:END -->
