---
id: TASK-184
title: >-
  Notes list rendering: escape markup in titles, fix stacked toolbar overlap,
  refresh modified age after edit
status: To Do
assignee: []
created_date: '2026-07-12 02:48'
labels:
  - ux
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: a note titled '[draft] Q3 plan [wip]' renders as ' Q3 plan' in the list (bracketed segments consumed as Rich markup - same crash-risk class as the search-history lesson; the editor shows the true title). The sort/Sync/Import note/Export toolbar renders as an overlapped vertical stack eating into the first row. After an in-canvas edit that persists (v2 in DB), the row's modified age and Newest ordering stay stale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Bracketed note titles render verbatim in the list and cannot crash rendering,Notes toolbar renders as a non-overlapping row,List modified-age and ordering reflect an edit without leaving the Notes canvas
<!-- AC:END -->
