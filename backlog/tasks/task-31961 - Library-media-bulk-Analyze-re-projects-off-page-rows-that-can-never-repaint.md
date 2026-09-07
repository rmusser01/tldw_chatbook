---
id: TASK-31961
title: Library media - bulk Analyze re-projects off-page rows that can never repaint
status: Done
assignee: []
created_date: '2026-09-07 08:27'
updated_date: '2026-09-07 20:20'
labels:
  - library
  - media-ux
  - perf
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J final review M2 and Task 3 review M1: _reproject_library_media_analysis_row guards only on 'not controller.retained_items', so a bulk Analyze over a multi-page selection issues one id-scoped SELECT per off-page item whose row is not retained and therefore cannot repaint. Lifting note_analysis_state's membership test above the fetch skips them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No fetch is issued for an item that is not in the retained page
- [x] #2 A pin over a multi-page selection counts the service calls
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin over a multi-page selection counting the scope service's calls. 2. Lift `note_analysis_state`'s membership test above the id-scoped fetch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_reproject_library_media_analysis_row` now tests membership in `controller.retained_items` (the same test `note_analysis_state` runs) BEFORE the id-scoped `search_media`, so a bulk Analyze over a multi-page selection issues no fetch for items whose row is not retained; on-page items still re-project (pinned end to end). `has_analysis` stays a SQL projection; the seven-key contract is unchanged.
<!-- SECTION:NOTES:END -->
