---
id: TASK-31961
title: Library media - bulk Analyze re-projects off-page rows that can never repaint
status: To Do
assignee: []
created_date: '2026-09-07 08:27'
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
- [ ] #1 No fetch is issued for an item that is not in the retained page
- [ ] #2 A pin over a multi-page selection counts the service calls
<!-- AC:END -->
