---
id: TASK-31956
title: >-
  Library media - the reviewed decoration loads the whole active review set to
  mark twenty rows
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
labels:
  - library
  - media-ux
  - perf
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
I final review M5: _decorate_library_media_reviewed calls get_active_review_set(), which loads the header and every item row (REVIEW_SET_CAP 500) to decorate at most twenty visible rows, at every viewer-flip sync site (~30). It sits off any per-keystroke loop and is idempotent, so nothing is visibly slow, but the cost is O(active-set size) rather than the 84 us measured against a 75-item set - the design note's premise that ReviewSet.items is already in memory for the banner is false.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decorating a page of rows costs O(visible rows) once the active review set is unchanged
- [ ] #2 Marking an item invalidates whatever is cached so the next decoration is correct
- [ ] #3 A measurement at REVIEW_SET_CAP records the before and after cost
<!-- AC:END -->
