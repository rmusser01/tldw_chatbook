---
id: TASK-31956
title: >-
  Library media - the reviewed decoration loads the whole active review set to
  mark twenty rows
status: Done
assignee: []
created_date: '2026-09-07 08:26'
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
I final review M5: _decorate_library_media_reviewed calls get_active_review_set(), which loads the header and every item row (REVIEW_SET_CAP 500) to decorate at most twenty visible rows, at every viewer-flip sync site (~30). It sits off any per-keystroke loop and is idempotent, so nothing is visibly slow, but the cost is O(active-set size) rather than the 84 us measured against a 75-item set - the design note's premise that ReviewSet.items is already in memory for the banner is false.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decorating a page of rows costs O(visible rows) once the active review set is unchanged
- [x] #2 Marking an item invalidates whatever is cached so the next decoration is correct
- [x] #3 A measurement at REVIEW_SET_CAP records the before and after cost
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pins: an unchanged active set issues one load across N builds; a done mark invalidates. 2. Cache the done-map on the screen keyed on the review-set service's revision, bumped inside the one `_write()` every writer uses. 3. Measure at REVIEW_SET_CAP.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The screen caches the done-map keyed on `ReviewSetService.revision`; every mutating method goes through `_write()` (the one `_db.transaction()` in the service), which bumps it after the commit, so a writer cannot forget to invalidate. Decorating a page with an unchanged set costs a dict comprehension over the visible rows. Measurement (500-item set, 20-row page, 200 runs, median): 0.541 ms → 0.001 ms; the first build after any write pays one 0.537 ms load; the script lived in the scratchpad and is not kept. Storage failure is excluded from the cache (fail-open preserved). Fix round: the revision is captured BEFORE the load so a worker-thread dismiss/undismiss during the load cannot be stamped as included (pinned). Ceiling: the revision is per service instance (one instance, cached on the screen).
<!-- SECTION:NOTES:END -->
