---
id: TASK-32086
title: >-
  Library media: the reader is not cleared when a 0-result filter hides its
  loaded item
status: To Do
assignee: []
created_date: '2026-09-08 20:48'
labels:
  - library
  - media
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #8 gap in task-32043. task-32043 added `_reset_library_media_reader_to_no_selection` and clears the reader when the loaded item is DELETED (works), and intended to clear it on a 0-result filter too (its AC#1). But critique #8 found live that a filter returning zero results still leaves the reader painting the filtered-out item, with '‹ Back' into an empty list. The filter-path guard in `_sync_library_media_browse_state` (settled page + zero retained rows + still-holding a local reader item) does not fire on the live filter path the assessors took, so AC#1's 0-result-filter half is unmet in practice even though its pin passes. Find why the guard misses the live scenario (e.g. `applied_selection_id`/`retained_items`/settled-state at the check moment, or the reader item's state) and widen it so the reader clears on a settled 0-result filter without over-firing on a page turn or mid-load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Applying a filter that returns zero results while a media item is loaded clears the reader to its no-selection state (does not keep painting the filtered-out item), verified LIVE and by a pin that drives the actual filter-input path (not only the internal `_apply` seam)
- [ ] #2 A filter with results still re-points to a matching item; a page turn and a mid-load do NOT clear the reader (no over-fire)
- [ ] #3 The pin reproduces the critique #8 scenario (reader open, then filter to zero) and would fail against the current guard
<!-- AC:END -->
