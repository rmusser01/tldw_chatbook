---
id: TASK-28241
title: 'Review sets - Phase 2: Reader walker integration'
status: To Do
assignee: []
created_date: '2026-09-02 22:28'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28240
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the media Reader walk an active review set instead of the current browse page (design: backlog/docs/design-library-review-sets.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a set is active, ] / [ advance the set cursor over the pinned list (whole set, page-independent) via the existing _select_library_media_reader_row actuator; with no active set, ] / [ keep task-28005's browse-row behavior (supersede, not replace)
- [ ] #2 Forward advance auto-marks the item left behind done; Prev does not un-mark; the last item and picker jumps do not auto-mark; an explicit toggle key sets/clears a mark
- [ ] #3 A Reader progress readout shows 'X of M (reviewed N)' over live items, an explicit all-reviewed state on completion, and Escape keeps the set active while a distinct Exit-review deactivates it
- [ ] #4 On launch the active set re-activates and loads at its cursor; per-item scroll resume (ReadingProgress) still works
<!-- AC:END -->
