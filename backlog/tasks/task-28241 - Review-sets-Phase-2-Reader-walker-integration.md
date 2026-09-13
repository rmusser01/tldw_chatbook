---
id: TASK-28241
title: 'Review sets - Phase 2: Reader walker integration'
status: Done
assignee: []
created_date: '2026-09-02 22:28'
updated_date: '2026-09-03 01:53'
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
- [x] #1 When a set is active, ] / [ advance the set cursor over the pinned list (whole set, page-independent) via the existing _select_library_media_reader_row actuator; with no active set, ] / [ keep task-28005's browse-row behavior (supersede, not replace)
- [x] #2 Forward advance auto-marks the item left behind done; Prev does not un-mark; the last item and picker jumps do not auto-mark; an explicit toggle key sets/clears a mark
- [x] #3 A Reader progress readout shows 'X of M (reviewed N)' over live items, an explicit all-reviewed state on completion, and Escape keeps the set active while a distinct Exit-review deactivates it
- [x] #4 SPLIT to task-28245: the set STATE resumes automatically (active flag + cursor persist), so ]/[ walk from the saved cursor on relaunch; only the convenience of AUTO-LOADING the cursor item on entry (startup-timing sensitive) is deferred there
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
2a. Wire ReviewSetService into the screen (LibraryCollectionsDB instance) + an is_live predicate resolving local:media:N against the Media DB. 2b. Branch _library_media_adjacent_row on an active set: cursor-walk the pinned list via _select_library_media_reader_row instead of mounted rows. 2c. Auto-mark-done on forward ] advance; toggle key; Prev/jumps don't auto-mark. 2d. Reader progress readout + all-reviewed state. 2e. Exit-review action (deactivate) distinct from Escape (keep active). 2f. Resume active set at cursor on launch. TDD each seam.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 2 walker shipped (AC1-3; AC4 auto-load-on-entry split to task-28245). AC1: _select_library_media_adjacent_item branches to _walk_active_review_set when a set is active -- walks the pinned cursor over the WHOLE set via _select_library_media_reader_row (pure plan_walk, single live snapshot); with no set, falls through to 28005 browse rows. AC2: forward ] auto-marks the item left done (last-item = completion gesture), Prev never marks; explicit 'm' toggles the loaded item's mark (action_library_media_toggle_reviewed). AC3: footer shows progress ('X of M · N reviewed' / 'All N reviewed') + relabels ]/[ 'in set' + 'R' exit-review (deactivate, stays resumable) distinct from Escape (keeps active). Liveness = ONE batched Media-DB query (_review_set_live_ids: id IN(...) AND deleted=0 AND is_trash=0). New bindings R/m gated in check_action (binding audit passes; only pre-existing focus_previous_workbench_pane fails). Added service.deactivate_active(). Tests: Tests/UI/test_review_set_walker.py (12) + review_set_state plan_walk/format (7). Diagnostic inventory regen'd (one logger.debug). Files: UI/Screens/library_screen.py, Library/review_set_state.py, Library/review_set_service.py.
<!-- SECTION:NOTES:END -->
