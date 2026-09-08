---
id: TASK-32086
title: >-
  Library media: the reader is not cleared when a 0-result filter hides its
  loaded item
status: Done
assignee: []
created_date: '2026-09-08 20:48'
updated_date: '2026-09-08 22:01'
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
- [x] #1 Applying a filter that returns zero results while a media item is loaded clears the reader to its no-selection state (does not keep painting the filtered-out item), verified LIVE and by a pin that drives the actual filter-input path (not only the internal `_apply` seam)
- [x] #2 A filter with results still re-points to a matching item; a page turn and a mid-load do NOT clear the reader (no over-fire)
- [x] #3 The pin reproduces the critique #8 scenario (reader open, then filter to zero) and would fail against the current guard
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
task-32043's 0-result-filter reset cleared the reader SESSION (`loaded_id`→None, so its state-pin passed) but the FILTER path re-renders only the Items pane via `_sync_library_canvas('media')` and never refreshes the sibling mounted `LibraryMediaViewer`, so it kept painting the filtered-out item; the DELETE path escaped this via a whole-screen `refresh(recompose=True)`. Fix: mirror the delete path on the settled 0-result reset branch in `_sync_library_media_browse_state` — `refresh(recompose=True)` + `is_mounted` guard + `call_after_refresh(self._focus_library_control, focus_identity)` + `return` (after the reset `loaded_id` is None so the branch can't re-fire — no recompose loop). Over-fire guards intact (page turn has rows; mid-load short-circuits; filter-with-results re-points via applied_selection_id). New pin `test_zero_result_filter_repaints_the_mounted_reader_placeholder` drives the REAL filter-Input path (not the internal seam the 32043 pin used) and asserts the MOUNTED widget (`#library-media-reader-empty` present, `#library-media-viewer-title` gone); red first. Live-verified at 235x52. LESSON (lessons-testing-evidence): clearing a state object is not clearing the UI — a sibling mounted widget rendering from that state must be recomposed, or a state-only pin passes while the screen paints the stale item. Files: library_screen.py, Tests/UI/test_library_media_render_fixes.py (or the reader pin file).
<!-- SECTION:NOTES:END -->
