---
id: TASK-31273
title: An explicit open bypasses review-set auto-resume
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 15:09'
labels:
  - library
  - media-ux
  - review-sets
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User ruling at the critique #4 close. Task-31234 made auto-resume open the set's cursor item on every Media entry; that now overrides explicit opens — a deep link, open-by-id from another surface, or Enter on a different row while a set is active pulls the user back to the cursor item (library_screen.py:39683-39692). The ruling: an explicit open wins; plain rail entry with no target still resumes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deep link, open-by-id, or Enter on a row that is not the cursor item opens that item even while a set is active
- [x] #2 The review banner states the off-set state honestly (e.g. `Reviewing paused: <name> — X of M · this item is not in the set`) and ] resumes the walk from the cursor
- [x] #3 Plain rail entry with no explicit target still auto-resumes to the cursor item
- [x] #4 Tests pin all three paths; live-verified
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: banner unit test for an off-set loaded item; walker tests for a cancel helper and a source-level pin that both explicit-open seams call it
2. GREEN: _cancel_pending_review_set_resume() cancels the library_review_set_resume worker group; called from handle_library_media_row (normal mode) and the media branch of _open_library_item_by_id; _active_review_set_banner adds ' · this item is not in the set'
3. Live tmux: set mid-walk, Escape, open an off-set item, ], leave and return via the rail
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Explicit opens (row press, open-by-id / deep link) now cancel any in-flight auto-resume worker before opening, so the entry-time landing can never override what the user chose; the worker's own still-on-the-list gate remains as the second guard. Plain rail entry with no target still auto-resumes (task-31234 ruling). When the loaded item is not in the active set the banner reads 'Reviewing: <name> — X of M · N reviewed · this item is not in the set' and ] resumes from the cursor without marking (existing walker behaviour). Tests: banner suffix, cancel helper, source-level pin of both seams, multiselect row-press fake gains the seam. Live-verified in tmux 235x52.
<!-- SECTION:NOTES:END -->
