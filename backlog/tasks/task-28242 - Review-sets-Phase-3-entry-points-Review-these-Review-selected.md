---
id: TASK-28242
title: 'Review sets - Phase 3: entry points (Review these / Review selected)'
status: Done
assignee: []
created_date: '2026-09-02 22:29'
updated_date: '2026-09-03 03:56'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28240
  - TASK-28241
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create review sets from the media browse result and from a Select-mode selection (design: backlog/docs/design-library-review-sets.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A 'Review these' action on the media list pins the WHOLE filtered browse result (page through to last_page in a worker, de-dupe by id, cap 500 with pin-first-500 + warn on overflow) and opens it active at cursor 0
- [x] #2 A 'Review selected' third Select-mode bulk action (next to Export/Delete selected) pins the selected ids ordered by a deterministic sort-order query (NOT the mounted rows, since RowSelection is unordered and can span pages)
- [x] #3 Both paths land the user in the Reader walking the new set; the filter-query surface is covered by 'Review these' (no separate RAG-search integration)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
3a. Build the ordered pinned-item list from a browse result (pure helper): dedupe by id, cap 500, (id,title) pairs. 3b. 'Review these' action on the media list toolbar: page through the WHOLE filtered result (with_page to last_page) in a worker, build items, create_review_set(origin=browse), land in Reader at item 0. 3c. 'Review selected' Select-mode bulk action next to Export/Delete: order selected ids by a deterministic sort-order query (not mounted rows), create_review_set(origin=selection), land in Reader. 3d. Both land via _select_library_media_reader_row. TDD the pure builder + the ordering; live-verify the buttons.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 3 entry points shipped + LIVE-VERIFIED end-to-end (tmux, seeded 4 media items). 'Review these' (browse toolbar #library-media-review) pages the WHOLE filtered result via search_media(library_summary=True, sort_by, media_types) in a worker, build_pinned_items (dedupe by backing id, cap 500, truncation flag), create_review_set(origin=browse), lands in Reader at item 0 via _open_library_media_viewer. 'Review selected' (select-mode #library-media-review-selected, between Export and danger Delete) orders the selection by search_media(id_allowlist=backing_ids, sort_by) -- deterministic browse order, NOT the unordered RowSelection -- then create(origin=selection) + land. Shared _create_and_open_review_set (empty->notify, truncation->warn). LIVE: clicked Review these -> set created + Reader landed on first item; footer showed '1 of 4 . 0 reviewed'; pressed ] -> advanced to item 2, footer '2 of 4 . 1 reviewed', left item auto-marked done. Cleaned up seeded data after. Pure build_pinned_items + _create_and_open_review_set unit-tested (Tests/UI/test_review_set_walker.py + test_review_set_state.py, 59 pass). NO new logger (branch doesn't touch diagnostic inventory). Files: Widgets/Library/library_media_canvas.py, UI/Screens/library_screen.py, Library/review_set_state.py.
<!-- SECTION:NOTES:END -->
