---
id: TASK-2238
title: 'Library: render hub recents as clickable rows (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 20:56'
labels:
  - ux-review
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recents currently render as one dim text line; the canvas void remains the dominant visual. Post-fix re-review P2. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recents render as clickable rows that jump into the item,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (hub presentation; dispatch reuses the existing _open_library_item_by_id route). Steps: 1. RED tests: landing renders one clickable row per source recent (ids library-hub-recent-{notes,media,conversations}) with title text; pressing the notes recent jumps into the note editor via _open_library_item_by_id state (_selected_note_id, editor view, notes row selected); empty library renders no recent rows; dead-code pin for the removed one-line helpers; next-action triad renders ABOVE the recents. 2. library_screen.py: new _hub_recent_items() helper (source_type, record_id, title) built from _local_source_records via _source_record_id/_source_title; compose renders the triad first, then one quiet Button per recent dispatching through a new @on(.library-hub-recent) handler that run_workers _open_library_item_by_id; delete _hub_recents_line/_source_recent_value if unreferenced. 3. Run shell suite + destination/parity + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recents are now one clickable row per source below the next-action triad (the old single dim line is gone). Each row dispatches through _open_library_item_by_id -- the same route as the Search/RAG evidence 'Open' action, guards included -- via a new @on(.library-hub-recent) handler (run_worker since the route awaits flushes). New _hub_recent_items() helper resolves (source_type, record_id, title, label) from _local_source_records via the existing _source_record_id/_source_title; unresolvable ids and empty sources are skipped; empty library renders nothing. Dead code deleted: _hub_recents_line, _source_recent_value (single-consumer helpers). Files: library_screen.py (helper, compose reorder, handler, .library-hub-recent CSS), Tests/UI/test_library_shell.py (3 new tests: rows render + open + triad ordering; empty-library absence; dead-helper pin). Verified: 2 RED->GREEN (empty-library pin passed pre-implementation as the correct contract pin); targeted hub/recents tests 8 passed; live 170x50 capture shows triad on top with three labeled clickable recents rows; full regression sweep (shell + destination + parity) confirmatory in background. Test-seed lesson noted in the test: get_note_detail matches on 'id', so seeds use 'id'. Ruff clean (1 pre-existing F401 untouched). ADR: not required (hub presentation; route reuse). Commit 28df2c84a.
<!-- SECTION:NOTES:END -->
