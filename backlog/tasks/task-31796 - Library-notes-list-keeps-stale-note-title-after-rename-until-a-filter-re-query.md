---
id: TASK-31796
title: Library notes list keeps stale note title after rename until a filter re-query
status: Done
assignee: []
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 15:24'
labels:
  - bug
  - ui
  - library
  - notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Create a blank note (autosaves as 'Untitled'), set its title in the editor, wait for 'Saved', Esc back to the list: the Unfiled entry still reads 'Untitled', and stays stale even after reopening the note. Only typing a filter query and pressing Enter re-queries and shows the real title. DB row confirmed correct throughout, so this is purely the list widget not refreshing on save. With several new notes, every one shows as 'Untitled' in the primary nav surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Renaming a note updates its row in the notes list on save (or at latest on returning to the list), without requiring a filter re-query.
- [x] #2 A regression test covers title propagation from editor save to the list row.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: create blank note, set title + body, wait Saved, Esc to list — confirm 'Unfiled' row stays 'Untitled' while DB row is correct.\n2. Find the render source: the tree projection renders from cached branch slices (and filter window), NOT the flat records the save patch touched.\n3. Add pure retitle helpers (patch_notes_tree_branches_title, patch_notes_filter_state_title) and wire them into _patch_library_note_list_from_session.\n4. Regression tests: pure helpers + screen-wiring test proving save updates the tree row without a filter re-query.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the Database Notes list renders placement rows from the cached tree branch slices (_library_notes_tree_branches), and while a filter is active from the filter window (_library_notes_tree_filter_state) -- NOT from the flat source records. The save-time patch (_patch_library_note_list_from_session) only updated the flat records (_local_source_records['notes'] + _library_notes_filter_records) via patch_note_records_after_save, so a renamed note's tree/list row kept the pre-rename title (e.g. 'Untitled') until typing in the filter box re-queried the DB and rebuilt the slices.

Fix: added two pure helpers -- retitle_note_placements + patch_notes_tree_branches_title (library_notes_tree_paging.py) and patch_notes_filter_state_title (library_notes_tree_state.py) -- that rebuild only the matching immutable NotePlacementRecord's note mapping (title, and last_modified for parity). _patch_library_note_list_from_session now also patches the branch slices and the active filter window. Because both save paths (autosave SAVED + flush) funnel through this helper and the Esc-back-to-list path re-renders via _sync_library_canvas, the fresh title shows on save and on return without a filter re-query.

Reproduced live (tmux) before the fix (list stayed 'Untitled' while the DB row was correct) and confirmed fixed after (a freshly renamed note showed its real title on Esc back to the list, no re-query).

Tests: Tests/UI/test_library_notes_rename_propagation_t31796.py -- pure-helper propagation tests (placement retitle -> build_paged/build_filtered projection row label) plus a screen-wiring test calling _patch_library_note_list_from_session; the wiring test is mutation-verified (fails without the branch patch).

Files: tldw_chatbook/Library/library_notes_tree_paging.py; tldw_chatbook/Library/library_notes_tree_state.py; tldw_chatbook/UI/Screens/library_screen.py; Tests/UI/test_library_notes_rename_propagation_t31796.py; Docs/User_Guide/library/notes.md (verified-against stamp).
<!-- SECTION:NOTES:END -->
