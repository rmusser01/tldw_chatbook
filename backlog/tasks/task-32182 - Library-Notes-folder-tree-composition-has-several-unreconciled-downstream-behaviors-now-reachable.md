---
id: TASK-32182
title: >-
  Library Notes: folder-tree composition has several unreconciled downstream
  behaviors now reachable
status: To Do
assignee: []
created_date: '2026-09-09 18:16'
updated_date: '2026-09-09 18:38'
labels:
  - library
  - notes
  - tests
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unblocking task-32175's flat #library-notes-row-0 wait let numerous Database Notes tests run far enough to reach assertions that were never previously exercised (the tests always died earlier at the row-0 NoMatches, before the folder tree existed by default). Several of those assertions now fail against real, reproducible (non-flaky, non-timeout) production behavior gaps in the folder-tree composition/sync path. No production code was touched by task-32175 (test-only); each is a pre-existing defect, independently confirmed by fast, deterministic re-runs (2-17s, not the 30s poll-timeout signature of environmental contention).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Filtering Database Notes to zero matches shows the #library-notes-empty state (test_library_shell.py::test_library_shell_notes_filtered_empty_keeps_clear_and_source_truth); typing a filter value and pressing Enter (with matches) shows #library-notes-filter-clear (test_library_notes_reader.py::test_database_notes_capability_inventory_and_modes)
- [ ] #2 Filtering narrows the rendered tree rows to matches only, excluding non-matching notes (test_library_shell.py::test_library_shell_filtered_delete_refreshes_list_without_ghost)
- [ ] #3 Saving an in-canvas edit refreshes that note's displayed relative age in the tree (test_library_shell.py::test_library_shell_note_save_then_back_refreshes_list_title_age_and_order)
- [ ] #4 #library-notes-task-return shows correctly across the Files<->Database round trip and compact/wide transitions (test_library_file_notes_workspace.py::test_notes_authority_round_trip_retains_both_workspaces and 3 related tests in the same file)
- [ ] #5 Bulk/select mode's #library-note-bulk-status banner displays when expected (test_library_notes_reader.py::test_bulk_mode_keeps_last_note_as_labelled_read_only_preview)
- [ ] #6 Pressing Back while a note's detail/keywords load is still gated/pending discards the pending load and returns to list (screen._notes_state.view == "list", _library_note_session.snapshot is None) -- currently the view stays on "editor" with a live snapshot (test_library_shell.py::test_library_note_coordinator_pending_detail_keeps_back_action and ::test_library_note_coordinator_pending_load_keeps_back_and_discards_late_reply)
- [ ] #7 test_library_shell.py::test_library_shell_pre_existing_note_emptied_out_still_saves_in_real_db, ::test_library_shell_blank_title_save_round_trip_agrees_with_the_row and ::test_library_notes_list_focuses_first_row_and_arrow_keys_move_it pass on a quiet machine (reproduced 3x under this session's heavy concurrent-pytest load with the app's own "Couldn't load folders/notes - Retry" error text visible, or a tree-data wait timeout; lower-confidence than the other findings here since a quiet re-run was not available this session -- re-verify before trusting as a real defect)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Discovered while verifying task-32175. Reproduction: each named test fails at the specific assertion cited in its AC once task-32175's row-selector fix lands; base (pre-task-32175) never reached these lines. Not fixed here -- task-32175 is test-only. task-32181 tracks a separate, already-filed items_width layout defect from the same unblocking.
<!-- SECTION:NOTES:END -->
