---
id: TASK-32185
title: >-
  Library Notes: folder-tree composition has several unreconciled downstream
  behaviors now reachable
status: In Progress
assignee: []
created_date: '2026-09-09 18:16'
updated_date: '2026-09-11 11:15'
labels:
  - library
  - notes
  - tests
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unblocking task-32175's flat #library-notes-row-0 wait let numerous Database Notes tests run far enough to reach assertions that were never previously exercised (the tests always died earlier at the row-0 NoMatches, before the folder tree existed by default). Several of those assertions now fail against real, reproducible (non-flaky, non-timeout) production behavior gaps in the folder-tree composition/sync path. No production code was touched by task-32175 (test-only); each is a pre-existing defect, independently confirmed by fast, deterministic re-runs (2-17s, not the 30s poll-timeout signature of environmental contention). AC#7-#9 were originally hedged as possible environmental contention; task-32175's review re-ran all three on a quiet machine, one pytest process, no sibling jobs, and all three failed (two with `.library-notes-row never mounted within 30.0s`, one with `Both note rows never mounted`), so the hedge is retired -- they are real.

Reproduction: each named test fails at the specific assertion cited in its AC once task-32175's row-selector fix lands; base (pre-task-32175) never reached these lines. Not fixed there -- task-32175 is test-only. task-32184 tracks a separate, already-filed items_width layout defect from the same unblocking, and task-32201/task-32202 the filter and compact-region reds that fall outside this cluster.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Filtering Database Notes to zero matches shows the #library-notes-empty state (test_library_shell.py::test_library_shell_notes_filtered_empty_keeps_clear_and_source_truth); typing a filter value and pressing Enter (with matches) shows #library-notes-filter-clear (test_library_notes_reader.py::test_database_notes_capability_inventory_and_modes)
- [x] #2 Filtering narrows the rendered tree rows to matches only, excluding non-matching notes (test_library_shell.py::test_library_shell_filtered_delete_refreshes_list_without_ghost)
- [ ] #3 Saving an in-canvas edit refreshes that note's displayed relative age in the tree (test_library_shell.py::test_library_shell_note_save_then_back_refreshes_list_title_and_age)
- [ ] #4 #library-notes-task-return shows correctly across the Files<->Database round trip and compact/wide transitions (test_library_file_notes_workspace.py::test_notes_authority_round_trip_retains_both_workspaces and 3 related tests in the same file)
- [x] #5 Bulk/select mode's #library-note-bulk-status banner displays when expected (test_library_notes_reader.py::test_bulk_mode_keeps_last_note_as_labelled_read_only_preview)
- [x] #6 Pressing Back while a note's detail/keywords load is still gated/pending discards the pending load and returns to list (screen._notes_state.view == "list", _library_note_session.snapshot is None) -- currently the view stays on "editor" with a live snapshot (test_library_shell.py::test_library_note_coordinator_pending_detail_keeps_back_action and ::test_library_note_coordinator_pending_load_keeps_back_and_discards_late_reply)
- [x] #7 A pre-existing Database note emptied to blank content still saves through the real DB (test_library_shell.py::test_library_shell_pre_existing_note_emptied_out_still_saves_in_real_db passes)
- [x] #8 A blank-title save round trip agrees with the row it came from (test_library_shell.py::test_library_shell_blank_title_save_round_trip_agrees_with_the_row passes)
- [x] #9 Entering the Notes list focuses its first row and Up/Down move that focus (test_library_shell.py::test_library_notes_list_focuses_first_row_and_arrow_keys_move_it passes)
- [ ] #10 Re-selecting the already-selected Notes rail row does not re-kick the folder tree's async reload and strand a row press issued right after it (double-click the rail row, then click a note -- nothing happens today; the tests worked around it by dropping their redundant browse-notes press)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run every named node and group them by CAUSE, not by AC number.
2. Fix the one product gap with a RED->GREEN test on the real route; repair the
   stale fixtures at their source; hand off what another wave-3 group owns.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Seven of the ten ACs are green. The cluster had three causes, not nine.

PRODUCT (AC#5): entering select mode beside an open note never re-applied the
work pane's presentation state. Traced by wrapping
`LibraryNotesCanvas.apply_session_state`: it is called ONCE, before the toggle,
with `bulk_read_only=False`, and never again -- only `_apply_library_row_toggle`
did that re-apply. So the editor stayed fully editable (Save, Delete, "Use in
Console", Copy, exports all live; no "Read-only preview" banner) while the list
was in bulk mode. Mechanism: `LibraryNoteWorkPane.sync_state` STORES the
fresh presentation state but skips its rebuild while the reader owns a field
(task-32062), so a same-surface sync that changes only that state was never
painted. The re-apply now lives at the one place every notes sync passes,
`_sync_library_canvas`'s notes branch (`canvas_sync.py`), right after the
work pane's `sync_state` and gated on exactly the skipped rebuild
(`notes_editor_owned`: same surface AND the reader's focus in a field). Every
other sync recomposes, and `_apply_post_compose_state` paints the stored
state itself; re-applying there too would run `apply_session_state` against
children a pending mode-change recompose has not mounted yet (the
`NoMatches` shape task-32467 owns). So every handler that flips the flag
(toggle, select-all, clear, Escape, the filter submit/clear, a sort choice)
gets it, and the first round's three per-handler calls are gone (task-8
review finding 2).
RED `assert bulk_status.display is True` -> GREEN
(`test_bulk_mode_keeps_last_note_as_labelled_read_only_preview`; re-proven
for the consolidation: with the per-handler calls removed and no choke-point
call, that node plus the `escape` and `filter-submit` params below fail at
the ENTRY assertion, 3 failed / 1 passed; with it, 4 passed).
Collateral, fix round 1: `test_library_notes_reader.py` whole file 37 passed;
the 67-node shell regression set (every test the branch touched or that
reaches its changed helpers) 65 passed / 2 failed, both intermittent and
reproduced on the HEAD baseline without this change --
`test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
(task-15741; fails with the re-apply disabled too) and
`test_library_shell_pre_existing_note_emptied_out_still_saves_in_real_db`
(the late-backlinks worker race, task-32467).

The review's "3 of 6 sites" was measured before consolidating. That the three
select-mode EXITS reachable from the UI (Escape, filter submit, filter clear)
do not regress is MECHANISM plus ONE measured route, not a dedicated run:
select mode hides the editor fields, so at exit the reader either no longer
owns a field (the recompose paints the stored state) or still does (the choke
point re-applies); in the RED run (`fix2-ac5-red.txt`, re-apply disabled)
only `[filter-clear]` passes its exit assertions -- `[escape]` and
`[filter-submit]` die at the ENTRY assertion there, so their exits were never
isolated. Under the choke-point design the question is moot: every exit
reaches the same notes sync, and all three are pinned by
`test_leaving_select_mode_beside_an_open_note_restores_its_editor[escape|filter-submit|filter-clear]`.
The sort choice cannot be pressed from select mode at all (the sort control
and its strip compose only in browse mode).
Verified live in a tmux app on a scratch profile: with a note open, pressing
Select repaints the pane as "Read-only preview · Not included in bulk
selection" over "Read-only — this note cannot be changed; your draft is
preserved." Capture:
`wave3-caps/test-health/live-select-mode-readonly-preview.txt`.

TEST-HARNESS STALENESS (AC#1, #2, #6, #7, #8, #9): four independent fixtures
had fallen behind the folder tree.
- The shared `StaticLibraryNotesScopeService` had no
  `search_note_tree_placements`, so the filter silently did nothing (see
  task-32201) -- that alone unblocked AC#1's first node, AC#2 and AC#6.
- `_wait_for_selector` returned the match it found BEFORE its settle pause;
  leaving select mode recomposes the notes canvas, so callers pressed a
  DETACHED Button and the editor never opened (AC#1's second node). It now
  re-queries after the pause.
- The real-DB `NotesScopeService` fixture built the facade with no
  `folder_repository`, which production always wires
  (`app._build_notes_scope_service`); without it every folder seam raises
  FolderCapabilityError and the tree painted "Couldn't load folders · Retry /
  Couldn't load notes · Retry" with no rows at all (AC#7, AC#8).
- `StaticLibraryNotesListScopeService` implemented `list_notes` ALONE, which
  cannot page a tree (AC#9, "Both note rows never mounted"). It now inherits
  the full fake -- but withholds `count_notes` (landing: the inherited
  callable turned the rail's capped-sample label "Notes (100+)" into an exact
  "(105)", RED `test_library_destination_labels_plain_list_notes_as_sample_
  snapshot` on the branch alone, `land-isolation-branch.out`; GREEN with
  `count_notes = None`, `land-snapshot-green.txt`).

NOT DONE, handed off:
- AC#3 (a saved note's row keeps its stale age) -> task-32503. Cause proven:
  `_patch_library_note_list_from_session` patches with
  `baseline.modified_at`, and the save path does not advance that stamp, so
  the age label re-renders unchanged. The row label and the projection age
  that feeds it are owned by the wave-3 list/tree group, which was editing
  that code in parallel -- this task deliberately did not touch it.
- AC#4 (`#library-notes-task-return` across the Files<->Database round trip)
  is unchanged and pre-existing. `Tests/UI/test_library_file_notes_workspace.py`
  run whole gives 8 failed / 152 passed on this branch AND 8 failed / 152
  passed on a detached `origin/dev` worktree, with an IDENTICAL FAILED name
  set (`test_notes_authority_round_trip_retains_both_workspaces`,
  `test_notes_authority_round_trip_resets_only_transient_work_session`,
  `test_notes_authority_switch_restores_visible_focus_and_typing_owner`
  [all three params], `test_wide_files_task_return_restores_database_browse_
  receipt`, `test_high_stakes_file_notes_states_are_legible_in_shipped_themes`
  [both sizes]). None of this task's repairs touch them, and none of the three
  causes above explains them -- they are an authority-round-trip behaviour
  cluster of their own and need their own pass.
- AC#10 (re-selecting the already-selected Notes rail row re-kicks the tree's
  async reload and strands a row press issued right after it) was not
  investigated: it names no node id, the tests that met it worked around it,
  and it is the same rail/tree reload surface as AC#3.

Files: `tldw_chatbook/UI/Library_Modules/canvas_sync.py`,
`tldw_chatbook/UI/Library_Modules/library_notes_controller.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_shell.py`, `Tests/UI/test_destination_shells.py`,
`Tests/UI/test_library_notes_reader.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
