---
id: TASK-32175
title: >-
  Library Notes: retire or repair tests that assume the flat list row the folder
  tree replaced
status: Done
assignee: []
created_date: '2026-09-09 09:13'
updated_date: '2026-09-09 18:39'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the final whole-branch review of the wave, most directly by task-32128's
own lessons-file entry ("A widget id that becomes conditional must be
reconciled across all of Tests/"). Because Database Notes now composes a
folder tree by default (Agent_Lessons is always seeded), a flat
`#library-notes-row-0` row never mounts, and every test that waits on it
fails before its real assertion runs. That includes the reader suite's 13
pre-existing reds — among them
`test_database_notes_capability_inventory_and_modes` — five shell/workspace
tests, and the `filter_sort` capability node. These were red at the wave's
base too, but the wave's own reconciliation pass (task-32128) only chased
the failures its own change surfaced, not this wider pre-existing set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every test currently asserting against `#library-notes-row-0` (or
  an equivalent flat-list assumption) either drives the tree row
  (`.library-notes-tree-note-row`) instead, or is retired with a written
  reason
- [x] #2 The affected files' failing-test-name sets shrink accordingly,
  recorded in the task's implementation notes
- [x] #3 No new test introduces a `#library-notes-row-N` assumption
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Census: grep -rn "library-notes-row-[0-9]" Tests/ to enumerate every hit.
2. Root-cause fix at the shared seams: `_open_note_editor` and
   `_task10_open_note_editor_with_keyboard`/`_task10_activate_with_keyboard`
   callers in test_library_shell.py drive ~78+ tests; swap their flat
   `#library-notes-row-0` wait+press for the mode-agnostic
   `.library-notes-row` class query (matches both the flat row and the
   tree's note row -- both carry that class).
3. Fix every remaining direct occurrence per file (test_library_shell.py,
   test_library_notes_reader.py, test_library_file_notes_workspace.py,
   test_library_entry_compose_once.py): row-0 -> first `.library-notes-row`
   match; row-1/second-row assumptions -> select by note identity (the
   tree orders by title, not the flat list's Newest/insertion order).
4. Drop `#library-notes-sort`-only assertions that assume flat-list mode
   (task-32128 removed Sort from tree composition) where they ride along
   in a row-N test; leave already-tree-aware Sort tests untouched.
5. test_wide_files_task_return_restores_database_browse_receipt: drop the
   Sort press (tree already orders by title) rather than retire -- repair,
   not retire, since equivalent rail+list scroll-restore coverage isn't
   duplicated elsewhere.
6. Leave test_library_notes_wave_list.py's flat-fallback pin
   (test_flat_rows_render_the_same_single_line_age) and out-of-scope Sort
   reds (not row-N) untouched.
7. Run base (d0ff40842f, throwaway worktree) vs head per file; record the
   failing-name-set census in the task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Census (grep -rn "library-notes-row-[0-9]" Tests/) found every hit repairable; none needed retiring. Root cause fixed at the shared seam: _open_note_editor (78+ callers) and _task10_open_note_editor_with_keyboard (~10 callers) in test_library_shell.py now wait on the mode-agnostic .library-notes-row class (both the flat row and every tree note row carry it, confirmed in library_notes_canvas.py) instead of the flat #library-notes-row-0 id. Every remaining direct occurrence across test_library_shell.py, test_library_notes_reader.py, test_library_file_notes_workspace.py and test_library_entry_compose_once.py was repaired the same way; row-1/"second row" assumptions were switched to select by note identity (candidate.note_id) since the folder tree orders rows by title (task-32128), not the flat list's Newest/insertion order two tests leaned on positionally. test_library_notes_wave_list.py's flat-fallback pin (test_flat_rows_render_the_same_single_line_age, which drives tree_projection=None directly) was left untouched -- it's a still-legitimate code path, not a stale assumption.

A related staleness class was repaired alongside: #library-notes-sort-only assertions riding along in a row-N test (Sort is flat-list-only since task-32128, 3 sites + the filter_sort->filter capability-matrix param), and two tests still asserting the row label as two splitlines() (title, then age) -- stale since task-32137 unified the label onto one "title * age" line, fixed to split on " * " instead. test_wide_files_task_return_restores_database_browse_receipt was repaired (Sort press dropped; the tree's own order is already title-based) rather than retired, per the task's own plan.

Two genuine bugs were found in my own repair and fixed before landing: (1) test_library_notes_reader.py had 6 tests that pressed browse-notes themselves and then called _open_note_editor (which presses it again) -- harmless under the flat list, but the redundant press re-kicks the tree's async reload (task-32126) and strands the row press that follows; fixed by removing the redundant press (reproduced the 30s timeout, then confirmed the fix drops it to 1-3s). (2) test_library_note_breakpoint_round_trips_restore_every_region_focus_role's repair had hardcoded the flat list's "note-row:<id>" semantic role; the tree's real role is "note-placement:<placement_id>" (library_screen.py's own focus-identity capture) -- fixed to read the row's placement_id instead of hardcoding the old string.

Test summary (base vs head, full detail in the wave's task-4-report.md):
- test_library_notes_reader.py (whole file): 13 failed/21 passed -> 3 failed/31 passed. The 3 residual reds are proven unrelated (different assertion/location, no prod change made).
- test_library_entry_compose_once.py (2 touched): 2/2 red -> 2/2 green.
- test_library_honesty_accessibility.py (1 touched): 1/1 red -> 1/1 green.
- test_library_file_notes_workspace.py (4 touched): 4/4 red -> 4/4 still red for one shared unrelated reason (proof: the failure moved from the row-0/Sort NoMatches to a different assertion several steps further in); filed task-32185.
- test_library_shell.py (explicit node ids only; base==this branch's own pre-edit HEAD, so no throwaway worktree needed for the comparison itself, though one was used and removed to spot-check two batches): the great majority of ~40 targeted tests/nodes went from a clean row-N/Sort NoMatches red to green, confirmed via isolated re-runs.

Two follow-up tasks filed from this verification pass (test-only task -- not fixed here): task-32184 (adaptive reader items_width mismatch, 12/24 parametrized cases of test_library_production_width_matrix_custom_preferences, reproduced identically every run) and task-32185 (a cluster of Database Notes filter/age-refresh/task-return/pending-load-cancel behaviors newly reachable through this fix, never exercised before since the affected tests always died first at the row wait this task repairs; also carries a lower-confidence note on 3 test_library_shell.py cases that failed 3/3 under this session's continuous 30-45 concurrent-pytest contention from sibling wave-2 groups, twice with the app's own "Couldn't load folders/notes" error visible -- never confirmed clean on a quiet machine this session).

No production code was touched; diff is Tests/ only (test_library_shell.py, test_library_notes_reader.py, test_library_file_notes_workspace.py, test_library_entry_compose_once.py).
<!-- SECTION:NOTES:END -->
