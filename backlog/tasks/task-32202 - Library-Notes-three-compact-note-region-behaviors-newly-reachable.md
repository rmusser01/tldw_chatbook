---
id: TASK-32202
title: >-
  Library Notes: three compact note-region behaviors newly reachable
status: In Progress
assignee: []
created_date: '2026-09-10 07:32'
updated_date: '2026-09-11 10:45'
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
Three Database Notes behaviors in the compact note region fail their own
assertions. They share a cause of *reachability*, not of mechanism: each
test used to die earlier on the flat `#library-notes-row-0` wait that the
folder tree replaced, so none of these assertions had ever run. task-32175
repaired the wait (test-only, no production change) and left them red; each
is a pre-existing production behavior gap, and each is separately verified
red at that task's base `d0ff40842f`.

They are filed together because they are one burn-down of the same region
and were measured in one pass, not because one fix serves all three.

Reproduction, exact assertion text per node:

- `Tests/UI/test_library_shell.py::test_library_note_60x20_loading_allocation_keeps_back_visible` —
  `AssertionError: assert 'Library notes · Library database' in 'Loading note…'`.
  At 60x20, while a note's detail is still loading, the status header is
  overwritten by the load copy instead of keeping the scope line beside it.
- `Tests/UI/test_library_shell.py::test_library_note_compact_surplus_allocation_expands_only_named_owner[navigator-#library-notes-list-fixed_selectors0-expected_fixed_heights0-11-17]` —
  `assert 10 == 11`; and
  `[context-#library-note-context-region-fixed_selectors2-expected_fixed_heights2-11-17]` —
  `assert 12 == 11`. The compact surplus rows are not all given to the named
  owner: the navigator owner is one row short and the context owner one row
  over. The third parametrisation of the same test passes.
- `Tests/UI/test_library_shell.py::test_library_note_local_shortcuts_are_region_scoped_and_flush_guarded` —
  `AssertionError: assert 'create-note' == ''`. A local shortcut stays
  registered outside the region that owns it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60x20, a Database note whose detail is still loading keeps the "Library notes · Library database" scope line in the status header alongside the loading copy, with Back visible (`test_library_note_60x20_loading_allocation_keeps_back_visible` passes)
- [ ] #2 Compact surplus rows go entirely to the named owner region for every owner, not only one (`test_library_note_compact_surplus_allocation_expands_only_named_owner` passes for all three parametrisations)
- [ ] #3 A Notes local shortcut is registered only while its owning region has focus, and is flushed when focus leaves it (`test_library_note_local_shortcuts_are_region_scoped_and_flush_guarded` passes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the three nodes and trace each to its cause.
2. Repair what is a stale expectation; file what is a behaviour decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1 done; AC#2 and AC#3 filed as task-32455 with their proven diagnoses.

AC#1 -- stale expectation, corrected. The failing line was in the shared
`_assert_task8_compact_chrome` helper, not in the test body: it asserts the
authority line always carries "Library notes" and "Next:". Both halves are
superseded. task-32360 (critique #10) DELIBERATELY drops the "Library notes"
prefix below 64 columns because the full line took three rows in a two-row box
and "files." was cut off with no ellipsis -- the source strip directly above
names the authority instead. And task-32063 deliberately gives the loading
state no "Next:" clause, because a "Next:" names a control the reader can press
and "wait for loading" names none. The helper now asserts the authority is
named on the SOURCE STRIP and that the authority line is in one of the two
states it admits (`"Loading note" in text or "Next:" in text` -- task-8 review
finding 5 dropped the filler `text.strip()` line that preceded it).

Newly reachable reds (task-8 review finding 7). Fourteen nodes reach
`_assert_task8_compact_chrome` (five test functions, four of them through
`_assert_task8_rows`); the helper repair took them off the old `'Library
notes' in authority_text` failure and 9 pass, while 5 now fail one assertion
deeper, on the compact ROW allocation -- the same kind of number as AC#2:

- `test_library_note_60x20_navigator_state_allocation[normal]` and
  `[sort-choice]`: `test_library_shell.py:24222` `AssertionError:
  ('#library-notes-list', Region(...))` -- the list region's height is not
  the expected row count;
- `test_library_note_60x20_navigator_state_allocation[filtered-empty]`:
  `NoMatches: No nodes match '#library-notes-empty'`;
- `test_library_note_60x20_editor_state_allocation[delete-confirmation]`:
  `:24222` `('#library-note-title-row', ...)`;
- `test_library_note_60x20_editor_state_allocation[context]`: `:24222`
  `('#library-note-context-region', ...)`.

Stable: the same five names on two consecutive runs of all fourteen nodes
(`wave3-caps/test-health/fixround-60x20-run1.txt` / `-run2.txt`, `5 failed,
9 passed` both). They belong beside AC#2's allocation question in task-32455
and are recorded there by reference rather than filed anew (fix round 1
minted no ids).

AC#2 (`assert 9 == 11` for the navigator owner, `assert 12 == 11` for the
context owner) and AC#3 (`assert 'browse-notes' == ''` -- ctrl+n now selects
the Notes rail row with no Notes region focused, which task-32356 changed from
"open Create" to "create the note") are behaviour questions for the owners of
the compact allocator and of the shortcut grammar, not constants a test-health
pass can re-pin. Diagnoses and current numbers recorded in task-32455.

Files: `Tests/UI/test_library_shell.py`.
<!-- SECTION:NOTES:END -->
