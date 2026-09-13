---
id: TASK-32201
title: >-
  Library Notes: the notes filter never reached search_note_tree_placements
  (the shared fake lacked the seam)
status: Done
assignee: []
created_date: '2026-09-10 07:30'
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
Typing into the Database Notes filter and pressing Enter never calls the
notes scope service's `search_notes`: the filter text stays in the Input,
`_notes_state.filter_records` stays `None`, and the list is never narrowed.
Three test nodes reach that assertion and fail on it; a fourth, related
failure sits on the same filter surface (Continue after landing admission
does not restore the Database Notes filter) and is filed here because it is
the same feature, though its root cause is not yet proven to be the same
one.

Discovered while reviewing task-32175 (test-only: the flat-list row repair).
Two of these nodes are red identically at that task's base — the repair did
not cause them and did not repair them. The third only became reachable
through the repair, in the same way task-32184 and task-32185 did: the test
used to die earlier on a `#library-notes-sort` press that the folder tree no
longer composes.

Reproduction, exact assertion text per node:

- `Tests/UI/test_library_shell.py::test_library_note_keyboard_capability_matrix[filter-terminal_size0]`
  and `[filter-terminal_size1]` —
  `AssertionError: Keyboard filter submit never reached search_notes.`
  (Tab to `#library-notes-filter`, type `retro`, Enter; `app.notes_scope_service.search_calls`
  stays empty.) Red identically at task-32175's base `d0ff40842f`.
- `Tests/UI/test_library_shell.py::test_library_note_editor_back_restores_exact_wide_browse_context` —
  `AssertionError: Filtered Notes scope did not settle: value='scope',
  focused=LibraryRailSearchInput(id='library-notes-filter', classes='-valid'),
  filter='scope', calls=[].` Newly reachable: at base this test died in ~5 s
  on `NoMatches: No nodes match '#library-notes-sort'`.
- `Tests/UI/test_library_shell.py::test_library_landing_continue_reapplies_database_notes_scope_after_admission` —
  `AssertionError: Continue did not restore the Database Notes filter.`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing a filter value into `#library-notes-filter` and pressing Enter calls the notes scope service's `search_note_tree_placements` (the folder tree pages placements; `search_notes` is only the shared fake's matching rule behind that seam) and narrows the rendered rows to the matches (`test_library_note_keyboard_capability_matrix[filter-terminal_size0]` and `[filter-terminal_size1]` pass)
- [x] #2 A filter submitted on a wide Database Notes list settles into `_notes_state.filter` and `_notes_state.filter_records`, and survives the note editor Back round trip (`test_library_note_editor_back_restores_exact_wide_browse_context` passes)
- [x] #3 Continue from the Library landing restores the Database Notes filter that was active before admission (`test_library_landing_continue_reapplies_database_notes_scope_after_admission` passes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the real submit route from `Input.Submitted` on `#library-notes-filter`.
2. Decide product vs fixture on evidence, not on the task's premise.
3. Repair at the cause, with the three named nodes as the RED->GREEN pin.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CORRECTION to this task's premise, with evidence. The filter is NOT missing
from the product: `handle_library_notes_filter` -> `_run_library_notes_filter`
submits through `NotesScopeService.search_note_tree_placements` (the folder tree
pages PLACEMENTS, not notes -- `library_screen.py`, the `method =
getattr(service, "search_note_tree_placements", None)` guard). The shared fake
`StaticLibraryNotesScopeService` (`Tests/UI/test_destination_shells.py`) carried
only `search_notes`, so that guard returned before calling anything: the filter
silently did nothing in EVERY Library suite, which is what the three nodes were
reporting.

Fix: the fake implements `search_note_tree_placements`, delegating its matching
to its own `search_notes` so the one substring rule (and the `search_calls`
record the filter tests assert on) stays in one place, and returning a
`NotePlacementPage` of unfiled placements exactly as `page_note_placements`
does.

RED -> GREEN per node:
- `test_library_note_keyboard_capability_matrix[filter-terminal_size0]` and
  `[filter-terminal_size1]`: RED `AssertionError: Keyboard filter submit never
  reached search_notes.` -> GREEN.
- `test_library_note_editor_back_restores_exact_wide_browse_context`: RED
  `Filtered Notes scope did not settle: ... calls=[]` -> reached the next
  assertion, `assert 20 == 32` on `filter_records`, which is the paged truth
  (`LIBRARY_NOTES_TREE_PAGE_SIZE` = 20). Re-pinned to the window AND the page
  total (`tree_filter_state.total == 32`), which says strictly more than the
  old "all 32 records" line -> GREEN.
- `test_library_landing_continue_reapplies_database_notes_scope_after_admission`:
  RED `Continue did not restore the Database Notes filter.` -> GREEN. Same
  cause, as the task suspected but could not prove.

Sibling filter tests re-run green in the same pass
(`test_library_shell_notes_filter_queries_search_seam`,
`test_library_shell_notes_filter_clears_before_stale_response_lands`).

Not changed, deliberately: `_run_library_notes_filter`'s `callable(...)` guard
still returns silently when a service lacks the seam. In production the facade
always has it; making that a hard failure is a separate call.

Task-8 review (finding 6): the title and AC#1 were reworded to name the real
seam before closing; the Description above is the filing-time premise, kept
as history.

Files: `Tests/UI/test_destination_shells.py`, `Tests/UI/test_library_shell.py`.
<!-- SECTION:NOTES:END -->
