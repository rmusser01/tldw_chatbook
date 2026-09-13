---
id: TASK-32201
title: >-
  Library Notes: the notes filter never reaches the search service
status: To Do
assignee: []
created_date: '2026-09-10 07:30'
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
- [ ] #1 Typing a filter value into `#library-notes-filter` and pressing Enter calls the notes scope service's `search_notes` and narrows the rendered rows to the matches (`test_library_note_keyboard_capability_matrix[filter-terminal_size0]` and `[filter-terminal_size1]` pass)
- [ ] #2 A filter submitted on a wide Database Notes list settles into `_notes_state.filter` and `_notes_state.filter_records`, and survives the note editor Back round trip (`test_library_note_editor_back_restores_exact_wide_browse_context` passes)
- [ ] #3 Continue from the Library landing restores the Database Notes filter that was active before admission (`test_library_landing_continue_reapplies_database_notes_scope_after_admission` passes)
<!-- AC:END -->
