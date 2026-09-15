---
id: TASK-32455
title: >-
  Library Notes: compact surplus rows and ctrl+n region scoping are still red
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The two halves of task-32202 that are NOT the status-header one (AC#1, fixed
there). Both reproduce at dev ff2dc03145 with the task-32201 fixture repair
applied; both are behaviour questions their owners have to answer, not stale
constants a test-health pass can re-pin.

1. `test_library_note_compact_surplus_allocation_expands_only_named_owner`
   -- `[navigator-...]` fails `assert 9 == 11` and `[context-...]`
   `assert 12 == 11`; the third parametrisation passes. The compact surplus
   rows are not all given to the named owner: the navigator owner is two
   rows short and the context owner one row over. (The navigator gap was 10
   vs 11 when task-32202 was filed and is 9 vs 11 now, so the number is not
   stable either.)

2. `test_library_note_local_shortcuts_are_region_scoped_and_flush_guarded`
   -- fails at its FIRST assertion, `assert screen._library_selected_row_id
   == selected_before`, with `'browse-notes' == ''`: pressing ctrl+n with no
   Notes region focused now selects the Notes rail row. task-32356 changed
   ctrl+n from "open Create" to "create the note", and the test's premise is
   that a Notes LOCAL shortcut must not fire while the Notes region does not
   own focus. Either task-32356 overreached (the binding should still be
   region-scoped) or this assertion is superseded -- that call belongs to
   whoever owns the shortcut grammar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Compact surplus rows go entirely to the named owner region for every owner (all three parametrisations of `test_library_note_compact_surplus_allocation_expands_only_named_owner` pass)
- [ ] #2 The ctrl+n scope question is decided and written down: either the binding is region-scoped again, or the assertion is corrected with task-32356's decision cited (`test_library_note_local_shortcuts_are_region_scoped_and_flush_guarded` passes either way)
<!-- AC:END -->
