---
id: TASK-32088
title: Notes-sync cutover guard is permanently vacuous and not naively retargetable
status: To Do
assignee: []
created_date: '2026-09-08 15:03'
labels:
  - notes
  - sync
  - test-debt
  - library-decomposition
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_timer_worker_or_mutating_handler` no longer guards the thing it was written to catch, and cannot be repaired by a mechanical repoint. It was written for the TASK-19011 atomic cutover to prove no legacy notes auto-sync timer survives on the Library screen, by AST-walking `library_screen.py` for any `ast.Attribute` whose name starts with `_library_notes_auto_sync_timer`. The wave-8 notes extraction folded that field into `LibraryNotesState`, so its only spelling anywhere is now `self._notes_state.auto_sync_timer`, whose `ast.Attribute.attr` is `auto_sync_timer` -- which does not start with the guard's prefix. The census can therefore never see the field again, in any file, at any revision: the guard passes vacuously and will keep passing whatever anyone does to the timer. This was disclosed at every step of the extraction rather than banked (the state module's docstring, the move commit message, and all three wave-8 task reports say so) and deliberately left for the team that owns the cutover, because the right fix has a product component the extraction series had no standing to decide. The vestigial field itself still exists on the state dataclass with zero readers and zero writers in production.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The guard either fails when a legacy notes auto-sync timer is present on any receiver, or is deliberately retired with the reason recorded -- decided, not left passing by accident
- [ ] #2 The vestigial `auto_sync_timer` field is either deleted from `LibraryNotesState` or documented as intentionally retained with its reason
- [ ] #3 If the census is retargeted, it does not fail on `_library_notes_sync_controller` (the WIRING binding accessor), which is the one false positive a naive prefix retarget produces
- [ ] #4 A mutation proves whatever guard remains actually fires: re-introduce a legacy-shaped timer and show the guard goes RED
<!-- AC:END -->
