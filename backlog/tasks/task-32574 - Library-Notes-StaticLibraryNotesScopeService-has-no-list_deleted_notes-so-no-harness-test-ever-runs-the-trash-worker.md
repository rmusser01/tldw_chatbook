---
id: TASK-32574
title: >-
  Library Notes: StaticLibraryNotesScopeService has no list_deleted_notes, so no
  harness test ever runs the trash worker
status: To Do
assignee: []
created_date: '2026-09-14 22:45'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This is why wave-4's clearest green-pin-over-broken-app got through. task-32539's obvious fix PASSED ITS PIN and did nothing live: every delete starts a Trash-reload worker that ends in a target-less canvas sync, and queue_after_recompose REPLACES, so it evicted 'focus the Undo' and installed a default restore that had captured 'nothing focused'. The race was invisible to every test because the Library harness's notes fake, StaticLibraryNotesScopeService, does not implement list_deleted_notes — so the trash worker cannot run under it at all. Exactly one test in the tree reaches that path, and only because it patches around the gap: Tests/UI/test_library_notes_w4_editor.py:551-570 attaches a list_deleted_notes closure to the service INSTANCE for the duration of that one pin (task-32539's own regression proof, and it says so in its docstring). Every other test that touches Notes deletion still cannot see the worker. The gap is the fake's, not the test's, and one test patching past it is evidence of the gap rather than coverage of it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 StaticLibraryNotesScopeService implements list_deleted_notes with the same contract as the real service (paging, ordering, the 20-row page)
- [ ] #2 At least one existing Library harness test drives a delete through the Trash-reload worker end to end
- [ ] #3 A grep records every other method the real notes scope service exposes that the fake does not, so the next hole is known rather than discovered by a defect
<!-- AC:END -->
