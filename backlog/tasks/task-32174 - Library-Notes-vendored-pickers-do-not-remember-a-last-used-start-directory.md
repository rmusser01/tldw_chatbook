---
id: TASK-32174
title: >-
  Library Notes: vendored pickers do not remember a last-used start
  directory
status: To Do
assignee: []
created_date: '2026-09-09 09:12'
updated_date: '2026-09-09 09:12'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - pickers
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32122. task-32122 AC#4 fixed the vendored folder
pickers to resolve a typed-but-unsubmitted **Folder path** before Select,
but left the "remember where I was" gap partially open: Library ingest
already does this caller-side, via `_library_ingest_browse_location`, but
Notes' three folder-picker call sites — Import once, Keep a folder synced,
and Folder files — still always open at the same default location instead
of the directory last browsed in that context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import once's folder picker reopens at its own last-used directory,
  falling back to home when none is recorded yet
- [ ] #2 Keep a folder synced's picker does the same, independently
- [ ] #3 Folder files' picker does the same, independently
- [ ] #4 Tests cover all three
<!-- AC:END -->
