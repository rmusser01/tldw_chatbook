---
id: TASK-23153
title: One Library notes test passes alone and fails in-file
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - library
priority: low
dependencies: []
---

## Description

`Tests/UI/test_library_shell.py::test_library_note_failed_discard_clears_shortcut_lock_status` fails
in a whole-file run and **passes standalone**. The widget it waits for still exists, and its label
appears in the failure's own visible-text dump — so this is test pollution, not a production defect.
Filed separately from the real regression in the same file so the two are not conflated.

## Acceptance Criteria

- [ ] The polluting interaction is identified (which earlier test leaves the state behind, and what
  state), and named in the implementation notes
- [ ] The test passes both standalone and in a whole-file run, without relying on run order
- [ ] Isolation is fixed at the leaking source rather than by adding cleanup only to the victim, if
  the source is shared

## Evidence

Widget present at `tldw_chatbook/Widgets/Library/library_notes_canvas.py:1084`. Failure is
`#library-note-discard-new never became visible`.
