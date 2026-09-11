---
id: TASK-32358
title: >-
  Library Notes editor: 'Draft — not saved yet' shows while the note is already
  listed and counted
status: Done
assignee: []
created_date: '2026-09-11 06:18'
updated_date: '2026-09-11 08:31'
labels:
  - library
  - notes
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Creating a first note shows 'Draft — not saved yet' while the same note already appears in the list and the rail count says (1) (A cap 11); the editor chip later says 'Saved · changes save automatically'. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The draft chip and the list/count agree at every moment of a note's life
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: the chip and the list agree the moment a blank note is created.
2. _library_note_status_line returns 'Empty note — type to keep it' for a pending blank (the row IS committed; task-32133 AC#2's intent kept).
3. Update the two pins and the guide lines that quoted 'Draft — not saved yet'.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A fresh blank note's editor chip read 'Draft — not saved yet' while the same note already stood in the list with the rail count beside it saying so. The row IS committed at that point -- that is exactly what `_library_note_pending_blank_gc_id` exists to clean up (LIB-14) -- so the chip was the only thing on screen that was false. It now reads 'Empty note — type to keep it', which is true and is also the thing the reader needs to know: the note will not survive being abandoned empty.

This refines task-32133 AC#2's wording, not its rule -- an untouched blank note still must never claim 'Saved'. The two pins that quoted the old string (test_library_notes_wave_editor_keys.py) keep their assertions and their reasons, with the new copy; the guide's New note view section now explains why the chip says what it says.

Live-verified on a genuinely starter profile at 100x30: ctrl+n -> chip 'Empty note — type to keep it' with 'Untitled · now' in the list and 'Notes (1)' in the rail; typing -> 'Unsaved changes' -> 'Saved'.

Modified: tldw_chatbook/UI/Library_Modules/library_notes_controller.py (`_library_note_status_line`), Docs/User_Guide/library/notes.md. Tests: Tests/UI/test_library_crit10_notes_details.py (chip and list agree at the create moment), test_library_notes_wave_editor_keys.py (2 pins re-worded).
<!-- SECTION:NOTES:END -->
