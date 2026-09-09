---
id: TASK-32134
title: >-
  Library Notes Import once: a selected source cannot be changed, Add another
  file is absent for a folder, and Last import is never offered after Back to
  Notes
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:36'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evidence assessor: after a wrong selection the configure phase offers only Check selection and Back to Notes; the guide's 'Add another file' does not render for a folder selection. Both assessors: after a completed import and Back to Notes no 'Last import' control appears; `can_revisit_receipt` requires the RECEIPT phase or a SELECT phase with nothing selected, which Back to Notes does not leave behind (INFERRED). Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The configure phase offers Change selection and Clear
- [x] #2 Last import is offered in the Notes list while a same-session receipt exists
- [x] #3 The guide matches the shipped controls
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. clear_selection() state transition back to an empty SELECT phase.
2. Change selection / Clear controls on a made selection; Add another file stays file-only.
3. Verify Last import survives Back to Notes and pin it.
4. Update the guide to the shipped controls.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `clear_selection(state)` to library_note_import_state (returns a fresh SELECT state that keeps `latest_receipt`, so the session receipt survives), `LibraryNoteImportController.clear_selection`, and two canvas messages (ChangeSourceRequested/ClearSourceRequested) wired through the notes controller and LibraryScreen. Change selection clears then reopens the picker; Clear just clears. Both render whenever a source is selected; Add another file stays file-only because a folder import is exclusive, and the guide now says that instead of promising the control (AC#3).

AC#2 was INFERRED in the critique and did not hold: `can_revisit_receipt` is already true in the RECEIPT phase and stays true after `clear_selection`, and the list canvas renders `#library-notes-import-receipt` when the flag is set. Both are now pinned by tests. It could NOT be confirmed live: on this branch point the Library ▸ Notes list pane renders blank on a populated profile, which reproduces identically on an unmodified HEAD worktree (crit#8's own P0, not this change).

Files: Library/library_note_import_state.py, UI/Library_Modules/library_note_import_controller.py, UI/Library_Modules/library_notes_controller.py, UI/Screens/library_screen.py, Widgets/Library/library_note_import_canvas.py, Tests/Library/test_library_note_import_state.py, Tests/UI/test_library_notes_wave_import_ux.py, Docs/User_Guide/library/notes.md. Change selection / Clear verified rendering live (cap 04).

Review addendum (Qodo finding 2): the receipt's skipped rows were derived from `state.plan`, which `clear_selection` discards, so revisiting the retained receipt after starting another selection showed a non-zero heading with no rows. They are now captured in `settle_import` and carried beside `latest_receipt`; `begin_selection` delegates to `clear_selection` so both reset paths behave identically.
<!-- SECTION:NOTES:END -->
