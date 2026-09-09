---
id: TASK-32144
title: >-
  Idea: Library Notes Trash view under the tree so the delete receipt is not the
  only safety net
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 17:34'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - idea
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improvement pitched by both assessors: a 'Recently deleted (N)' row under the folder tree holding soft-deleted notes for 30 days, matching the Media Trash grammar. The receipt is a good immediate affordance but a bad only one (task-32123, task-32124). Size M. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design agreed with the user before implementation — satisfied by the controller's recorded ruling on the user's delegation of this riders wave (the shape it fixed: a `Recently deleted (N)` row under the folder tree, a list of soft-deleted notes with per-row Restore, the Media Trash key grammar, and no permanent delete).
- [x] #2 Restore from the Trash view returns the row and count exactly as Undo does
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the Undo seam (_undo_library_note_delete -> NotesScopeService.restore_note) and the Media Trash grammar.
2. Repository: bounded 'deleted = 1' query newest-first (ChaChaNotes_DB -> Notes_Library -> NotesScopeService.list_deleted_notes), paged 20 with an exact total.
3. Domain: LibraryNotesTrashState/Row + builder in Library/library_notes_state.py (title, age, version).
4. Canvas: 'Recently deleted (N)' row under the folder tree, hidden at zero; a 'trash' mode listing the rows with a per-row Restore and a back action. No permanent delete.
5. Controller/screen: open/back, restore through the SAME _undo_library_note_delete receipt seam, Escape back to the list, 'r' on the focused row, footer chips.
6. Tests first (RED), then live-verify on the power profile, guide section, backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adds a **Recently deleted (N)** row as the folder tree's last row, opening a Trash view of the soft-deleted notes with a per-row **Restore**.

Restore is not a second implementation of recovery: the pressed row is turned back into a `LibraryNoteDeleteReceipt` (a tombstone's own `version` is what `restore_note` expects) and handed to `_undo_library_note_delete` -- the exact seam the delete receipt's Undo commits through -- so the row returns to its folder or Unfiled via the tree reconciler and the rail count moves identically (AC#2). That seam now also reloads the Trash snapshot, so both entry points keep the count truthful from one place.

Layers: `list_deleted_notes` on `CharactersRAGDB` (page + exact total in ONE transaction, `deleted = 1`, `ORDER BY last_modified DESC, id`), through `NotesInteropService` and a local-only `NotesScopeService.list_deleted_notes`; `LibraryNotesTrashRow`/`LibraryNotesTrashState`/`build_library_notes_trash_state` + `LIBRARY_NOTES_TRASH_PAGE_SIZE = 20`; two `LibraryNotesState` fields (`trash`, `trash_loading`); `_compose_trash_opener`/`_compose_trash` on the canvas; open/back/restore handlers, the Escape branch and the `"trash"` focus region on the controller; `Binding("r", …)` with its `check_action` gate, the footer tier and two refresh hooks on the screen; one CSS rule.

Decisions: the opener is the row list's LAST ROW, not a sibling after it -- live at 235x52 the list is `height: 1fr`, so a sibling docked to the pane foot twelve blank rows adrift from the tree (the detached-affordance shape task-28015 fixed in the Media Trash). Absent at zero rather than disabled. Paged 20 with an honest "Showing the N most recently deleted of TOTAL" line instead of a second paging state machine. No Danger group: ADR-055 keeps destruction behind its own receipt and this surface exists to recover. Escape needed no new binding -- `library_notes_escape` already gates on the Notes workflow.

Reconciled pins, each with the reason at the pin: the notes-state field census 100 -> 102, and the wave-list Undo fake gains a `_refresh_library_notes_trash` stub (the shared seam genuinely gained that step).

Tests: new `Tests/UI/test_library_notes_riders_trash.py` (14), three DB cases and two scope-service cases. Failing NAME sets match the `d0ff40842f` baseline exactly (5 in `test_library_notes_canvas.py`, 1 in `test_chachanotes_db.py`, 3 in `test_library_notes_folder_navigator.py`). Live-verified on the power profile at 235x52 and 100x30: delete two, open the Trash, restore with `r` and with the button, count and row both return, Escape goes back, the row disappears at zero (captures 01-09 under SCRATCH/notes-crit/wave2/i-trash/caps/).

Files: `DB/ChaChaNotes_DB.py`, `Notes/Notes_Library.py`, `Notes/notes_scope_service.py`, `Library/library_notes_state.py`, `UI/Library_Modules/library_notes_state.py`, `UI/Library_Modules/library_notes_controller.py`, `UI/Screens/library_screen.py`, `Widgets/Library/library_notes_canvas.py`, `css/components/_agentic_terminal.tcss` (+ regenerated bundle), `Docs/User_Guide/library/notes.md`, and the tests above.
<!-- SECTION:NOTES:END -->
