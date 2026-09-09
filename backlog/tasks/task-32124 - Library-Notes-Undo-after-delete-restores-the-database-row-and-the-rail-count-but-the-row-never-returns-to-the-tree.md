---
id: TASK-32124
title: >-
  Library Notes Undo after delete restores the database row and the rail count
  but the row never returns to the tree
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 07:14'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed once by the evidence assessor: after Undo the count returned to Notes (11) and `deleted=0` in the DB, but the tree still listed 10 titles and Clear filter did not bring the row back. Not re-verified by the parent because the Undo button was unreachable (task-32123). Cause INFERRED: `_undo_library_note_delete` appends the restored record to the flat source records and re-syncs the canvas, while the tree projection is built from paged branch state that is not invalidated. The guide promises that Undo 'immediately returns its row and the Notes rail count'. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Undo the restored note's row appears in its folder (or Unfiled) without any further action, and focus lands on it
- [x] #2 Covered by a test through the tree projection, not only the flat list
- [x] #3 Verified live on a seeded profile
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test through the tree projection: delete, undo, assert the row returns and is selected.\n2. Route the successful restore through the existing tree locator seam.\n3. Live-verify undo on the power profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The inferred cause was right: `_undo_library_note_delete` patched the flat source records and re-synced the canvas, but the tree is projected from paged branch slices that nothing invalidated. Written as a failing test through the projection first (delete leaves the branch without the placement; undo must put it back), and reproduced live twice on the seeded profile.

Fixed by committing the restore through the seam a create already uses -- `_reconcile_library_notes_tree_mutation("note_create", {"note_id": ...})` -- which reloads exactly the affected branches (the note's folders plus Unfiled) and selects the restored placement. No second refresh mechanism. The controller gained one wiring parameter for that screen-resident method, the way every other unbound `LibraryScreen.<x>(self, ...)` target in this cluster is bound.

Recorded because it cost a live round: the obvious seam, `_locate_library_notes_tree_target`, does NOT work here. Repainting the canvas removes the receipt the pressed Undo button lives in; the focus move that follows is classified as user intent by `on_descendant_focus`, which bumps `focus_intent_generation` and supersedes the locator's navigation before its first await returns. It returns False silently, with no warning event -- live proof: the count went back to 10 and the row never came back.

AC#1 is pinned for BOTH halves of "in its folder (or Unfiled)": the projection test is parametrized over an Unfiled restore and a restore into a folder, and asserts the exact placement id in each. The FOCUS half is evidenced live rather than in that test, because the fake screen stubs `_restore_library_notes_focus_identity` (it has no DOM); what the test pins there is the selection the restore lands on, which is what focus follows.

Live (AC#3): caps/03-delete-receipt.txt (Notes (9), row gone) then caps/05-undo-row-returns.txt (Notes (10), "Groceries (not work) · now" back under Unfiled, selected and focused).

Files: tldw_chatbook/UI/Library_Modules/library_notes_controller.py, tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_notes_wave_list.py.
<!-- SECTION:NOTES:END -->
