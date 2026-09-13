---
id: TASK-32518
title: >-
  Library Notes: activating a lasting-sync root leaves the Notes list stale
  until restart
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 00:30'
updated_date: '2026-09-13 03:24'
labels:
  - library
  - notes
  - rider
dependencies:
  - TASK-32269
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a profile that already holds notes, **Activate reviewed root** writes the
synced notes and their managed folder to the database ("Sync root
activated. 60 applied · durable receipt recorded" — the `notes` row count
goes 10 → 70 and `note_folders` gains the display-name folder), but the
Notes list beside it does not pick any of it up: the count stays
"Notes (10)", no **⇄ Sync managed** folder row appears, and neither a manual
**Check changes**, re-selecting the rail's Notes row, nor a Folder files →
Library notes round trip refreshes it. Restarting the app shows
"Notes (70)" and the "t12b sync ⇄ Sync managed" folder at once. The wave-3
sync walk (task-32269) saw the folder appear because it ran on a fresh
profile, where the empty state recomposes; the seeded case was never walked.

**Cause (inferred from code, not yet proven by a test).** The import path has
`_refresh_after_library_note_import` (`library_notes_controller.py`), which
calls `_refresh_local_source_snapshot()` and
`_request_library_notes_tree_initial_load()` once execution settles. The
lasting-sync path has no counterpart: `LibraryNotesSyncController.activate_root`
ends in phase "receipt" and `refresh_roots()`, and the notes controller's
consumer of that snapshot (`_publish_library_notes_lasting_sync_snapshot`)
only re-syncs the canvas while the lasting view is open;
`_exit_library_notes_lasting_sync` (Back to Notes) sets the view to "list" and
re-syncs the canvas from the retained snapshot without invalidating it. Why
the rail re-select and the source round trip do not refresh either needs
tracing in the fix (the tree loader appears to short-circuit on an
already-loaded scope).

Evidence: wave-3 docs sweep on dev 7159fc0b99, reproduced twice —
`wave3-caps/docs-sweep/b06-root-activated-235x52.txt` (receipt, 60 applied),
`b07-list-after-activate` (Notes (10), no folder row), `b08-manual-check`,
`b09-list-after-check`, `b10-list-after-source-roundtrip` (still 10, no
folder), `b11-list-after-restart` (Notes (70), "t12b sync ⇄ Sync managed");
the database was read directly between b06 and b10 (`notes` 70 rows,
`note_folders` holds "t12b sync"). First run, older profile (69 → 129):
`48-root-activated`, `49-list-after-activate`, `52-list-after-check`,
`53-list-after-reload`, `54-list-after-restart`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Activate reviewed root, returning to the Notes list shows the managed folder row and a count that includes the synced notes, without restarting the app
- [x] #2 A manual Check changes that applies changes refreshes the list the same way
- [x] #3 A test on a seeded profile (existing notes and folders) pins the refresh; the fresh-profile path keeps working
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on the real route: mount LibraryScreen (Tests/UI/test_library_notes_files_sync_journey.py harness) over a REAL NotesScopeService/CharactersRAGDB seeded with existing notes and a real lasting-sync runtime (build_notes_sync_runtime_owner) over a vault of .md files; walk Add from files -> Keep a folder synced -> Check changes -> Activate reviewed root -> Back. RED: the list still paints the pre-activation count and no managed folder row.
2. Cause: the lasting-sync path has no counterpart of the import path's refresh_after_settlement. LibraryNotesSyncController.activate_root/apply_reviewed end in a receipt and refresh_roots() only; nothing tells the Notes list to refetch. Fix: give LibraryNotesSyncController one optional refresh_notes callback (same shape as the import controller's refresh_after_settlement), invoked after an accepted activation and after an apply that changed anything; the screen wires it to the existing _refresh_after_library_note_import (source snapshot + tree initial load) -- the same path a note import uses. No second refresh mechanism.
3. Keep the fresh-profile journey green (existing test_lasting_review_activation_receipt_and_remount_recovery_journey).
4. Live-verify on a seeded scratch profile at 235x52 and 100x30; captures under wave3-caps/sync-tail/.
5. Guide: note the refresh in the lasting-sync chapter and stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause (now proven by test): the lasting-sync path had no counterpart of the import path's refresh_after_settlement. LibraryNotesSyncController.activate_root / apply_reviewed ended in a receipt and refresh_roots() only, so the Notes list kept its pre-activation source snapshot and tree until restart; a fresh profile hid it because the empty list recomposes on its own.

Fix: LibraryNotesSyncController takes one optional refresh_notes callback (tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py; same shape as the import controller's refresh_after_settlement), invoked after an accepted activation and after an apply that changed anything (applied > 0); LibraryScreen wires it to the existing _refresh_after_library_note_import (source snapshot worker + tree initial load) -- the same refresh an import uses, no second mechanism. Undo / cleanup paths are not wired (they change note content, not the count or folder rows).

Tests (Tests/UI/test_library_notes_files_sync_journey.py): test_activating_a_lasting_root_refreshes_a_seeded_notes_list mounts the real LibraryScreen over a real NotesScopeService/CharactersRAGDB seeded with 2 notes + a folder and a real runtime over a 3-file vault, walks Add from files -> Keep synced -> Check -> Activate -> Back and sees 'Notes (5)' plus the managed folder row (RED on origin/dev: 'list count never refreshed after activation'); test_applying_a_reviewed_conflict_refreshes_the_notes_list_once pins apply refreshes once and a bare Check does not. The fresh-profile journey stays green.

Live: 235x52 seeded profile (10 notes, 58-file vault): Activate '60 applied' -> Back -> 'Notes (70)' + '▸ t13 sync ⇄ Sync managed' at once (wave3-caps/sync-tail/06, 07); 100x30 second root: '5 applied' -> 'Notes (75)' + 't13 second' (26, 27). Guide: notes.md step 5 + new stamp.
<!-- SECTION:NOTES:END -->
