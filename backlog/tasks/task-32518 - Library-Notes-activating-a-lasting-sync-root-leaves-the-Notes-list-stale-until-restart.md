---
id: TASK-32518
title: "Library Notes: activating a lasting-sync root leaves the Notes list stale until restart"
status: To Do
assignee: []
created_date: '2026-09-13 00:30'
updated_date: '2026-09-13 03:40'
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

Reproduction (dev 7159fc0b99, 235x52):
1. A seeded scratch profile (10 notes) and a 60-`.md` vault under `$HOME`
   (`mkvault.py <dest> --git --archive 45`).
2. Library ▸ Notes → **Add from files…** → **Keep a folder synced** → type a
   label → **Choose folder…** → type the vault path into "File name" → Enter
   → **Select folder** → **Check changes** ("60 safe · 0 need attention").
3. **Activate reviewed root** ("Sync root activated. 60 applied") → **Back
   to Notes**: still "Notes (10)", no ⇄ Sync managed row.
4. **Manage sync folders** → **Check changes** → **Back to Notes**; re-select
   the rail's Notes row; **Folder files** → **Library notes**: still 10.
5. Quit and relaunch: "Notes (70)" and the folder row.

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
- [ ] #1 After Activate reviewed root, returning to the Notes list shows the managed folder row and a count that includes the synced notes, without restarting the app
- [ ] #2 A manual Check changes that applies changes refreshes the list the same way
- [ ] #3 A test on a seeded profile (existing notes and folders) pins the refresh; the fresh-profile path keeps working
<!-- AC:END -->
