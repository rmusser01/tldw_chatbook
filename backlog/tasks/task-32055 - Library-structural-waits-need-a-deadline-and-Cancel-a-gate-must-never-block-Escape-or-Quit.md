---
id: TASK-32055
title: >-
  Library structural waits need a deadline and Cancel; a gate must never block
  Escape or Quit
status: Done
assignee: []
created_date: '2026-09-08 18:23'
updated_date: '2026-09-08 19:41'
labels:
  - library
  - file-notes
  - notes
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The File Notes folder change sat on 'Folder Files · No folder selected · Changing folder…' indefinitely and Escape, the '‹ Library / Notes' cue, a palette deep link and Ctrl+Q were all swallowed (observed once, in a session already wedged by the note-load hang; the second assessor linked the same folder fine). Note loads, skill imports and exports share the same shape: a wait with no progress, no timeout and no exit. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Any structural wait longer than about 3 s shows a 'Still working…' line with a Cancel action
- [x] #2 Escape, the back cue and Ctrl+Q keep working while a wait is in progress; the gate vetoes the write, not the exit
- [x] #3 A folder change that fails or times out reports why and leaves the previously linked folder intact
- [ ] #4 Covered by tests that simulate a never-resolving service call for folder change and note load
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. StructuralWait helper (Tests/Library/test_library_structural_wait.py first): label/started_at/cancel + status_line(now, patience=3.0).
2. File Notes folder change: run set_root under asyncio.wait_for(30s) in a cancellable task owned by the wait; render status_line in the root-status slot; #library-structural-wait-cancel button; cancel/timeout keep the previous folder.
3. Never block the exit: LibraryScreen._flush_active_file_notes cancels the active structural wait (the single seam Escape, the back cue and app navigation all reach), so the gate vetoes only the write; pin check_action('quit') is not vetoed.
4. Same helper for the skill import wait and the export bundle write, one never-resolving-service test each.
5. Live-verify on the power profile (socket crit8-waits), docs + commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: `set_root` runs unbounded and holds `_root_transitioning`, and every leave path funnels into one veto -- `LibraryFileNotesWorkspace.flush_pending_work()` returns False while that flag is set. Escape, the '< Library / Notes' cue, the Database strip button (all via `_return_to_library_database_notes`) and the app's pre-navigation flush (the palette) therefore did nothing while a folder change hung. A gate meant to stop a second WRITE was gating the EXIT. Ctrl+Q was never gated by this screen (it defines no confirm_quit/prepare_for_quit); the critique's swallowed Ctrl+Q came from the already-wedged session. Tests now pin quit as un-vetoed.

Approach: one pure `StructuralWait` (`tldw_chatbook/Library/library_structural_wait.py`) carrying label/started_at/cancel/owner, with `status_line(now, patience=3.0)` -> `'<label>…'` then `'<label>… · still working · Cancel'` (the Cancel half is dropped when there is no cancel, so a wait never advertises one it does not have). The folder change now runs in a cancellable task under `asyncio.wait_for(..., 30)`; Cancel and timeout both bump `_root_generation` and clear `_root_transitioning` SYNCHRONOUSLY -- `set_root`'s own finally only runs once the cancelled coroutine unwinds, and every guard consulted in between would still refuse, which is exactly the race that swallowed Escape. `_flush_active_file_notes` (the one seam every exit reaches) abandons the wait instead of being blocked by it. The same helper drives the skill import (new `LibrarySkillImportCoordinator.cancel_running_import`, with an honest 'may still have landed' receipt since the write is on a thread) and the export bundle write (its existing `#library-export-cancel` button, status line upgraded).

Trade-offs: export keeps its own Cancel button id rather than mounting a second `#library-structural-wait-cancel` (it already ships a real cooperative cancel; two Cancels would be duplicate UI); its running copy changed to 'Exporting (N items)…' so the shared '<label>…' contract holds (one pinned assertion updated). A `receipt.focus` that `LibraryNotesTreeReceipt` has never had was removed from `_return_to_library_database_notes` -- it raised AttributeError and killed the whole Escape/back handler the moment Escape stopped being swallowed; the settled focus restore moved to where its identity is already built, which also turned `test_wide_files_task_return_restores_database_browse_receipt` from red to green.

AC#4 is ticked for folder change, skill import and export (never-resolving service in each). The note-load timeout half belongs to task-32050 by the wave's own split and is NOT covered here.

Tests: Tests/Library/test_library_structural_wait.py (5, new), Tests/UI/test_library_crit8_waits.py (7, new). File-notes regression: same failing NAME set as the stashed baseline, minus the one now fixed. Live-verified on the power profile (socket crit8-waits): folder change to a 25k-file directory showed 'Changing folder…' then 'Changing folder… · still working · Cancel' at ~3.8 s; Cancel left 'Folder change cancelled · previous folder kept' with the old folder still linked; Escape mid-change returned to Database notes; Ctrl+Q during a skill import quit cleanly. Captures in the wave's caps/ dir.

Files: tldw_chatbook/Library/library_structural_wait.py (new), Widgets/Library/library_file_notes_workspace.py, Widgets/Library/library_skills_canvas.py, UI/Screens/library_screen.py, UI/Library_Modules/library_export_controller.py, UI/Library_Modules/library_skill_import_controller.py, Tests/Library/test_library_structural_wait.py, Tests/UI/test_library_crit8_waits.py, Tests/UI/test_library_export_cancel.py, Tests/UI/test_library_export_receipt.py, Tests/UI/test_library_shell.py, Docs/User_Guide/library.md, Docs/User_Guide/library/{file-notes,skills,import-and-export}.md
<!-- SECTION:NOTES:END -->
