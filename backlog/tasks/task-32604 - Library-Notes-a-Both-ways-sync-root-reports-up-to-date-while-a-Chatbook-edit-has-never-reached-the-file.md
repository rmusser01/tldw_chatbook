---
id: TASK-32604
title: >-
  Library Notes: a Both-ways sync root reports up to date while a Chatbook edit
  has never reached the file
status: To Do
assignee: []
created_date: '2026-09-15 06:36'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P0, persona Alex / solo-operator, Keep-a-folder-synced workflow. Heuristic 1 setter (2 -> 1).

What happened. Root PowerVault activated, direction Both ways. Edited the bound note 2026-09-07 in Chatbook; the editor said 'Saved 23:08' (A cap 51). md5 of vault/Daily/2026-09-07.md was 641d3a99... before the edit, 6 s after, 26 s after, and after an explicit Check changes. The root row read '✓ Up to date · Next: Check changes' throughout (A caps 52, 53); the manual check then set the status line to 'Manual check finished. Review exact effects.' with nothing to review and no Review control; no receipt was written. No pending state exists anywhere on the surface.

Attribution: INDEPENDENT of wave 4, PROVEN. A headless probe built on the production runtime/executor/controller stack (the shape Tests/UI/test_library_notes_files_sync_journey.py::_start_real_conflict_stack uses) was run twice -- against 77eb2601a6 and against a git archive extraction of 5fd502dbac, the critique-#3 tip. Both runs are identical: after a note-side edit and 3 s the row stays up_to_date/Check changes and the file is unchanged; sync_now sets phase=review and 'Manual check finished. Review exact effects.' while leaving the row untouched; request_sync_now returns [('update_file', 'note_changed')]; apply_reviewed on that plan writes the file correctly. Wave 4's own pre-fix probe recorded the same asymmetry in task-32534's description. Critique #3 missed it because its walk also edited the disk, which woke the watcher and carried the Chatbook edit across as a side effect.

Cause, PROVEN, three links. (1) Nothing on the note side ever triggers sync: _ProductionRuntimeAdapter.changed_root_ids (Notes/notes_sync_runtime.py:1063-1078) signs only filesystem metadata (display_path, device, inode, size, modified_ns, changed_ns) via _discovery_signature (:1048-1061), and PollingNotesSyncWatcher is the only producer of hints (:1919, :2979 -- no other schedule_hint caller exists). (2) The root row is projected from stored root state, never from a fresh plan. (3) The review the manual Check builds has no door: LibraryNotesSyncController.sync_now (UI/Library_Modules/library_notes_sync_controller.py:1378-1419) installs it and sets phase='review', but handle_library_notes_lasting_root_action (UI/Library_Modules/library_notes_controller.py:5152-5153) leaves the view on lasting_roots for the 'check' action, and only library_notes_add_from_files_canvas.py:482 renders phase=='review'; the roots canvas offers a Review button only when next_action == 'review_changes' (library_notes_sync_roots_canvas.py:174-176), which only the automatic pass can set.

Docs contradicted: notes.md promises Receipts shows 'Wrote note to file' for a note you edited in Chatbook. No user-reachable path produces that receipt today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A root holding note-side changes that are not on disk never renders '✓ Up to date'; it renders an explicit pending state naming the count and the next action
- [ ] #2 A Chatbook-side save inside an active root either writes to disk on the same terms a disk-side edit is picked up, or the editor's own status line says the note is saved in Notes and not yet written to the named file
- [ ] #3 Check changes from Manage sync folders reaches the review it builds: when the plan has actions the user lands on them, and when it has none the status line says 'Nothing to review', not 'Review exact effects'
- [ ] #4 Every note-to-file write leaves the Receipts row the guide already promises, and notes.md's Receipts paragraph matches what ships
- [ ] #5 Regression test on the production runtime: note edit -> check -> the plan's update_file is reachable and applied; a row is never projected as up to date while an update_file action is pending
<!-- AC:END -->
