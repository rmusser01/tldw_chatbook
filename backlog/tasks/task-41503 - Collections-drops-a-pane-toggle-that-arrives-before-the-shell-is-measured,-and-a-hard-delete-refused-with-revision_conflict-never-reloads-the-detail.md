---
id: TASK-41503
title: "Collections drops a pane toggle that arrives before the shell is measured, and a hard-delete refused with revision_conflict never reloads the detail"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- collections
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
dependencies:
- TASK-41126
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41126 (libcoll, round 15; both also on base). (1) `LibraryScreen._toggle_library_media_reader_pane` (`library_screen.py:7919`) drops a Collections pane toggle (Escape or the grip) that arrives before the shell is measured, so the first key press does nothing. (2) A Collections hard-delete refused with `revision_conflict` only notifies; it never reloads the detail, so a retry is refused again until the user reselects the capture (code reading, not reproduced live). The saved-search `show_page` overlap with the pane-sync lock was fixed in 41126's fix round and is not part of this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pane toggle sent before the first layout measure is applied once the shell is measured, or is refused visibly
- [ ] #2 A hard-delete refused with `revision_conflict` reloads the capture so the next Delete press works without reselecting
- [ ] #3 Each behaviour has a test shown red before the fix
<!-- AC:END -->
