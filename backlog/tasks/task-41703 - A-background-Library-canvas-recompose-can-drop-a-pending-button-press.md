---
id: TASK-41703
title: "A background Library canvas recompose can drop a pending button press"
status: To Do
assignee: []
created_date: 2026-11-18 04:24
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/canvas_sync.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every Library canvas `sync_state` recomposes all of the canvas's children. If an unrelated sync arrives while a click's `Pressed` message is still inside the canvas, the press is lost and the user's click does nothing.

Found by TASK-41532 on the Media type-strip opener in `test_library_media_complete_facets_commit_during_initial_recovery[loading-All]`. The opener was pruned before its press reached the handler, and focus ended on the replacement type filter.

Lead, unverified: Media's empty-list entry fallback (`_library_media_empty_list_fallback_target`) runs while the 2 s entry window is armed.

Until this is fixed, the test works around it by re-pressing the opener (`Tests/UI/test_library_shell.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pilot test presses a canvas button, forces an unrelated canvas sync before the press bubbles up, and shows the press handled; red before the fix
- [ ] #2 The notes name what syncs the Media canvas while page 1 is held back in `[loading-All]`
- [ ] #3 The re-press workaround in `test_library_media_complete_facets_commit_during_initial_recovery` is removed once the press can no longer be lost
<!-- AC:END -->
