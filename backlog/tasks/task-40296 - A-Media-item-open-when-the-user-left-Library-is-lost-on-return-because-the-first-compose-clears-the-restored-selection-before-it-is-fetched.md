---
id: TASK-40296
title: "A Media item open when the user left Library is lost on return, because the first compose clears the restored selection before it is fetched"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- media
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/app_navigation.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified both nodes as one product defect and left them red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** A Media item that was open when the user left Library is not reopened when they come back. Its detail is never fetched, and a deleted item never shows its recovery state.

**Cause, from TASK-39152's notes.** `restore_state` (`UI/Screens/library_screen.py:10664`; `app_navigation.py` calls it) sets the selected id. The first `compose_content` then clears it through `_adopt_library_media_highlight` (`UI/Library_Modules/library_media_controller.py:3224`) before `on_mount` holds the highlight.

**Nodes.**
- `Tests/UI/test_library_shell.py::test_library_shell_restored_media_viewer_fetches_detail_on_mount`: the restored selection is `None`.
- `::test_library_shell_restored_media_viewer_with_deleted_item_shows_recovery`: "#library-media-viewer-error never mounted within 30.0s".

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Returning to Library with a Media item open reopens that item's viewer and fetches its detail
- [ ] #2 Returning with an open item that was deleted meanwhile shows the viewer's recovery state
- [ ] #3 Both nodes pass 3 of 3 at `-n 4`, shown red before the fix, and their lines leave `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->
