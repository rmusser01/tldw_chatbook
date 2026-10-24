---
id: TASK-36527
title: File Notes git panel raises MountError when its row render lands after quit starts
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- library
- notes
- bug
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
dependencies:
- TASK-36383
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36383's implementation (PR #3232, 2026-10-23). That task fixed three flaky tests from the test side, by waiting past the races. The product race behind one of them is still on dev 5b5920b8e2. It runs in a worker started with Textual's default `exit_on_error=True`, so an exception ends the app with an error; in the tests it surfaced as `WorkerFailed`. (The other race TASK-36383 recorded, `NoMatches('#file-notes-path-label')` from `_initialize`'s control update at teardown, is TASK-32570.)
- **Git row render at quit.** `LibraryFileNotesGitPanel._replace_rows` (git_panel.py:3156-3180) starts `_render_rows`. Its `list_view.extend(...)` (:3200) raises `MountError("Can't mount widget(s) before ListView(id='file-notes-git-rows') is mounted")` when it runs after `TldwCli._shutdown` has set `app._exit`, because `is_attached` is False under `_exit`. TASK-36383 caught five such failures, all after a Restore whose rescan was still running at quit. TASK-32800.5 already guarded the lookup in this worker but not the mount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Quitting while a File Notes rescan's git-row render is pending ends without a worker error
- [ ] #2 A test pins it by holding the render until quit has started, and it is red before the fix and green after
<!-- AC:END -->
