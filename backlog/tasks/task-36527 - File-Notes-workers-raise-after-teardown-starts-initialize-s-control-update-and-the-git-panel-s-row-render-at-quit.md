---
id: TASK-36527
title: 'File Notes workers raise after teardown starts: initialize''s control update and the git panel''s row render at quit'
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
Found by TASK-36383's implementation (PR #3232, 2026-10-23). That task fixed three flaky tests from the test side, by waiting past the races. The two product races behind two of them are still on dev 5b5920b8e2. Both run in workers started with Textual's default `exit_on_error=True`, so an exception ends the app with an error. In the tests it surfaced as `WorkerFailed`.
- **Initialize.** `LibraryFileNotesWorkspace._initialize` (workspace.py:2772-2794) awaits `_build_runtime` in a thread. It then checks only `self._active` and calls `_update_root_surface()` and `_update_controls()`. Textual removes a widget's children before its `on_unmount` clears `_active`, so a worker that resumes in that gap queries a removed control. TASK-36383 recorded `WorkerFailed: NoMatches('#file-notes-path-label')` at teardown. `_render_projection` (:3100-3121) already tolerates that gap with an `except NoMatches`, and its comment explains why.
- **Git row render at quit.** `LibraryFileNotesGitPanel._replace_rows` (git_panel.py:3156-3180) starts `_render_rows`. Its `list_view.extend(...)` (:3200) raises `MountError("Can't mount widget(s) before ListView(id='file-notes-git-rows') is mounted")` when it runs after `TldwCli._shutdown` has set `app._exit`, because `is_attached` is False under `_exit`. TASK-36383 caught five such failures, all after a Restore whose rescan was still running at quit. TASK-32800.5 already guarded the lookup in this worker but not the mount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Quitting while a File Notes rescan's git-row render is pending ends without a worker error
- [ ] #2 Removing a File Notes workspace while its initialize worker waits on its runtime thread ends without a worker error
- [ ] #3 A test pins each case by holding its producer until the teardown step, and it is red before the fix and green after
<!-- AC:END -->
