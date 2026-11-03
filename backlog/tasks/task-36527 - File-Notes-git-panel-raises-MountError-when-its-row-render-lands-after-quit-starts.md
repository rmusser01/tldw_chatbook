---
id: TASK-36527
title: File Notes git panel raises MountError when its row render lands after quit starts
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-03 07:24'
labels:
- follow-up
- library
- notes
- bug
priority: medium
references:
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
- [x] #1 Quitting while a File Notes rescan's git-row render is pending ends without a worker error
- [x] #2 A test pins it by holding the render until quit has started, and it is red before the fix and green after
<!-- AC:END -->

## Implementation Plan

1. Add a test in `Tests/UI/test_library_file_notes_git.py` that holds `_render_rows` (the panel harness), releases it and calls `app.exit()` in that order so the real body resumes with `app._exit` set, and asserts the body really ran under quit. Show it red (`WorkerFailed: MountError`) on the unfixed source.
2. In `_render_rows`, reuse the panel's own liveness guard (`is_attached`, as `_sync_push_availability` and the focus helpers already do): return before touching the list when the panel is not attached, re-check after the `clear()` await before mounting, and skip the `finally` UI sync when the panel is no longer attached.
3. Green the new test plus the file's covering tests, run the size ratchets and preflight, then finish the task file.

## Implementation Notes

`_render_rows` now reuses the panel's own liveness guard, `is_attached` (as in `_sync_push_availability` and the focus helpers). Quit sets `App._exit` before its drains, and from then on every widget reads as detached. So after the `clear()` await (the last yield before the mount) the render returns when the panel is no longer attached, and the `finally` skips its UI sync in that case. A render that lands after quit, or after the panel was removed, now does nothing. The diff is two conditions; the new comment is folded into the existing TASK-32800.5 comment so the module stays at its size-ratchet pin (4207).

- **Test:** `test_a_git_row_render_resuming_after_quit_does_nothing` (Tests/UI/test_library_file_notes_git.py). `_FencedQuitHarness` quits the way `TldwCli._shutdown` does: it sets `_exit`, then runs a drain that releases the held rescan render and waits for it to land. The test asserts that the render resumed under `_exit` and left the replacement untouched. A bare `App.exit()` could not reproduce the bug (see the new lessons entry).
- **Red before the fix:** `WorkerFailed: MountError("Can't mount widget(s) before ListView(id='file-notes-git-rows') is mounted")` from git_panel.py:3200 `extend`. Mutation controls: dropping the post-clear check brings back the same `WorkerFailed`; dropping the `finally` gate fails "the late render still synced the panel".
- **Green:** 10 of 10 serial runs at load average ~27-30.
- **Not done:** the commit-notes render (`_render_commit_review_notes`) has the same unguarded `extend`. It is outside these criteria and is proposed as a follow-up.
- **Files:** tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py, Tests/UI/test_library_file_notes_git.py, backlog/docs/lessons-testing-evidence.md.
