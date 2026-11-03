---
id: TASK-36527
title: File Notes git panel raises MountError when its row render lands after quit starts
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-03 08:23'
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
- [x] #3 Every other worker render in the git panel that clears and then mounts into a list after an await stops without error when the panel is detached or the app is quitting
<!-- AC:END -->

## Implementation Plan

1. Add a test in `Tests/UI/test_library_file_notes_git.py` that holds `_render_rows` (the panel harness), releases it and calls `app.exit()` in that order so the real body resumes with `app._exit` set, and asserts the body really ran under quit. Show it red (`WorkerFailed: MountError`) on the unfixed source.
2. In `_render_rows`, reuse the panel's own liveness guard (`is_attached`, as `_sync_push_availability` and the focus helpers already do): return before touching the list when the panel is not attached, re-check after the `clear()` await before mounting, and skip the `finally` UI sync when the panel is no longer attached.
3. Green the new test plus the file's covering tests, run the size ratchets and preflight, then finish the task file.
4. (AC#3) Apply the same post-`clear()` `is_attached` guard to every other worker render in the panel that clears and then mounts after an await (only `_render_commit_review_notes`), pinned by a fenced-quit test that is red first.

## Implementation Notes

The git panel's worker renders (`_render_rows`, `_render_commit_review_notes`) and its deferred commit focus now stop once the panel is going away. The test is `_going_away()`, the repo's liveness idiom `not self.is_attached or self._pruning`. Quit sets `App._exit` before its drains, so `is_attached` reads False from then on; a removal sets `_pruning` before the children go. Each render checks it after the `clear()` await (the last yield before the mount); the commit-notes render checks it again after `extend`; the rows `finally` skips its UI sync. A render that lands after quit, or during a removal, does nothing. The module stays at its size-ratchet pin (4207): the new comments replace the old ones.

- **Test:** `test_a_git_render_resuming_after_quit_does_nothing[_render_rows]` (first named `test_a_git_row_render_resuming_after_quit_does_nothing`) (Tests/UI/test_library_file_notes_git.py). `_FencedQuitHarness` quits the way `TldwCli._shutdown` does: it sets `_exit`, then runs a drain that releases the held rescan render and waits for it to land. The test asserts that the render resumed under `_exit` and left the replacement untouched. A bare `App.exit()` could not reproduce the bug (see the new lessons entry).
- **Red before the fix:** `WorkerFailed: MountError("Can't mount widget(s) before ListView(id='file-notes-git-rows') is mounted")` from git_panel.py:3200 `extend`. Mutation controls: dropping the post-clear check brings back the same `WorkerFailed`; dropping the `finally` gate fails "the late render still synced the panel".
- **Green:** 10 of 10 serial runs at load average ~27-30.
- **AC#3:** a grep for `clear()`/`extend()`/`mount()` after an await in a worker finds one other render, `_render_commit_review_notes`. It first got the same post-`clear()` check; `test_a_git_render_resuming_after_quit_does_nothing[_render_commit_review_notes]` was red first with `WorkerFailed: MountError(... ListView(id='file-notes-git-commit-included-notes') ...)` at git_panel.py:2823.
- **AC#3, removal (review round 1):** the earlier claim that "a detach cancels the worker on the await" was wrong. A removal sets `_pruning` on the whole subtree at once, the children exit first, and the panel's workers are cancelled only at its own Unmount. Until then `is_attached` stays True. A render resuming in that window passed the guard, its `extend` was a silent no-op, and the next query raised `NoMatches`. Every guard now uses `_going_away()` (`not self.is_attached or self._pruning`, the repo idiom). It sits at both renders' post-`clear()` checks, at the commit-notes post-`extend` check, at the rows `finally`, and at `_focus_commit_control_if_current`.
  - New pins: `test_a_git_render_resuming_during_removal_ends_quietly[rows|notes]` and `test_a_commit_focus_landing_during_removal_ends_quietly`. On f106a13c58 they were red 5 of 5 runs, with `WorkerFailed: NoMatches('#file-notes-git-empty')`, `...('#file-notes-git-commit-included-selected')` and `NoMatches('#file-notes-git-commit-edit')`.
  - Mutation checks, each failing only its own pin: the rows `finally` back to `is_attached`; the notes post-clear and post-extend checks back to `is_attached`; the focus guard removed; the rows post-clear check reduced to the generation test.
  - Not pinned on its own: the commit-notes post-`extend` check, which covers a removal that starts during the mount.
  - All five pins green 10 of 10 runs (`-n 4`, load ~15). The reviewer's probe went from 15 of 24 crashing to 24 of 24 ok.
  - A render cut off by a removal leaves `_replacing_rows` set. The next `render_status` or `render_checking` replaces the rows; on remount, `_rehydrate_git_presentation` does that.
  - The quit tests are parametrized now, and the five pins are appended at the census tail. The module stays at its 4207 size pin: the new helper is paid for by shorter comments.
- **Files:** tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py, Tests/UI/test_library_file_notes_git.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py, backlog/docs/lessons-testing-evidence.md, backlog/docs/lessons-textual.md.
