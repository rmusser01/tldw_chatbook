---
id: TASK-36527
title: File Notes git panel raises MountError when its row render lands after quit starts
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-03 10:02'
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

The git panel's worker renders (`_render_rows`, `_render_commit_review_notes`) and its deferred commit and push focus callbacks now stop once the panel is going away. The test is `_going_away()`, the repo's liveness idiom `not self.is_attached or self._pruning`. Quit sets `App._exit` before its drains, so `is_attached` reads False from then on; a removal sets `_pruning` before the children go. Each render checks it after the `clear()` await (the last yield before the mount); the commit-notes render checks it again after `extend`; the rows `finally` skips its UI sync. A render that lands after quit, or during a removal, does nothing. The module stays at its size-ratchet pin (4207): the new comments replace the old ones.

- **Test:** `test_a_git_render_resuming_after_quit_does_nothing[_render_rows]` (first named `test_a_git_row_render_resuming_after_quit_does_nothing`) (Tests/UI/test_library_file_notes_git.py). `_FencedQuitHarness` quits the way `TldwCli._shutdown` does: it sets `_exit`, then runs a drain that releases the held rescan render and waits for it to land. The test asserts that the render resumed under `_exit` and left the replacement untouched. A bare `App.exit()` could not reproduce the bug (see the new lessons entry).
- **Red before the fix:** `WorkerFailed: MountError("Can't mount widget(s) before ListView(id='file-notes-git-rows') is mounted")` from git_panel.py:3200 `extend`. Mutation controls: dropping the post-clear check brings back the same `WorkerFailed`; dropping the `finally` gate fails "the late render still synced the panel".
- **Green:** 10 of 10 serial runs at load average ~27-30.
- **AC#3:** a grep for `clear()`/`extend()`/`mount()` after an await in a worker finds one other render, `_render_commit_review_notes`. It first got the same post-`clear()` check; `test_a_git_render_resuming_after_quit_does_nothing[_render_commit_review_notes]` was red first with `WorkerFailed: MountError(... ListView(id='file-notes-git-commit-included-notes') ...)` at git_panel.py:2823.
- **AC#3, removal (review round 1):** the earlier claim that "a detach cancels the worker on the await" was wrong. A removal sets `_pruning` on the whole subtree at once, the children exit first, and the panel's workers are cancelled only at its own Unmount. Until then `is_attached` stays True. A render resuming in that window passed the guard, its `extend` was a silent no-op, and the next query raised `NoMatches`. Every guard now uses `_going_away()` (`not self.is_attached or self._pruning`, the repo idiom). It sits at both renders' post-`clear()` checks, at the commit-notes post-`extend` check, at the rows `finally`, and at `_focus_commit_control_if_current`.
  - New pins: `test_a_git_render_resuming_during_removal_ends_quietly[rows|notes]` and `test_a_commit_focus_landing_during_removal_ends_quietly`. On f106a13c58 they were red 5 of 5 runs, with `WorkerFailed: NoMatches('#file-notes-git-empty')`, `...('#file-notes-git-commit-included-selected')` and `NoMatches('#file-notes-git-commit-edit')`.
  - Mutation checks, each failing only its own pin: the rows `finally` back to `is_attached`; the notes post-clear and post-extend checks back to `is_attached`; the focus guard removed; the rows post-clear check reduced to the generation test.
  - The commit-notes post-`extend` check (a removal that starts during the mount) has its own pin, `test_a_commit_notes_render_removed_during_its_mount_ends_quietly`. A patched `extend` mounts the notes, starts the host's removal, and awaits the `#file-notes-git-commit-included-selected` Static's task. That Static exits before the panel's Unmount cancels the worker. Red 5 of 5 runs with the check reduced to the generation test: `WorkerFailed: NoMatches('#file-notes-git-commit-included-selected')`.
  - Review round 2 (same root): `_focus_push_list_control` and `_focus_push_control_if_current` now use `_going_away()` too. Pin: `test_a_push_focus_landing_during_removal_ends_quietly[list|operation]`, which calls `return_to_push_list()` / `restore_push_focus()` and then removes the panel. Red on 879619ca23: `NoMatches('#file-notes-git-back')` from both callbacks. All eight pins green 10 of 10 runs (`-n 4`, load ~13).
  - Census: five node ids at the tail at this point (seven after round 3).
  - Round 1's five pins green 10 of 10 runs (`-n 4`, load ~15). The reviewer's probe went from 15 of 24 crashing to 24 of 24 ok.
  - A render cut off by a removal leaves `_replacing_rows` set. Only `render_status`, or a `render_checking` that does not retain rows, starts a new row render that clears it; `render_checking(retain_rows=True)` leaves it set. On remount, `_rehydrate_git_presentation` re-renders the status in git mode.
  - The quit tests are parametrized now. Through round 2 the module stayed at its 4207 size pin, because the new helper is paid for by shorter comments.
- **Round 3 (review: 2 Important, 2 Minor).** Rule: every `call_after_refresh` target, and every panel entry point a workspace task can reach, opens with `if self._going_away(): return`.
  - Why: the workspace's push observer is a plain `asyncio.create_task` that nothing cancels on unmount, and workspace workers are cancelled only at the workspace's own Unmount. Either can call the panel after its children are gone, while it is still attached.
  - Guard added to 22 public entry points: `render_status`, `render_untrusted`, `render_checking`, `render_unavailable`, `mark_stale`, `set_mutating`, `set_last_action`, `clear_last_action`, `render_commit_form`, `render_commit_checking`, `render_commit_review`, `render_commit_confirming`, `render_commit_executing`, `render_commit_recovery_checking`, `render_commit_result`, `return_to_commit_list`, `render_push_review`, `render_push_progress`, `render_push_result`, `return_to_push_list`, `restore_push_focus`, `set_action_status` (through `set_last_action`).
  - Guard added to the deferred targets `_finish_mount`, `_fit_fixed_regions`, `_settle_action_focus` and `_sync_commit_body_height`.
  - State-only wrappers keep their state assignment, and the guard sits in the projection helper they call: `render_/clear_push_availability` use `_sync_push_availability` (Important #1, guard swapped); `render_/clear_commit_availability` and `invalidate_commit_binding` use `_sync_commit_availability`; `set_current_status` uses `_fit_fixed_regions`. A top-of-method return would drop state that `_finish_mount` re-projects on remount. The `set_current_status` mutation passes its own pin for exactly that reason.
  - Pins:
    - `test_a_panel_entry_called_during_removal_does_nothing[27 entry points]` starts `host.remove()`, awaits every child's task (the panel is then attached, pruning and childless), and calls the entry point.
    - `test_a_deferred_panel_refresh_landing_during_removal_does_nothing[_finish_mount, _fit_fixed_regions]` queues the target with `call_after_refresh` and removes the panel.
    - Red first: all 29 cases raised `NoMatches` on eb2cca31f3.
  - Mutation checks (each guard removed alone): 24 of 25 fail their pin with `NoMatches`. The exception is `set_current_status`, whose own guard was then dropped, as above.
  - Kept but not pinnable: `restore_push_focus` only schedules, and its target is pinned. `_settle_action_focus` holds widget references and runs no query. `_sync_commit_body_height` already catches `NoMatches`. Their probe-style cases stayed green on eb2cca31f3, so they are not in the test.
  - Out of the panel class and left unchanged: the row items' `_fit_copy` (its `not self.children` check covers a prune, since children exit first) and `PushDestinationAuthorizationDialog._sync_action_layout` (a separate screen). The row items' probe-style cases were green on eb2cca31f3. I did not probe the dialog.
  - Size: re-pinned 4207 -> 4253 in `Tests/Architecture/test_module_size_ratchet.py`, under the owner's 2026-10-31 size waiver (dated comment).
- **Round 4 (review: 1 Important, 1 Minor).** A guard must skip only the DOM projection; persistent state is recorded first.
  - The incident: the top guards in `return_to_commit_list` and `return_to_push_list` returned before `_show_*_phase` assigned the phase. `_finish_mount` re-projects `_commit_phase` on remount, and `_rehydrate_commit_presentation` does not reset it when there is no draft. So a commit that settled mid-removal (`_observe_commit_outcome` calls `invalidate_commit_binding` and then `return_to_commit_list`) remounted on its "executing" screen. Probe `gp-rev3/probe_phase_drop.py`: stale phase and surface at 3efa4df463, list/list at the fix.
  - Verdict per guarded method:
    - **Commit/push renders keep their top guard; the guard branch records the phase** (and the push operation id): `render_push_review` (review, op id), `render_push_progress` (phase, op id), `render_commit_checking` (checking), `render_commit_confirming` (confirming), `render_commit_executing` and `render_commit_recovery_checking` (executing).
    - **Same, with projection state moved above the guard:** `render_push_result` (`_push_result`), `render_commit_form` (`_active_commit_draft`), `render_commit_review` (`_commit_review`, `_commit_notes`, `_commit_included_expanded`), `render_commit_result` (`_commit_result`).
    - `return_to_push_list`: `_push_result = None` above the guard; the branch sets phase "list" and op id None.
    - `return_to_commit_list`: review, result, notes and expanded state above the guard; the branch sets phase "list". Its focus requests (`_commit_list_focus_pending`/`_preferred_group_id`/`_focus_selector`, the entry-focus consumption) stay behind the guard on purpose. They are DOM focus work, and left pending they would make the first row render after a remount pull focus into the panel.
    - `invalidate_commit_binding`: was already state-first; it is fixed by `return_to_commit_list`.
    - **Status family, top guard removed** (each is only state plus helper calls): `render_status`, `render_untrusted`, `render_checking`, `render_unavailable`, `mark_stale`, `set_mutating`. The guard moved into the DOM helpers they use: `_replace_rows` (selection, generation and `_replacing_rows` first), `_sync_empty_state`, `_update_actions`, and `_focus_is_inside` (which returns False, so no entry focus is armed). So every flag, row and status-text assignment now persists.
    - `set_last_action` / `clear_last_action`: the text is assigned before the guard.
    - Unchanged: `set_current_status`, `render_/clear_push_availability`, `render_/clear_commit_availability` (already state-first) and `restore_push_focus` (no state).
  - Pins:
    - `test_a_panel_entry_called_during_removal_does_nothing` now checks the recorded state for each of 27 entry points, both mid-removal and after the panel is detached (54 cases). Leave-type entry points start from an executing commit behind a push review, with a last action shown.
    - New: `test_a_commit_that_settles_mid_removal_remounts_on_the_list`.
    - At 3efa4df463, 21 entry cases failed with "dropped its state mid-removal", and the remount test failed.
  - Mutation checks: 18 of 18 fail their pin. They cover the 12 guard-branch phase assignments and six helper guards (`_replace_rows`, `_sync_empty_state` and `_update_actions` with `NoMatches`, set/clear_last_action, and `_focus_is_inside` with `NoScreen`, which only the detached case catches).
  - Minor: `_ENTRY_POINTS` now lists `restore_push_focus` (workspace.py:7697) instead of `set_action_status`, which has no production caller, and its comment says "each panel entry point the workspace calls".
  - Size: re-pinned 4253 -> 4259 under the same waiver.
- **Files:** tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py, Tests/UI/test_library_file_notes_git.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py, Tests/Architecture/test_module_size_ratchet.py, backlog/docs/lessons-testing-evidence.md, backlog/docs/lessons-textual.md.
