---
id: TASK-38647
title: File Notes conflict-discard test asserts Cancel focus before the deferred focus lands
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 05:51
updated_date: 2026-11-03 06:57
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_file_notes_workspace.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
dependencies:
- TASK-38646
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-38646 (2026-11-03). `Tests/UI/test_library_file_notes_workspace.py::test_conflict_resolution_discard_keeps_cancel_first_confirmation` fails now and then at `assert cancel.has_focus` (:5964 on dev 18482c1cf7). It already failed on dev-identical code, through the scratch-profile wrapper:
- run on its own: 2 of 6;
- in a 3-test run with `-n 4`: 5 of 8;
- in the whole file with `-n 4`: 1 of 3.

**Cause.** The test waits only for `reload_confirmation_active`, then asserts focus at once. `_set_reload_confirmation` places that focus later, through `self.call_later(cancel.focus)` (library_file_notes_workspace.py:3627). This is the same family of race as TASK-38646.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes 10 of 10 runs on its own and 10 of 10 three-test runs with `-n 4`, each its own pytest process, because it waits for Cancel to take focus
- [x] #2 It still fails if Cancel never takes focus after the confirmation opens (shown under a temporary source edit)
- [x] #3 Every other test in the file that asserts Cancel focus right after a reload confirmation opens waits for the deferred focus the same way
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Test-only synchronization fix; no product or architectural change.

1. Show the red at base 18482c1cf7: the test alone and in three-test `-n 4` runs, each its own pytest process through the scratch wrapper.
2. Replace the immediate `assert cancel.has_focus` with the file's `_wait_until` on `cancel.has_focus`, the same form the test already uses for the focus that Cancel returns to Discard. No fixed pause.
3. AC#1: 10 single-test runs and 10 three-test `-n 4` runs, each its own pytest process.
4. AC#2: temporary source edit that drops the deferred `cancel.focus`; the test must fail on the new wait, then revert.

## Implementation Notes

This change touches only the test, in `Tests/UI/test_library_file_notes_workspace.py`. The immediate `assert cancel.has_focus` after the confirmation opens is now `_wait_until(pilot, lambda: cancel.has_focus, "destructive confirmation did not focus Cancel")`. That is the form the same test already used for the focus Cancel returns to Discard, and the form `test_reload_confirmation_keeps_target_and_save_copy_reachable` already used. There is no fixed pause, and product code is unchanged.

**Red at base 18482c1cf7.** The test alone failed 4 of 6 runs, each its own pytest process through the scratch wrapper. All 4 failed at :5964 (`cancel.has_focus` was False). Load averages were 24 to 45.

**AC#1.**
- On its own: 10 of 10 runs passed, each its own pytest process. Load averages were 34 to 40.
- Three-test runs with `-n 4`, each its own pytest process: this test passed in all 10. The three tests were this one, `test_folder_files_capability_inventory_and_modes` and `test_folder_files_shared_shell_retains_state_across_breakpoints`. Load averages were 29 to 39.
- 3 of those 10 runs failed only `test_folder_files_capability_inventory_and_modes`, at :1886 (`restore.disabled`). That is TASK-38646's race, which PR #3356 fixes and which is not on dev yet.

**AC#2.** A temporary source edit replaced `self.call_later(cancel.focus)` in `_set_reload_confirmation` with `pass`. The test then failed 2 of 2 runs at the new wait ("destructive confirmation did not focus Cancel"). The edit was reverted with `git checkout`.

**AC#3.** `test_conflict_reload_requires_keyboard_confirmation_in_library_shell` now waits for Cancel focus after its first open, instead of `assert cancel.has_focus`. It also waits before its third open's `tab`, which assumed Cancel already held focus. With focus dropped on every open, the first wait failed (2 of 2 cases). With focus dropped only on the third open, the second wait failed (2 of 2), and the old form failed at `assert confirm.has_focus`. Both edits were reverted. It then passed 5 of 5 `-n 4` runs together with the conflict-discard test (load 23 to 53). No other test in the file asserts Cancel focus right after a confirmation opens.

**Checks.** Preflight passes. Ruff format is clean, and the Ruff lint count is unchanged from base (24).

Files: `Tests/UI/test_library_file_notes_workspace.py`.
