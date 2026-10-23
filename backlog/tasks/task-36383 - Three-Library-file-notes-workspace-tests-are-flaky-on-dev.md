---
id: TASK-36383
title: Three Library file-notes workspace tests are flaky on dev
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 22:12
labels:
- follow-up
- tests
- flaky
- library
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-35649's implementation (2026-10-22). On dev 28beeb9ed5, four reruns of `Tests/UI/test_library_file_notes_workspace.py` failed 1, 1, 3 and 2 of these tests:
- `test_file_notes_navigation_and_key_guidance_use_one_phrase`
- `test_initial_root_scan_projects_checking_authority_while_actions_are_gated`
- `test_create_move_delete_protect_and_restore_use_real_service`

The cause is not yet identified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each test's failure mode is identified and recorded
- [x] #2 Each test passes 20 of 20 serial single-node runs on dev
- [x] #3 A fix that changes a wait shows the old form failing under the same conditions (a negative control)
<!-- AC:END -->

## Implementation Plan

1. Reproduce each test's failure under the one-window idle probe and machine load; record where the asserted (or crashing) state comes from.
2. Test 1: the body ends after one `pilot.pause()` while the workspace's `_initialize` worker is in flight; wait for `initialized` instead.
3. Test 2: the root-status read races the deferred `_fit_root_status`, which elides the row to 11 cells at 60x20; assert the row's unfitted summary instead of the painted text.
4. Test 3: the body ends once the restored file exists on disk, while the Restore handler is still rescanning and the git rows are still rendering; wait for the reopened file and the settled row render, and replace the fixed pause before Restore with a wait for a pressable Restore.
5. Negative controls: inject a delay at each producer so the old form fails the same way, and show the new form passes under it.
6. 20 serial single-node runs per test, one `-n 4` run of the whole file, census check.

## Implementation Notes

Test-only fix in `Tests/UI/test_library_file_notes_workspace.py`. No product code changed. Each failure mode below was reproduced at base 52b3207ff1, and each matches the error recorded in earlier dev runs.

**Failure modes (AC#1)**
- `test_file_notes_navigation_and_key_guidance_use_one_phrase`: `run_test` re-raised `WorkerFailed: NoMatches('#file-notes-path-label')`. The body ended after one `pilot.pause()` while the workspace's `_initialize` worker was still in its `_build_runtime` thread. Teardown removed the workspace's children before its Unmount cancelled the worker. The worker resumed in that gap, and `_update_controls()` queried a removed control. Fix: wait for `workspace.initialized` before asserting.
- `test_initial_root_scan_projects_checking_authority_while_actions_are_gated`: `assert 'Checking' in 'Chec...otes'`. `_update_root_surface` paints the full summary, then `call_after_refresh(_fit_root_status)` elides it to the status's width. At 60x20 that width is 11 cells, because the root row always shows "Review recovered pairing…" (27 cells, added by b5251e9a6e). The test passed only when it read the row before the deferred fit ran. No wait can fix this: after the fit settles, the painted row never contains "Checking". Fix: assert on `_root_status_summary`, the copy the row is fitted from (the sibling linked-root test already reads `_root_status_detail`).
- `test_create_move_delete_protect_and_restore_use_real_service`: `run_test` re-raised `WorkerFailed: MountError("Can't mount widget(s) before ListView(id='file-notes-git-rows') is mounted")`. The last wait was "the restored file exists on disk". That is true as soon as `restore_file` returns, while `_restore_file` is still rescanning (`_rescan_after_action` -> `_apply_scan` -> `_refresh_session_changes` -> git `mark_stale` -> `_replace_rows`). `TldwCli._shutdown` sets `app._exit = True` before it drains app-owned lifecycles, and Textual's `is_attached` is False under `_exit`. So the row-render worker's `extend()` raised during that drain. A diagnostic plugin caught five failures. In all five, `_replace_rows` came from `_render_scan_state` after `_exit` was set, and the one full stack captured names `_restore_file`. Fix: wait until the restore reopens the file and `_replacing_rows` clears. The fixed `pilot.pause()` before pressing Restore is now a wait for Restore to be displayed and enabled (a press on a hidden or disabled button is a silent no-op).

**Negative controls (AC#3).** A scratch script (not committed) applies one temporary edit per test to either form, then reverts it with `git checkout`:
- NC1: `_build_runtime` holds its finished result until teardown has removed the path label (2 s cap).
- NC2: the scan thread reports its start 0.3 s late.
- NC3: the restore's rescan starts when `app._exit` is set (1 s cap), and the app's File Notes shutdown drain is 0.3 s slower.

Delaying only the producer's start did not reproduce NC1 or NC3: Textual cancels a node's workers at its Unmount. Results, 10 serial runs per form:
| Test | Old form | New form |
|---|---|---|
| navigation | 5 of 10 failed, each the recorded NoMatches | 10 of 10 passed |
| root scan | 10 of 10 failed, `'Checking' in 'Chec...otes'` | 10 of 10 passed |
| create/move/restore | 5 of 10 failed, each the recorded MountError | 10 of 10 passed |

Under the one-window idle probe with 12 parallel loops, base failed test 3 at rates from 1/48 to 4/8. The new form passed 60 of 60 under the same probe and load (40 runs of test 3, 10 each of tests 1 and 2).

**AC#2.** 20 of 20 serial single-node runs, each a separate invocation through the scratch wrapper, for each test. The whole file with `-n 4` gave 152 passed and 10 failed. At base, the same command gave 151 passed and 11 failed: the same 10, plus test 3. Those 10 are outside this task.

The file is not in `scripts/ui_pr_gate_census.txt` or any CI lane (see the TASK-33370 note in `Tests/conftest.py`). Preflight passes. The Ruff lint count is unchanged from base (24), and the file is format-clean.

Files: `Tests/UI/test_library_file_notes_workspace.py`, `backlog/docs/lessons-testing-evidence.md` (new entry on teardown WorkerFailed and holding the producer's result).

