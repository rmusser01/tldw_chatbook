---
id: TASK-38646
title: Folder Files capability test asserts Restore is enabled before Delete's path transition ends
status: Done
assignee:
- '@claude'
created_date: 2026-11-02 21:52
updated_date: 2026-11-03 00:49
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
dependencies:
- TASK-36383
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_capability_inventory_and_modes` failed 2 of 6 serial runs on dev. Each run was its own pytest call through the scratch-profile wrapper. Both failures were at :1886 with `restore.disabled=True`. That is the same rate TASK-36383's notes recorded on 2026-10-24, and TASK-36386 noted this test too.

**Cause.** The test waits only for `_selected_deleted_path == "working.md"` (:1880), then asserts Restore is shown and enabled (:1886). Delete sets that field inside `_hold_path_transition()` before it awaits the rescan (library_file_notes_workspace.py:8927, :8935). Restore stays disabled while the transition runs (:5770, :5843-5845), and the transition's `finally` re-enables it (:6141-6150). TASK-36383 fixed this race in the file's sibling tests by waiting for a pressable Restore, but not here.

No PR lane runs the file: it has no UI census entry, and Nightly Deep last ran on 2026-09-27. So far it only fails local whole-file runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes 10 of 10 serial runs on dev, each its own pytest process, because it waits for a shown and enabled Restore
- [x] #2 It still fails if Restore stays hidden or disabled after Delete settles (shown under a temporary source edit)
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Test-only synchronization fix; no product or architectural change.

1. Show the red at dev: serial single-node runs of the test through the scratch wrapper.
2. Replace the immediate `restore.display and not restore.disabled` assert after Delete with TASK-36383's pattern: `_wait_until` a shown and enabled Restore before pressing it. No fixed pause.
3. AC#1: 10 serial single-node runs, each its own pytest process.
4. AC#2: temporary source edit that keeps Restore disabled after Delete settles; the test must fail on the new wait, then revert.
5. Whole file with `-n 4` against dev's failures; preflight.

## Implementation Notes

This change touches only the test, in `Tests/UI/test_library_file_notes_workspace.py`. The immediate `assert restore.display and not restore.disabled` after Delete is now TASK-36383's wait: `_wait_until(pilot, lambda: restore.display and not restore.disabled, "Restore was not offered after Delete")`. It has no fixed pause, and product code is unchanged. Before Delete, Restore is hidden. During Delete's path transition it is shown but disabled. So the wait can only pass once Delete has settled.

Why only one window races: `pilot.pause()` first runs `_wait_for_screen`, which waits for each widget's queue. The Delete handler holds the workspace's queue. Any pause that starts before the bubbled Pressed message reaches the workspace can overlap the handler, and in practice that is usually only the first pause after the press. The old assert failed only when the delete thread set `_selected_deleted_path` inside that first ~20 ms window.

**Red at dev (base 2dcf038cb8).** 2 of 10 serial runs failed, each its own pytest process through the scratch wrapper. Both failed at :1886 with `restore.disabled=True`. A pinned control added `asyncio.sleep(0.5)` before Delete's rescan, inside the transition. Under it, the old form failed 2 of 3 runs and the new form passed 5 of 5. The edit was reverted.

**AC#1.** The new form passed 10 of 10 serial runs, each its own pytest process. Load averages were 30 to 42.

**AC#2.** Two temporary source edits, each reverted with `git checkout`:
- Restore always disabled in `_update_controls`: the test failed at the new wait (:1889, "Restore was not offered after Delete").
- Restore never displayed in the visibility map: same failure.

**Whole file with `-n 4`.**
- Dev's test file: 152 passed, 10 failed.
- This branch, run twice: 151 passed and 11 failed each time.
- Both branch runs include dev's same 10 ids, and this test passes in both.
- Each branch run had one different extra failure, in a test this change does not touch:
  - `test_folder_files_shared_shell_retains_state_across_breakpoints`: `_push_phase` was 'idle' at :757.
  - `test_conflict_resolution_discard_keeps_cancel_first_confirmation`: Cancel did not have focus at :5971.
- Each extra passed 5 of 5 serial runs on the branch. The review then ran each one alone, on code byte-identical to dev, because the changed test is not collected.
  - `test_conflict_resolution_discard_keeps_cancel_first_confirmation` failed 2 of 6 alone at :5971, so it is pre-existing on dev. It asserts the Cancel focus right after the confirmation shows, but `_set_reload_confirmation` focuses through `call_later`.
  - `test_folder_files_shared_shell_retains_state_across_breakpoints` did not reproduce in 27 review runs; it is a rare load flake whose dev status is unproven.
  - This change cannot cause either: it touches only this test's body. Both are proposed as follow-ups.

**Checks.** Preflight passes. Ruff format is clean, and the Ruff lint count is unchanged from base (24).

Files: `Tests/UI/test_library_file_notes_workspace.py`.
