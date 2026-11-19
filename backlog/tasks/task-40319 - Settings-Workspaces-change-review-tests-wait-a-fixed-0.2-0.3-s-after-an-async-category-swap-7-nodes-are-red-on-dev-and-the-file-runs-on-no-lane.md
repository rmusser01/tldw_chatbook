---
id: TASK-40319
title: "Settings Workspaces change-review tests wait a fixed 0.2-0.3 s after an async category swap: 7 nodes are red on dev and the file runs on no lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 03:09
labels:
- follow-up
- tests
- settings
- workspaces
- flaky
priority: low
references:
- Tests/UI/test_settings_workspaces_category.py
- tldw_chatbook/UI/Screens/settings_screen.py
dependencies:
- TASK-39169
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream review r13p-rev1: red at base ae9f9e8d21 and at the cprod head). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Each change-review step presses a row that starts an async same-category swap (`_swap_category_panes` under `_category_swap_lock`, since 4fe45538f5), then waits a fixed `pause(0.2)` or `pause(0.3)` and asserts. Under load the swap has not finished. `test_change_review_unavailable_consent_has_no_toggle` was flaky at base already (TASK-39169 notes).

**Nodes** (`Tests/UI/test_settings_workspaces_category.py`): the six `test_change_review_*` tests and `test_folder_bindings_add_toggle_remove_and_inline_errors`. Fixed pauses at `:392-427`, `:548`, `:559`, `:563`, `:584`, `:591`, `:623`, `:667`, `:673`, `:709`, `:737`, `:765`. The file runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each change-review and folder-binding step waits for the swap and the state it asserts within a deadline instead of a fixed pause
- [x] #2 The file passes 3 of 3 whole-file runs at `-n 4`, and runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Re-run the file at -n 4 on base 26b2e677dd and read each failure.
2. Replace every fixed `pilot.pause(0.2/0.3)` after a row press or a change-review/folder action with a bounded wait: first for the category swap to finish (the screen's `_category_pane_swap_pending` flag, cleared by `_swap_category_panes`), then for the asserted state, under a deadline.
3. Append the file to the UI census tail, raise `MINIMUM_FILES`, and record its local serial time.
4. Run the whole file 3 times at -n 4.

## Implementation Notes

Every fixed `pilot.pause(0.2/0.3)` that followed a row press, a change-review toggle/retry or a folder action now calls `_until(pilot, screen, ready)`: it polls (0.05 s) until `screen._category_pane_swap_pending` is False (the flag `_swap_category_panes` clears when the async same-category swap finishes) and `ready()` holds, under a 10 s deadline. At the deadline it falls through, so the test's own assertion reports what is missing. The swap flag alone is not enough: `WorkspaceChangeReviewPanel` paints its rows in its own `on_mount`, after the pane's `recompose()` returns, so each wait also checks the state that step asserts (the copy, the registry value, or the control). The flag is set synchronously in the handler that writes the registry, so a registry-based predicate cannot pass before the swap starts.

Same pattern, outside the named nodes, because AC #2 needs the whole file 3 of 3: `test_default_workspace_card_is_protected` and `test_pane_refreshes_after_external_registry_change` (a fixed pause after a swap), and `test_create_rename_archive_unarchive_flow`. Its Set active step waited for `#settings-workspace-archive`, which the old card already had, so it pressed a button the swap then removed (`NoMatches`, seen once in this task's first run). It now waits for the active workspace and the swap; the archived-row-gone loop uses the same wait. The two `pause(0.2)` calls left in `test_resume_refreshes_workspaces_pane_only_when_active` count a monkeypatched synchronous refresh, so they are not swap waits.

**Evidence.** Base 26b2e677dd, whole file at `-n 4` (load about 50): 8 failed (the seven `test_change_review_*` tests and `test_folder_bindings_add_toggle_remove_and_inline_errors`), each on the first assertion after a fixed pause. After: 20 passed in 3 of 3 whole-file runs at `-n 4` (load 65-72; 247 s, 245 s, 259 s).

**Lane.** The file's 20 tests are appended to `scripts/ui_pr_gate_census.txt`, one entry each (the whole file is far over the per-entry budget), and `MINIMUM_FILES` is the measured count. Local serial times with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`), load 55-67 on 18 CPUs, 703.6 s in all: activation_failure 43.1, archived_card_only_unarchive 32.3, failed_root_retry 33.9, git_absent 27.3, global_kill 24.8, preparing 30.5, stale_toggle 32.2, toggle_flips 29.3, unavailable_consent 31.2, compact_overview 37.6, create_rename_archive_unarchive_flow 55.5, default_card_protected 35.7, folder_bindings 42.7, overview_recovery_copy 30.0, external_registry_change 36.5, resume_refreshes 26.3, scope_inspector 29.4, archive_receipt_undo 33.9, archived_label_restore_name 29.8, category_registered 40.5.

Files: `Tests/UI/test_settings_workspaces_category.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
