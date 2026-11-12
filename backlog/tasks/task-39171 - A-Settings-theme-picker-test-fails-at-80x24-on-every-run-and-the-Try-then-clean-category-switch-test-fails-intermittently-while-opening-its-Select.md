---
id: TASK-39171
title: "A Settings theme-picker test fails at 80x24 on every run, and the Try-then-clean category switch test fails intermittently while opening its Select"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:36
labels:
- follow-up
- tests
- settings
- ui
priority: low
references:
- Tests/UI/test_settings_theme_picker_screen.py
dependencies:
- TASK-39135
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39135's covering runs, round-12 branch fu/r12-settings: they "fail the same way on `fu-base-r12`"). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08, and re-checked on dev 3a6498ae14, 2026-11-09 (run.sh). Both tests are `@private_profile_test`, so they run plain. Line numbers are dev 3a6498ae14's.

**`test_highlighted_launch_and_override_status_is_on_screen[size0]`** (80x24, `:2468`). It fails on every run.
- The private-profile child fails an `AssertionError` on the 80x24 screen text; `[size1]` and `[size2]` pass. TASK-39135's notes call it "(row width)".
- Red in every run: 3 of 3 runs that list failures by name on dev 3a6498ae14 (`-n 4`, load 23-37); five earlier runs of the same four nodes each had at least one failure (load 8-14); once each on dev b590a15afd and the stack; and every run in review r12fr.

**`test_try_then_clean_category_switch_restores_the_pre_editor_theme`** (`:1869`). It fails intermittently.
- Observed rates:
  - dev 3a6498ae14, alone and serial (`-p no:xdist`): 5 of 20 failed at load 14-30 (5 of the first 10, none of the next 10);
  - dev 3a6498ae14, beside the three `[size*]` cases at `-n 4`: 2 of 3 runs that list failures by name (load 23-37); five earlier runs failed 1, 2, 1, 1 and 1 of the four nodes (load 8-14), so it failed in at most one of them;
  - 2026-11-08: failed in its one run on dev b590a15afd and in its one run on the stack (load ~21-27);
  - review r12fr: 0 of 4 alone on dev 3a6498ae14, 0 of 2 on the stack.
- The failure is the private-profile child's `NoMatches` while it opens a Select:
  - on dev b590a15afd: "No nodes match '#label' on SelectCurrent(classes='-has-value')";
  - on the stack: "No nodes match 'SelectOverlay' on Select(id='settings-appearance-density', …)", from `select_overlay = self.query_one(SelectOverlay)`.

**Lanes.** Three other nodes of this file are UI census entries; these two are not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_highlighted_launch_and_override_status_is_on_screen[size0]` passes on dev, or is retired with the commit that changed what it pinned named; its red is shown first with the same command
- [x] #2 It is classified as test drift or a product defect (a status row that does not fit at 80x24 would be one); a product defect gets a test that is red before its fix
- [x] #3 The cause of `test_try_then_clean_category_switch_restores_the_pre_editor_theme`'s intermittent `NoMatches` is named from its own failure, and the test then passes 20 of 20 runs alone, with load recorded, at least ten of them at a one-minute load of 20 or more
- [x] #4 Both nodes run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Show `[size0]` red with the filed command (`-n 4`, the two nodes), dump what 80x24 paints, and find which row pushed the status line off.
2. Classify: a status row that no longer fits is a product defect; fix it in the compact theme-list CSS (rebuild the bundle), with `[size0]` as the red-before test.
3. Reproduce the try-then-clean `NoMatches` under load and name the press or wait it races from its own traceback; wait on that end condition, bounded and named.
4. 20 runs of the try-then-clean node alone with load recorded (ten or more at load >= 20).
5. Append both nodes at the UI census tail with measured serial times, raise MINIMUM_FILES, replay recent shard logs with the entries charged, and record three timings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base ae9f9e8d21, 2026-11-10. Every run below went through run.sh; loads are one-minute averages before and after each run.

**`[size0]` (AC#1, AC#2): a product defect, fixed.** Red first with the filed command (`-n 4`, the three `[size*]` cases and the try-then-clean node): `[size0]` failed `assert "launch" in painted and "overrides built-in" in painted` (load 22-34); `[size1]` and `[size2]` passed. The row itself fit (`len(row) <= row_width()`); "(row width)" in TASK-39135's notes was not the cause. A temporary dump of the painted 80x24 screen showed the status line (`#settings-theme-row-status`, "launch default · overrides built-in") at y=21 with a visible height of 0: the compact detail body shows 14 rows, and title + Filter (3) + the 9-row list used all of them. 177a2936c7 (Plugins browser, committed 2026-10-31) added the "Plugins — installation and marketplaces" button row to the Settings header, which took the 15th row the line used to sit on. Control: with that button temporarily hidden, `[size0]` passed and the line sat at y=20. So a status row that does not fit at 80x24 is a product defect, and `[size0]` is its red-before test. Fix: the compact theme list's `max-height` is 8 rows, not 9 (`components/_settings_splash_theme.tcss`, bundle rebuilt with `build_css.py`); the list still shows at least 5 rows, as the test asserts. After: the 80x24 and compact covering set (all three `[size*]` cases, try-then-clean, every picker/editor reachability case at 80x24-190x55, the 80x24 one-row tests, `test_settings_theme_picker.py`, and the Settings sweep's 80x24 groups) passed 82 of 82 (load 55-65). `test_boot_css_byte_budget.py`, `test_design_token_governance.py` and `test_timer_path_layout_cost.py` pass. `test_css_build_integrity.py::test_library_notes_compact_source_module_is_exactly_bundled` fails on the clean base worktree `fu-base-r13` too, so it is pre-existing. The User Guide (`settings.md`) already says the line under the list stays visible; no copy changed.

**Try-then-clean (AC#3).** Red before: 5 of 8 runs alone (`-n 0`, four concurrent copies) at load 41-59, with both filed messages: "No nodes match '#label' on SelectCurrent(classes='-has-value')" and "No nodes match 'SelectOverlay' on Select(id='settings-appearance-density', …)". Its own traceback names the cause: the error is re-raised at `run_test`'s exit, from `Select._on_mount` -> `_init_selected_option` -> `_watch_value`, so a Select was still being mounted when the app was torn down. A temporary probe at the end of the test body found `host._exception` still None in all 8 runs, the `_swap_category_panes` worker still running, and the Appearance Selects not yet mounted (or not yet created). So the test ended, after a fixed `pilot.pause(0.3)`, while the category switch's pane swap was still mounting the Appearance pane; teardown pruned the half-mounted Select's children mid-Mount. It is not a product defect: swaps are serialized by a lock and a revision, never cancelled mid-mount. The test now waits (bounded, `_wait_for` from `test_settings_category_sweep.py`) for `active_category == appearance and not _category_pane_swap_pending`, the flag the swap clears in its `finally`, and fails "the Appearance category swap (_swap_category_panes) never finished". Negative control: the flag's clear temporarily skipped for Appearance -> the test failed with exactly that message. After: 20 of 20 runs alone (`-n 0`), load 49.6-88.0, so all 20 at load 20 or more.

**Lane (AC#4).** Both nodes are appended at the UI census tail, one entry each, and `MINIMUM_FILES` is 1915 (the measured count). Three serial timings (`-p no:xdist`, both nodes): `[size0]` 18.9, 38.5 and 29.1 s; try-then-clean 31.2, 46.4 and 38.3 s (load 44-71). Replayed over the 22 cached PR runs the census header names (`scripts/ui_lane_shard_replay.py --shards 36`, each node charged at its slowest timing): the two shards they land on go from 11.5 to 12.1 and 12.2 min, and the maximum stays 12.3 min, under the header's 13.3. No rebalance was needed. `Tests/CI` census, replay and gate tests: 372 passed. Not the fast PR contract.

**Files.** `tldw_chatbook/css/components/_settings_splash_theme.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated), `Tests/UI/test_settings_theme_picker_screen.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
