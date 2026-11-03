---
id: TASK-36525
title: The Settings category sweep is red on dev and runs on no PR lane
status: Done
assignee:
- '@claude'
updated_date: '2026-11-03 22:41'
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- settings
priority: medium
references:
- Tests/UI/test_settings_category_sweep.py
- Tests/UI/test_settings_theme_picker_screen.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-36381
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36381's review (branch fu/h-settingsreds, 2026-10-23), checked on dev 5b5920b8e2.

**The category sweep.** `Tests/UI/test_settings_category_sweep.py::test_every_settings_category_renders_and_restores_focus` fails at both sizes. It is not in the UI census, so no PR run reports it.
- Run plainly, it fails in the call phase with `RecoveryRequired: raw_source_selection_changed` (`raw_participants.py:307`). This is the trap TASK-36381 explains: the test builds app config on the per-test path.
- Under a temporary `@private_profile_test` (since reverted), it fails on a stale expectation: "Expected focus on settings-category-theme, got ThemeOptionList(id='settings-theme-list') at (80, 24)". A rail click on Theme focuses the theme list on purpose; `test_settings_theme_picker_screen.py::test_rail_click_and_open_theme_focus_the_theme_list` pins that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_every_settings_category_renders_and_restores_focus passes on dev at both sizes, and its Theme expectation matches the Theme picker's focus rule
- [x] #2 test_settings_category_sweep.py runs on a PR-required lane
<!-- AC:END -->

## Implementation Plan

1. Reproduce at base: the sweep plainly (`-n 4`), then under a temporary `@private_profile_test` to see every category that is red once RecoveryRequired is out of the way (the brief says Theme; check the others too).
2. The test builds app config in its body (`_build_test_app()` -> `load_settings()`), the class TASK-36381 settled with `@private_profile_test`. To pass on a PR lane it keeps that marker. The guard itself and TASK-38111's setup-time cases are out of scope.
3. Theme: a rail press on Theme focuses `#settings-theme-list` on purpose (pinned by `test_rail_click_and_open_theme_focus_the_theme_list`). The sweep expects that for Theme and the category's own rail button elsewhere. Fix any other red the same way only if the product rule says so; report a product defect instead of rewriting its expectation.
4. AC#2: measure the file's runtime; append it (whole, or as node ids if over the 30 s per-file budget) at the census tail and set `MINIMUM_FILES` to the entry count. Census checker, `Tests/CI` census/shard tests, the file with `-n 4`, preflight.

## Implementation Notes

Test-only; no product code changed. Base dev eba3ad14b5.

**What was red** (both sizes, `-n 4` through the scratch-profile wrapper):
- Plain: 2 failed in the call phase with `RecoveryRequired: raw_source_selection_changed` (`raw_participants.py:307`), from the body's `_build_test_app()` -> `load_settings()` under the per-test config redirect.
- Under a temporary `@private_profile_test` with a diagnostic that recorded every category's focus instead of stopping at the first: only Theme was red, at both sizes (focus on `ThemeOptionList(id='settings-theme-list')`). The other 32 rail categories each focused their own rail button, and every category kept the Settings title and content.

**Fix.**
- The test now runs under `@private_profile_test` (with `request`). It builds app config in its body, the case TASK-36381 settled this way (AC#3 there: the guard is intended and plain tests that build config are wrong). This is not TASK-38111's setup-time import path, and nothing in the profile guard or the shared conftests changed. `bootstrap_profile` was not used: the sweep mounts every Settings pane against the shared session profile, and TASK-36381 recorded app-config leaks between tests on that profile.
- The focus expectation now follows the product rule: a rail press on Theme lands in `#settings-theme-list` (TASK-33072, `settings_screen.py` queues `_focus_theme_list` behind the rail restore; pinned by `test_rail_click_and_open_theme_focus_the_theme_list`). Every other category still expects its own rail button. `_FOCUS_AFTER_SELECT` names the exception.

**Red/green.** Red at base as above. Green: `-n 4` 2 passed; 10 of 10 serial single-node runs (5 per size). Negative control: with `_after_category_panes(self._focus_theme_list)` removed from `settings_screen.py` (temporary, reverted), both sizes failed with "Expected focus on settings-theme-list, got Button(id='settings-category-theme')".

**AC#2, runtime and lane (first pass).** Serially, each whole-size node took 28-31 s (call) at load average 22-24 on 18 CPUs, 33-53 s at 33-43, and once 95 s at 47. The call time includes the private-profile child's start-up. The census tail first listed the two size nodes. The follow-up round below replaced them.

**Follow-up round (2026-11-03, after review of the first pass).**
- **Press, not click.** The shared `_click_settings_category` helper scrolled the rail and clicked the row by coordinate. It now calls `Button.press()` on the rail button, waits until that category is active and its pane swap is done, and then settles. If a press opens nothing (Textual silently ignores a press on a hidden or disabled row), the helper fails and names the category. The sweep also waits for the focus it asserts before asserting it. The helper keeps its name because 21 other test files import it. Their 486 tests run with `-n 4`: 330 passed. The other 156 fail identically with the old click helper: 152 are plain tests that hit `RecoveryRequired`, and 4 have unrelated causes (two `SimpleNamespace.app_config` errors and two search-index drift assertions). All 107 census entries from those files passed. The helper's old `try/except: pass` is gone, and ruff now reports no findings in the file.
- **Split.** A whole size was over the 30 s per-entry budget, so the test is parametrized over (group, size). Each group is a contiguous third of the rail, so every rail-to-rail switch except the two group boundaries is one the unsplit sweep made. A new pure test, `test_the_sweep_groups_cover_every_category_once`, asserts that the groups in order are exactly the rail and that none is empty. A temporary off-by-one in the group size (reverted) turned it red ("Right contains 3 more items").
- **Per-node time,** serial, one invocation per node, call phase: 11-23 s at load average 16-35 over three rounds (18 runs, all passed). The same nodes took 27-44 s while the machine's load spiked to 41-69 from other jobs. In the whole-file serial run at load 53-62, they took 15-47 s. A third is not much cheaper than the whole rail under load, because a node's fixed cost (child start-up plus app boot, about 8-10 s) is most of it.
- **Negative control, re-run on the split file:** with Theme's list focus removed from `settings_screen.py` (temporary, reverted), the two group-0 nodes, which hold Theme, failed at both sizes ("Expected focus on settings-theme-list, got Button(id='settings-category-theme')"). The other four nodes and the coverage test passed.
- **Census:** the two size nodes are replaced at the tail by the six (group, size) nodes and the coverage test. `MINIMUM_FILES` is the entry count. Round-robin puts each of the seven on its own shard, and `--collect-only` over each of those shards collected without errors.
- **Runs:** the file with `-n 4`: 7 passed; serially: 7 passed; census checker OK; `Tests/CI` census, shard and workflow tests: 423 passed; preflight passed. Ruff: clean and format-clean.

**Not verified.** The nodes' CI time.

Files: `Tests/UI/test_settings_category_sweep.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
