---
id: TASK-36525
title: The Settings category sweep is red on dev and runs on no PR lane
status: To Do
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
- [ ] #1 test_every_settings_category_renders_and_restores_focus passes on dev at both sizes, and its Theme expectation matches the Theme picker's focus rule
- [ ] #2 test_settings_category_sweep.py runs on a PR-required lane
<!-- AC:END -->
