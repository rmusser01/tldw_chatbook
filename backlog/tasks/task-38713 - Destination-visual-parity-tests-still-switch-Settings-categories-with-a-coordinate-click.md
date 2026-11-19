---
id: TASK-38713
title: "Destination visual-parity tests still switch Settings categories with a coordinate click"
status: Done
assignee: ['@claude']
created_date: 2026-11-04 00:59
updated_date: 2026-11-04 06:31
labels:
- follow-up
- tests
- settings
- flaky
priority: low
references:
- Tests/UI/test_destination_visual_parity_correction.py
- Tests/UI/test_settings_category_sweep.py
dependencies:
- TASK-36525
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36525's notes). Checked on dev c2cdb6b82e, 2026-11-03.

TASK-36525 changed the shared `_click_settings_category` helper in `Tests/UI/test_settings_category_sweep.py` (:83-103), which 20 test files import. It used to click the rail row by coordinate. Now it calls `Button.press()` on the rail button, waits until that category is active and its pane swap is done, and fails with the category's name if nothing opened. The reason is in Textual 8.2.8: `Button._on_click` presses only when the button lacks the `-active` class (`textual/widgets/_button.py:416-419`), and `press()` sets that class for `active_effect_duration = 0.2` s (:370, :444-447). A coordinate click that lands in that window is dropped silently, and the test goes on asserting against the category that was already open.

`Tests/UI/test_destination_visual_parity_correction.py` does not import the shared helper. It defines its own `_click_settings_category` (:411-423): it scrolls the row into view inside a bare `try/except: pass` and then calls `pilot.click(selector)`. Two tests use it:
- `test_settings_dirty_category_status_has_visual_marker_class` (:2598; click at :2605);
- `test_settings_advanced_config_controls_use_action_and_status_rows` (:2628; click at :2634).

Both can therefore assert against the wrong Settings pane when a click is dropped. No PR lane runs this file (no census entry, no workflow target), so a flake here would go unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both parity tests open the Settings category they name in every run; a switch that opens nothing fails the test and names the category, instead of the test asserting against the previous pane
- [x] #2 Both tests pass 5 of 5 runs with -n 4 on a loaded host, with the load recorded in the task notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the file-local coordinate-click `_click_settings_category` in `Tests/UI/test_destination_visual_parity_correction.py` with the shared press helper from `Tests/UI/test_settings_category_sweep.py` (press, wait for the category and its pane swap, fail naming the category).
2. Show a switch that opens nothing now fails naming the category (temporary negative control), then run both tests 5 times with -n 4 and record the load.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The file-local `_click_settings_category` (scroll in a bare `try/except: pass`, then `pilot.click(selector)`) is gone. Both tests import the shared helper from `Tests/UI/test_settings_category_sweep.py` and call `_settle_settings(pilot)` then `_click_settings_category(pilot, "<category>")`, as its other importers do. The helper presses the rail button, waits until that category is active and its pane swap is done, and asserts `Pressing the <category> rail row did not open it`.

Evidence (base b8608e4cdb; both tests are plain tests, so locally they hit `RecoveryRequired: raw_source_selection_changed` from TASK-38111; every run below was under a temporary `@private_profile_test` marker that was then removed):
- Negative control: with each test's rail row set `disabled = True` just before the switch (a press that opens nothing), both failed with `AssertionError: Pressing the console-behavior rail row did not open it` and `... the advanced-config rail row ...`. The old helper had no such check: a dropped click left the test asserting against the previous pane.
- 5 of 5 runs green with `-n 4` (2 passed each, 15-20 s), load averages 20.2-24.4 (1 min) while another agent's 4-worker sweep ran on the same 18-CPU host.

No PR lane runs this file (unchanged; the criteria ask for none).

Files: `Tests/UI/test_destination_visual_parity_correction.py`.
<!-- SECTION:NOTES:END -->
