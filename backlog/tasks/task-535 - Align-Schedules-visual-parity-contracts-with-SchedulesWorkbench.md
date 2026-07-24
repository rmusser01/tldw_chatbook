---
id: TASK-535
title: Align Schedules visual-parity contracts with SchedulesWorkbench
status: To Do
assignee: []
created_date: '2026-07-24 20:09'
labels:
  - ui
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful Schedules visual-parity coverage by replacing retired SchedulesScreen selectors, loading hooks, and action assumptions with the active SchedulesWorkbench.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Schedules pane-title and operational-workbench tests mount `SchedulesWorkbench` and use its current selectors.
- [ ] Empty, blocked, and loading-state coverage verifies states that the active Schedules destination can actually render.
- [ ] Compact-size and keyboard-order coverage reaches the current `Follow in Console` action.
- [ ] All seven Schedules cases listed below pass without reintroducing the retired `SchedulesScreen`.
- [ ] Ruff lint, formatting, and diff-integrity checks pass for changed files.

## Observed Failures

The 2026-07-24 full replay of `Tests/UI/test_destination_visual_parity_correction.py` found these Schedules failures:

- `test_destination_pane_titles_are_user_facing_not_ordinal[schedules-#schedules-workbench-expected_titles2]`: waits for retired `#schedules-workbench`.
- `test_schedules_screen_matches_approved_control_plane_columns`: waits for retired `#schedules-empty-state`.
- `test_operational_destinations_use_timing_or_procedure_workbench[schedules-#schedules-filter-strip-#schedules-workbench-panes0-actions0]`: targets retired filter/workbench selectors.
- `test_operational_empty_or_blocked_states_preserve_workbench_geometry[schedules-#schedules-filter-strip-#schedules-workbench-panes0-actions0-markers0-#schedules-detail-pane]`: targets retired state and pane selectors.
- `test_operational_loading_states_preserve_workbench_geometry[schedules-SchedulesScreen-_refresh_latest_console_context-#schedules-loading-state-#schedules-detail-pane-#schedules-filter-strip-#schedules-workbench-panes0-actions0]`: patches the retired screen and loading hook.
- `test_top_level_destinations_keep_primary_workbench_visible_at_compact_size[schedules-contract6]`: waits for retired workbench selectors.
- `test_tab_order_reaches_visible_primary_action[schedules-targets6]`: waits for retired workbench selectors before reaching the current action.

ADR required: no
ADR path: N/A
Reason: This task updates visual regression coverage to the already-active Schedules workbench and does not change architecture.
