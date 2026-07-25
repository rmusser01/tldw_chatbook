---
id: TASK-535
title: Align Schedules visual-parity contracts with SchedulesWorkbench
status: Done
assignee: []
created_date: '2026-07-24 20:09'
updated_date: '2026-07-24 20:32'
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
<!-- AC:BEGIN -->
- [x] #1 Schedules pane-title and operational-workbench tests mount `SchedulesWorkbench` and use its current selectors.
- [x] #2 Empty, blocked, and loading-state coverage verifies states that the active Schedules destination can actually render.
- [x] #3 Compact-size and keyboard-order coverage reaches the current `Follow in Console` action.
- [x] #4 All seven Schedules cases listed below pass without reintroducing the retired `SchedulesScreen`.
- [x] #5 Ruff lint, formatting, and diff-integrity checks pass for changed files.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map each stale Schedules visual contract to the active SchedulesWorkbench selectors, rendered states, and Follow in Console action.
2. Replace the retired screen-only loading hook with a deterministic state owned by the active workbench and preserve meaningful pane geometry assertions.
3. Update compact-layout and keyboard-order coverage to current controls without weakening shared helpers.
4. Run the seven focused cases, the full visual-parity module, and Ruff/diff-integrity checks.
5. Request independent review, resolve actionable findings, and document the ADR decision (no ADR: test alignment within the existing UI architecture).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced every retired Schedules screen selector and loading hook in the visual-parity suite with the active `SchedulesWorkbench` sync strip, queue/detail/inspector panes, empty state, task table, and `Follow in Console` action.
- Added semantic pane-title classes and made Console follow disabled until async context loading establishes an available target.
- Loaded the production stylesheet for Schedules geometry checks and constrained its tabbed content to the remaining screen height, preventing the workbench from extending beneath the footer.
- Added a responsive three-pane layout at 120 columns and below. Exact 120/121 resize coverage proves the class toggles while queue, detail, inspector, and the Console action remain reachable.
- Loading coverage now proves the initial worker starts and is cancelled during test teardown; keyboard coverage injects a real active-work item and focuses the enabled Console action.
- Verification: 7 focused Schedules cases, all 83 visual-parity cases, 44 adjacent Schedules workbench/shell cases, and 5 CSS-integrity cases pass. Ruff lint, Ruff formatting, and `git diff --check` pass.
- Independent review identified and verified fixes for loading-worker lifecycle coverage, the responsive breakpoint transition, and Backlog marker placement; final review approved with no remaining actionable findings.
- ADR required: no. These are localized Schedules layout/interaction corrections and regression-test alignment within the existing destination architecture.
<!-- SECTION:NOTES:END -->

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
