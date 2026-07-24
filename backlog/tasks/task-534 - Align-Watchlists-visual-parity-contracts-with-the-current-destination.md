---
id: TASK-534
title: Align Watchlists visual-parity contracts with the current destination
status: Done
assignee: []
created_date: '2026-07-24 20:09'
updated_date: '2026-07-24 20:21'
labels:
  - ui
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful Watchlists visual-parity coverage by replacing retired filter-strip and compact-action assumptions with the current Watchlists destination structure and actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists workbench, empty-state, and loading-state geometry tests use selectors that exist in the active destination.
- [x] #2 The Watchlists backend header occupies exactly the three rows required by its Select instead of expanding into workbench space.
- [x] #3 The approved control-plane copy assertion describes the current Watchlists filter and action surface.
- [x] #4 Compact-size coverage reaches a visible current Watchlists primary action.
- [x] #5 All five Watchlists cases listed below pass without weakening shared geometry helpers.
- [x] #6 Ruff lint, formatting, and diff-integrity checks pass for changed files.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map the current Watchlists filter, list/detail/inspector panes, loading marker, and compact primary actions.
2. Correct any live geometry defect exposed by those contracts and rebuild generated CSS when required.
3. Update the five recorded parameterized contracts and copy assertions to current controls.
4. Run the five focused cases, the full visual-parity module, and static checks.
5. Request independent review and resolve actionable findings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated Watchlists visual contracts to the current backend header, list/detail/inspector panes, list-owned state markers, navigator actions, and current control-plane copy.
- Added a Watchlists-only visual harness that loads the production stylesheet, keeping shared non-visual destination tests unchanged.
- Fixed `#watchlists-header-bar` from `height: auto` (18 rendered rows) to the three rows required by Textual’s Select and rebuilt the generated CSS bundle.
- Added an exact three-row assertion and preserved the shared geometry helpers without weakening their thresholds.
- Six Watchlists visual cases and 11 Watchlists/CSS-integrity tests pass. The full visual module now has 76 passes and only the seven TASK-535 Schedules cases remain.
- Ruff lint, Ruff formatting, and `git diff --check` pass.
- Independent review approved the Select sizing, harness isolation, selector mappings, and action coverage with no actionable findings.
- ADR required: no. This is a localized layout correction and visual-test alignment within the existing destination.
<!-- SECTION:NOTES:END -->

## Observed Failures

The 2026-07-24 full replay of `Tests/UI/test_destination_visual_parity_correction.py` found these Watchlists failures:

- `test_source_prep_destinations_use_list_detail_inspector_workbench[watchlists_collections-contract2]`: expects retired `#watchlists-filter-strip`.
- `test_source_prep_default_empty_or_unavailable_states_preserve_workbench_geometry[watchlists_collections-contract2]`: expects retired `#watchlists-filter-strip`.
- `test_watchlists_screen_matches_approved_control_plane_columns`: expected control-plane copy no longer matches the current destination.
- `test_source_prep_loading_states_preserve_workbench_geometry[watchlists_collections-WatchlistsCollectionsScreen-_refresh_local_wc_snapshot-#wc-loading-state-contract1-#watchlists-detail-pane]`: expects retired `#watchlists-filter-strip`.
- `test_top_level_destinations_keep_primary_workbench_visible_at_compact_size[watchlists_collections-contract5]`: expects a retired compact primary-action contract.

ADR required: no
ADR path: N/A
Reason: This task updates visual regression coverage to the already-active Watchlists destination and does not change architecture.
