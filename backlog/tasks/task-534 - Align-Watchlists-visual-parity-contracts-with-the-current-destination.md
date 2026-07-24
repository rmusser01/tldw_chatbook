---
id: TASK-534
title: Align Watchlists visual-parity contracts with the current destination
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
Restore meaningful Watchlists visual-parity coverage by replacing retired filter-strip and compact-action assumptions with the current Watchlists destination structure and actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Watchlists workbench, empty-state, and loading-state geometry tests use selectors that exist in the active destination.
- [ ] The approved control-plane copy assertion describes the current Watchlists filter and action surface.
- [ ] Compact-size coverage reaches a visible current Watchlists primary action.
- [ ] All five Watchlists cases listed below pass without weakening shared geometry helpers.
- [ ] Ruff lint, formatting, and diff-integrity checks pass for changed files.

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
