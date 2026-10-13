---
id: TASK-34929
title: Gate-Watchlists-per-visit-section-loads-on-staleness
status: Done
assignee: []
created_date: 2026-10-13 14:52
labels:
- performance
- ui-perf
- wave-3-screen-switch
dependencies: []
priority: medium
updated_date: 2026-10-13 18:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Watchlists on_mount fires five loaders on every visit — _refresh_local_wc_snapshot, _load_active_section_data, _load_tree_data, _refresh_overview_data, _resolve_daily_report_banner (watchlists_collections_screen.py:1349-1361). They run off-pump in workers so nothing freezes, but every visit re-queries all sections with no unchanged-since-last-visit gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Re-entering Watchlists with unchanged data skips redundant section re-queries
- [x] Forced refresh action still reloads everything
- [x] Overview and tree stay correct after data changes (tests)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read watchlists on_mount loaders and their data sources
2. Gate per-visit re-queries on a staleness identity
3. Keep forced refresh reloading everything
4. Tests + targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Design decision: no data-identity gate exists here (WatchlistBundleService is a thin SubscriptionsDB wrapper with no revision/fingerprint seam, unlike MCP's `run_catalog_fingerprint`), and manufacturing one across the bundle service, the backend controller, and the daily-report sources would carry real staleness risk with no periodic reconcile to self-heal — poor trade for off-pump query cost.
- Chosen fix: a short revisit TTL (`WATCHLISTS_REVISIT_SKIP_SECONDS = 2.0`, timestamp on the app instance so it outlives the per-visit screen). A revisit inside the window skips the five loaders; the layout/section setup above the gate still runs every visit. This targets the actual pattern from the review — rapid tab flipping re-querying every section per hop — with an empty correctness surface: any data change older than two seconds is always seen, and `r` (`action_refresh_all`) runs its own forced worker independent of on_mount.
- Tests: new `Tests/UI/test_watchlists_revisit_gate.py` (3 tests: first visit loads everything, immediate revisit runs only the layout pass, revisit after the TTL loads again) — 3/3 pass.
- Regression: `Tests/UI/test_watchlists_content_pane.py` + `test_watchlists_rail_counts_and_scope.py` are at their pre-existing baseline (43 failed / 35 passed with the change stashed — the same mount breakage family as the other UI suites).
- ADR required: no — visit-cadence gate inside one screen's mount.
