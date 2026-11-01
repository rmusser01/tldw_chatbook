---
id: TASK-34967
title: Gate-Watchlists-per-visit-section-loads-on-staleness
status: To Do
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
- [ ] Re-entering Watchlists with unchanged data skips redundant section re-queries (two attempts reverted; see notes)
- [x] Forced refresh action still reloads everything
- [ ] Overview and tree stay correct after data changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read watchlists on_mount loaders and their data sources
2. Gate per-visit re-queries on a staleness identity
3. Keep forced refresh reloading everything
4. Tests + targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- SECOND REVERT (2026-10-14): the TTL read-cache (attempt 2) broke two P0 admission tests (`test_watchlists_bulk_source_authoring.py::test_a_load_landing_while_a_section_pane_mounts_reaches_that_pane` and `::test_a_load_landing_while_a_rail_is_recreated_reaches_it`). Root cause: those tests pin a load/mount ORDERING contract — a load landing while a pane mounts must reach that pane through the watcher publish path. The cache made revisit loads land BEFORE the pane mounted (instantly, from the app-level cache), so the pane mounted after the reactive write and stayed in LOADING. Any future attempt must preserve the publish timing relative to pane mounts (e.g., defer applying a cached result until the destination pane exists), which is a publish-path change, not a cache-in-front change.
- Attempt 2 design (reverted in db3d6df48c): loaders unchanged, TTL read-cache in front of `_list_local_wc_snapshot` and `get_overview_data`. Kept the correctness surface empty for STATE (loaders always ran) but not for TIMING.



- Second attempt (2026-10-14, follow-up batch): a TTL read-cache instead of the reverted mount gate. `_list_local_wc_snapshot_cached` and `_overview_data_cached` sit IN FRONT of the two whole-screen reads and serve a fresh (<2s) app-level cached result; the loaders themselves — and every state write they perform — run unchanged on every visit. This keeps the correctness surface empty by construction (the first attempt's failure was skipping the loaders on a fresh instance), while rapid revisits stop re-issuing the identical page query and overview aggregation.
- Scope: the tree loader's five batched reads (verified efficient in the wave-1 audit) and the per-section loaders (only the ACTIVE section's branch runs) are left uncached; `r` / sync completions refresh through their own paths and naturally age the cache out.
- Tests: `Tests/UI/test_watchlists_read_cache.py` (3 tests: cache-hit within TTL, re-query after TTL with fresh result cached, overview cache is backend-keyed). 3/3 pass; `test_watchlists_revisit_gate.py` pins the every-visit-load contract from the revert.
- ADR required: no — read-side cache with a TTL inside one screen's loaders.


- REVERTED after independent review (PR #3083): the 2s revisit-TTL gate skipped the five mount loads on a rapid revisit, but this route is NOT reusable — every visit constructs a fresh screen with empty data state (`_wc_loaded = False`), so the skip left the screen in a permanent "Loading local Watchlists snapshot..." state with no timeout fallback. Exactly the rapid-tab-flip scenario the task targeted became a broken-looking screen. The gate and its constant are removed; every visit loads again.
- What a correct fix needs (re-scope for whoever picks this up): a data-carrying snapshot cache — cache the last completed snapshot/tree/overview payloads on the app instance and SEED the fresh screen's reactives from that cache when the gate trips — or service-level snapshot caching so the per-visit loads are cheap. A bare TTL on the app instance cannot work when the rendered state lives on a per-visit instance.
- The original investigation stands: WatchlistBundleService is a thin SubscriptionsDB wrapper with no revision/fingerprint seam, and manufacturing one across the bundle service, backend controller, and daily-report sources carries real staleness risk with no periodic reconcile.


- Design decision: no data-identity gate exists here (WatchlistBundleService is a thin SubscriptionsDB wrapper with no revision/fingerprint seam, unlike MCP's `run_catalog_fingerprint`), and manufacturing one across the bundle service, the backend controller, and the daily-report sources would carry real staleness risk with no periodic reconcile to self-heal — poor trade for off-pump query cost.
- Chosen fix: a short revisit TTL (`WATCHLISTS_REVISIT_SKIP_SECONDS = 2.0`, timestamp on the app instance so it outlives the per-visit screen). A revisit inside the window skips the five loaders; the layout/section setup above the gate still runs every visit. This targets the actual pattern from the review — rapid tab flipping re-querying every section per hop — with an empty correctness surface: any data change older than two seconds is always seen, and `r` (`action_refresh_all`) runs its own forced worker independent of on_mount.
- Tests: new `Tests/UI/test_watchlists_revisit_gate.py` (3 tests: first visit loads everything, immediate revisit runs only the layout pass, revisit after the TTL loads again) — 3/3 pass.
- Regression: `Tests/UI/test_watchlists_content_pane.py` + `test_watchlists_rail_counts_and_scope.py` are at their pre-existing baseline (43 failed / 35 passed with the change stashed — the same mount breakage family as the other UI suites).
- ADR required: no — visit-cadence gate inside one screen's mount.
