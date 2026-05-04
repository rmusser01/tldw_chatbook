---
id: TASK-5.5
title: Phase 4.5 Adopt W+C services in W+C destination
status: Done
assignee: []
created_date: '2026-05-04 07:55'
labels:
  - unified-shell
  - phase-4
  - watchlists
  - collections
  - service-adoption
dependencies: []
parent_task_id: TASK-5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the top-level W+C destination use existing Watchlists and Collections services so users can see whether local monitored sources and saved collection items exist and stage concrete context into Console instead of only reading static explanatory copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 W+C route lists a local snapshot from Watchlists and Collections services when available
- [x] #2 W+C route shows honest loading empty service-unavailable and service-error recovery states
- [x] #3 Stage W+C Context in Console stages concrete watchlist and collection summaries rather than generic placeholder copy
- [x] #4 Existing W+C active-run Console follow and legacy Watchlists route remain intact
- [x] #5 Focused automated tests cover W+C available empty error handoff existing navigation and tracking behavior
- [x] #6 QA evidence documents functional behavior visual usability residual risks and verification output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing destination shell tests for W+C local snapshot available empty error and concrete Console handoff behavior.
2. Implement the smallest `WatchlistsCollectionsScreen` local snapshot loader using `watchlist_scope_service.list_watch_items()` and `media_reading_scope_service.list_read_it_later()`.
3. Render loading available empty and error states with stable selectors while preserving the existing `Open current Watchlists` route and active W+C run follow controls.
4. Disable `Stage W+C Context in Console` when no concrete local W+C context exists or services are unavailable.
5. Build `ChatHandoffPayload` body and metadata from actual listed watchlist and collection records.
6. Add Phase 4.5 QA evidence and roadmap links.
7. Run focused W+C destination and live-work tests plus git diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added local W+C snapshot loading in `WatchlistsCollectionsScreen` through `watchlist_scope_service.list_watch_items()` and `media_reading_scope_service.list_read_it_later()`.
- Rendered loading, available, empty, service-unavailable, and service-error states while preserving the existing `Open current Watchlists` route and latest active W+C run Console follow controls.
- Added `Stage W+C Context in Console` as a disabled-until-ready `wc-context` Chat handoff with concrete local watchlist and collection counts plus sample titles.
- Cached the active-run Console follow item per screen instance so asynchronous W+C snapshot refreshes do not change the run promised by the visible follow button.
- Added focused destination shell and Console handoff regression coverage plus Phase 4.5 QA evidence and roadmap links.
- Verified with `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py Tests/Subscriptions/test_watchlist_scope_service.py::test_scope_service_routes_local_and_server_actions_with_watchlists_action_ids Tests/Media/test_media_reading_scope_service.py::test_scope_service_list_read_it_later_normalizes_local_saved_state -q` resulting in `116 passed, 8 warnings in 89.31s`.
- Addressed PR review feedback by converting the W+C snapshot loader to a native async Textual worker, preserving safe comparison titles with `<` and `>` while rejecting dangerous text, and making transient Home active-work adapter failures retryable without repeatedly logging the same failure.
- Added review regression coverage for Console-follow recovery after an initial adapter failure and for safe W+C title handling through both visible UI and staged Console handoff payloads.
- Re-verified after PR review fixes with `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py Tests/Subscriptions/test_watchlist_scope_service.py::test_scope_service_routes_local_and_server_actions_with_watchlists_action_ids Tests/Media/test_media_reading_scope_service.py::test_scope_service_list_read_it_later_normalizes_local_saved_state -q` resulting in `118 passed, 8 warnings in 91.23s`.
<!-- SECTION:NOTES:END -->
