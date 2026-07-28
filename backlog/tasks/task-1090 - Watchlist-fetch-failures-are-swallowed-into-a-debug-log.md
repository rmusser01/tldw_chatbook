---
id: TASK-1090
title: >-
  A failed watchlist fetch is swallowed into a debug log the user never sees
status: Done
assignee: []
created_date: '2026-07-28 08:00'
labels:
  - watchlists
  - bug
  - observability
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_check_now_source` wraps the whole fetch in `except Exception`, logs at **debug**, and shows a transient toast. Nothing durable records that a check failed, and `subscriptions.last_error` is only written by the service on paths that get that far.

**This is the swallow that hid TASK-1100.** Check now was raising `ValueError` on every single press — the entire feature was dead — and the only evidence was a debug line nobody reads and a toast that had vanished before anyone looked. Three UAT runs and a full test suite reported the screen as working while it fetched nothing.

The same shape appears throughout this screen: `except Exception: logger.opt(exception=True).debug(...)` around a service call whose failure the user needs to know about. A fetch is the one operation in Watchlists that *routinely* fails for ordinary reasons — the feed moved, the host is down, the XML is malformed, the network is out — so it is exactly the operation that must report.

AC #4 of TASK-1100 was left unchecked for this reason; it belongs here rather than folded into that fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed check writes `subscriptions.last_error` and surfaces it in the Sources table's Status column
- [x] #2 The failure is visible after the toast has gone — the user can find out why without repeating the action
- [x] #3 An unexpected exception in the fetch path logs at `warning` or above, not `debug`
- [x] #4 A run that fails is recorded in `local_watchlist_runs` with its error, not silently absent
- [x] #5 A test makes the fetch raise and asserts the user-visible outcome, proven to fail against current code
- [x] #6 The other `except Exception: ... .debug(...)` handlers on this screen are audited, and any that hide a user-facing failure are listed here or fixed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**A worse failure than the swallow, found while fixing it.** A check that *ran* and failed did not raise at all: `LocalWatchlistsService.execute_run` catches the fetch error, records a `failed` run and **returns** it. So the screen's `try` succeeded and it told the user **"Check now started."** over a feed that had just 404'd. The swallow hid dead code; this reported success over a real failure. `_check_now_source` now inspects the returned run status (`_check_failure_message`) and reports the actual reason.

**AC#1 had two halves and only one existed.** `execute_run` already wrote `subscriptions.last_error` via `record_check_error`. Nothing surfaced it: `SourcesPane._source_row_cells` read `source.get("status")`, a key **no** watchlists normalizer emits — they publish `status_summary` (`active`, `inactive`, `error (3)`). The Status column read `-` for every source in every state, and `Last scraped` did the same (`last_checked_or_scraped_at`). The Status **filter** shared the bug, so filtering to `Error` — the one thing that filter is for — always returned nothing, and the Overview's "Sources in error" card was a permanent zero for the same reason. All four now resolve `status_summary` with the bare `status` kept as a fallback for hand-built rows. `_check_now_source` also reloads the source list so the column carries the outcome once the toast has gone (AC#2).

**AC#3.** The fetch path logs at `warning` with the source id and the exception message, and the toast carries the reason rather than a generic "Failed to check source."

**AC#4.** `execute_run` guarded only the fetch. Anything that escaped it — a subscription deleted between launch and execution, a service fault, TASK-1100's namespaced-id `ValueError` — left the run row inserted a moment earlier sitting at `queued` forever with no error on it. `LocalWatchlistsService.record_run_failure` was extracted from `execute_run`'s own `except` branch (it writes both the failed run and `last_error`), and `WatchlistScopeService.launch_run` now calls it before re-raising.

**AC#6 — audit of the 30 `except Exception: ... .debug(...)` handlers on this screen.**

*Fixed (promoted to `warning`, 15).* Everything behind a control the user pressed, where a swallowed failure means the button did nothing and nothing said so: `_start_tree_write` (×2), `_run_tree_write`, `_create_source`, `_cancel_run`, `_rerun_run`, `_preview_source`, `_on_opml_import_complete`, `_export_opml`, `_update_item_status`, `_save_rule`, `_delete_source`, `_delete_run`, `_delete_rule`, `_delete_item` (whose message also said "Failed to ignore item"). `_update_item_status` is not hypothetical — TASK-1120 proved it had been raising `NotImplementedError` on every press. The set is pinned by a parametrized contract test so a new handler cannot quietly join it.

*Left at `debug`, deliberately (15).* The load/refresh coroutines — `_load_sources`, `_load_runs`, `_load_items`, `_load_rules`, `_load_notifications` (×2), `_refresh_overview_data`, `_load_tree_data`, `_load_source_rows_for_tree`, `scoped_source_rows`, `_resolve_breadcrumb_labels`, `_refresh_feeds_region_for_scope`, `_list_local_wc_snapshot` (×2), `_apply_layout`. These are background reads whose failure is already visible as an empty region plus a "Failed to load ..." toast, `_list_local_wc_snapshot` has its own recovery state, and `_apply_layout`'s is not a failure at all ("workbench not mounted yet"). Promoting them would make an offline session log a wall of warnings and devalue the level for the 15 above.

**Files:** `UI/Screens/watchlists_collections_screen.py`, `UI/Watchlists_Modules/sources_pane.py`, `UI/Watchlists_Modules/watchlists_backend_controller.py`, `Subscriptions/local_watchlists_service.py`, `Subscriptions/watchlist_scope_service.py`, `Tests/UI/test_watchlists_check_now_failure.py` (new).
<!-- SECTION:NOTES:END -->
