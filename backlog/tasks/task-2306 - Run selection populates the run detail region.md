---
id: TASK-2306
title: Run selection populates the run detail region
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: clicking a run row (and click+Enter) never populates "Run detail" — it
stays "No run selected", leaving the detail/Items/Logs sub-regions
unreachable. Dead interaction on the primary object of the tab.

UAT finding F34 (high).

## Acceptance Criteria (the what)

- [x] Selecting a run row (mouse and keyboard) populates Run detail, its
      Items list and Logs.
- [x] A regression test drives selection through the real table and asserts
      the detail region updates.
- [x] Verified live in a real terminal.

## Implementation Plan

1. Diagnose the selection wiring end to end (click -> `RowHighlighted` ->
   `select_run_by_id` -> `selected_run` -> `RunSelected` -> screen handler ->
   detail region) and record which link is actually broken.
2. Fix the render half in `RunsPane`: `selected_run` is deliberately not
   `recompose=True`, so its watcher must push the detail stats into the live
   `#runs-detail-stats` the same way `_update_selection_highlight` patches the
   table, rather than relying on a compose pass that never happens again.
3. Fix the data half: give the screen a run-detail loader that resolves the
   selected run's items and log text and pushes them into the mounted pane
   (`_dom_is_live`-guarded, stale-result-discarding), since nothing in the
   product ever wrote `RunsPane.run_items` / `run_logs`.
4. Add the read path the loader needs: filter local items by `run_id` through
   the existing `items.list` route, and carry `alert_count` on normalized
   items so the Items table's Alerts column has a real source.
5. Convert `run_items` / `run_logs` to in-place pushes so selecting a run does
   not rebuild `#runs-table` under the user's cursor.
6. Render every run/item-derived string inert (`Text`, no markup).
7. Regression tests: drive selection through the real `DataTable` (mouse
   click and Enter) and assert the mounted detail widgets update; a
   mount-window test for the new loader push.
8. Verification gates + live run in a real terminal.

## Implementation Notes

The selection wiring was never broken. The UAT click reached the table and
moved the highlight; two other things were wrong, and each alone was enough to
produce the reported dead region.

### 1. Nothing repainted the detail block

`RunsPane.selected_run` is deliberately not `recompose=True` (a pane recompose
rebuilds `#runs-table` — the very table the click just moved a cursor in — and
remounts it unfocused, which `highlight_is_user_driven` then reads as a
non-user highlight). Its watcher moved the row highlight and armed the toolbar
but never touched `#runs-detail-stats`, so that `Static` kept whatever the
FIRST `compose()` wrote: `No run selected.`, forever.

`watch_selected_run` now pushes the stats in place, exactly like
`_update_selection_highlight` does for the row.

### 2. Nothing ever produced the data

`RunsPane.run_items` / `run_logs` had **no writer anywhere in the product** —
grep found the pane, and the pane's own unit test. So Items and Logs were
structurally empty in the running app whatever was selected, and the existing
detail test passed only because setting those two `recompose=True` reactives
forced a compose pass that happened to re-read `selected_run`.

`WatchlistsCollectionsScreen._load_run_detail` now answers each selection:
items from the run, log text off the run record (`normalize_watchlist_run`
already carried `log_text`, so no second query), pushed into the mounted pane
under `_dom_is_live` with a stale-result guard on the run id. Both reactives
became plain (in-place) for the reason in (1). The result is mirrored on the
screen (`_run_detail_items`/`_run_detail_logs`) so a workbench rebuild
re-seeds a fresh pane, and a screen-level `watch_selected_run` drops the
mirror the moment the selection moves — the three paths that clear
`selected_run` (`_apply_tree_scope`, the backend switch, `_delete_run`) never
go near the loader, and two of them then call `_reseed_live_detail_pane`.

### The read path was new; the storage was not

`subscription_items.run_id` and `idx_subscription_items_run_id` have existed
since the column was added and **nothing had ever queried them**.
`get_new_items` gains a `run_id` predicate (the same fragment shape TASK-2301
introduced) threaded through `list_items` on the local service and the scope
service, so it rides the existing `items.list` policy action rather than
inventing one. `normalize_watchlist_item` now carries `run_id` and
`alert_count` — the Alerts column had no possible source before this and drew
`0` over every item however many content-alert rules had fired.

Items are a local-backend read (`WatchlistScopeService.list_items` refuses the
server backend outright), so a server run gets its stats and log and an honest
empty item list rather than a "Failed to load" toast for a route that does not
exist.

### Two things the first round of tests did not actually pin

Both found by mutation, not by reading:

* **The deep-link `await`.** `RunsPane` posts `RunSelected` only
  `if self.is_mounted`, and `_load_runs` is started by `on_mount` — inside the
  window where that is still False (TASK-2200). The first test passed with the
  `await` removed because its pane WAS mounted; it now reconstructs the window
  (`pane._is_mounted = False`) and fails without it.
* **The screen-side mirror clear.** With the pane mounted, its own
  `RunSelected(None)` reaches the loader and cleans up, so the mutation
  survived. The test now clears the selection from a DIFFERENT tab — where no
  pane exists to self-correct — and the stale items would otherwise be seeded
  into the next `RunsPane` built.

### Inert rendering

`DataTable.default_cell_formatter` runs `Text.from_markup` over any plain `str`
cell, and a run item's title is a feed entry's own `<title>`. Item rows and the
detail block are built as `Text`.

### Verification

* `Tests/UI/test_watchlists_run_detail.py` (new, 9 tests) — real clicks and
  real cursor keys through the production screen, a workbench-rebuild test, a
  reconstructed mount-window test, a deep-link test and an inertness test.
* `Tests/Watchlists/test_watchlists_runs_pane.py` +3,
  `Tests/Subscriptions/test_local_watchlists_service.py` +2.
* Suites: `Tests/Subscriptions/` + `Tests/Watchlists/` + `Tests/DB/
  test_subscriptions_db_watchlists.py` + `Tests/Home/test_active_work_adapter.py`
  + `Tests/RuntimePolicy/test_runtime_policy_core.py` **1071 passed**;
  `test_watchlists_destination_shell.py` **71 passed**; the watchlists UI
  sweep **96 passed**; poisoned order (`test_watchlists_content_pane.py` + the
  create-form e2e, one invocation) **50 passed**; `--collect-only Tests/UI
  Tests/Watchlists` **8699 collected**, no errors.
* **12 mutations**, each reverted individually → RED → restored byte-exact
  (md5-verified, `git status --short` unchanged between).

One pinned assertion was updated on purpose:
`Tests/Watchlists/test_watchlist_scope_service.py` pinned the exact delegation
kwargs, which `run_id` joins.

### Live verification (fresh profile, real `https://hnrss.org/frontpage`)

Watchlist + source created through the UI, assigned, checked. Clicking the run
row:

```
Source: Hacker News
Watchlists: Morning read
Status: completed
Started: 2026-08-04T23:50:29.582049+00:00
Duration: 490ms
Found: 20 | Processed: 20 | Filtered: 0 | Errors: 0
Items   Title                                                         Status  Alerts
        libexpat now funded by the City of Munich for up to 6 months  new     0
        ... (20 rows)
Logs    Local watchlist execution completed with 20 item(s).
```

Keyboard: `Down` moved to the previous run and the whole detail followed
(`Started: …23:49:47`, `Duration: 453ms`, Items empty — that run's item rows
were re-claimed by the later check — `Logs` showing its own line); `Up`
brought the newest run's 20 items back.

### Files

* `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py`
* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
* `tldw_chatbook/Subscriptions/watchlist_normalizers.py`,
  `local_watchlists_service.py`, `watchlist_scope_service.py`
* `tldw_chatbook/DB/Subscriptions_DB.py`
