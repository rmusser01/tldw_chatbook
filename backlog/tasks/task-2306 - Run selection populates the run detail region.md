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

- [ ] Selecting a run row (mouse and keyboard) populates Run detail, its
      Items list and Logs.
- [ ] A regression test drives selection through the real table and asserts
      the detail region updates.
- [ ] Verified live in a real terminal.

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
