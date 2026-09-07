---
id: TASK-2304
title: Rail counts and the scoped sources table tell the truth
status: Done
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

UAT: the rail counts stayed frozen at 0 across create → assign → check while
the centre header simultaneously read "(1 source)" — two counts of the same
fact disagreeing in one frame. Separately, with scope "AI Research News
(0 sources)" the sources table still listed an Unassigned source: the table
ignores the scope the header claims (or the header counts a filter the table
does not apply). What the rail number even counts is not discoverable.

UAT findings F15 (high), F16 (high).

## Acceptance Criteria (the what)

- [x] Rail counts update when sources/watchlist membership change (create,
      assign, remove, delete) without requiring a tab switch or restart.
- [x] The sources table and the header agree: selecting a watchlist scope
      shows exactly that watchlist's sources, and the header count matches
      the visible rows.
- [x] What the rail count counts is visually self-evident or labeled.
- [x] Regression tests cover count-updates-on-assign and scope-filtering.

## Implementation Plan

1. Establish what the two numbers actually are: the rail's number is the
   UNREAD ITEM count per bucket (`get_watchlist_item_counts`), the centre
   header's is the SOURCE count for the tree scope
   (`_staging_summary_line`). They never described the same fact.
2. AC#1: find the write paths that change item counts but never reload the
   rail (`Check now`, the item-status writes, ignore-item) and route a
   `_load_tree_data()` through them -- which already publishes via TASK-2200's
   `_request_surface_refresh` drain, so no new recompose is introduced.
3. AC#2: `_load_sources` lists every source regardless of scope. Filter the
   loaded rows to the scope's own source ids (`scoped_source_rows()`, the
   same resolver the header counts) and re-push on scope change, in place.
4. AC#3: label the rail number as unread items -- in the rail's own copy, not
   only in a tooltip -- so what it counts is readable without hovering.
5. Regression tests: counts refresh after a check/assign; the scoped table
   and the header agree; the rail's label is present.

## Implementation Notes

The two numbers never described the same fact. The rail's is the UNREAD ITEM
count per bucket (`SubscriptionsDB.get_watchlist_item_counts`); the centre
header's is the SOURCE count for the current tree scope
(`_staging_summary_line`). Nothing on screen said so, so a rail reading 0 next
to a header reading "(1 source)" read as a contradiction -- and the rail
genuinely was stale as well, so both halves needed work.

**AC#3.** The rail carries a one-row legend, `Counts: unread items`, above the
roots, and every node's tooltip says it too including the zero case ("No
unread items") -- which is the reading the UAT could not resolve. A legend
rather than a per-node suffix ("All sources  0 unread"): the rail's interior
is 26 columns, a suffix costs 7 of them on EVERY node, and an overflowing
label renders with an ellipsis that
`test_watchlists_left_rail_is_labelled_when_expanded` rightly fails on. One
row labels every node at once and cannot truncate.

**AC#1.** `Check now` is the one gesture that manufactures items and it never
reloaded the tree, so the counts sat on 0 while a feed's worth of items
arrived in the centre. `_check_now_source`, `_update_item_status`'s refresh
tail and `_delete_item` now call `_load_tree_data()`, which publishes through
TASK-2200's `_request_surface_refresh` drain -- a rail rebuild, not a screen
recompose. The silent mark-read-on-open path (`refresh=False`) is deliberately
excluded and the trade-off is stated at the call site: it fires on every arrow
key and carries no reload at all by design, so two SQLite queries and a rail
rebuild per keystroke is the wrong price for a number that is one out. It is
corrected by the next deliberate action or a tab switch.

**AC#2.** `_load_sources` lists every source with no scope predicate at all,
so the table ignored the tree entirely. `scoped_loaded_sources()` narrows each
push through `scoped_source_rows()` -- the SAME resolver the header counts --
so the two cannot drift by construction; making the table agree by
re-deriving the scope some other way would just create a third answer. The
`all` scope short-circuits before paying for that query (it is the default,
and its answer is "everything" regardless), and `_loaded_sources` itself stays
UNSCOPED: it is the mirror the Console handoff and pane rebuilds read, and
narrowing it would make the scope sticky where nothing asked.

Four paths can re-widen the table, and all four are covered: the reload, the
pane rebuild (`_build_detail_pane`), a scope change (`watch_tree_scope`), and
-- **found in live verification, not by the suite** -- a MEMBERSHIP change.
Adding a source to the watchlist already in view moves no scope and triggers
no source reload, so scoping the table left `Add source` writing a membership
row while the table stayed empty under a header that had already updated to
"(1 source)": the disagreement this task exists to remove, reintroduced in the
opposite direction by its own fix. `_apply_tree_data_to_live_surfaces` -- which
runs after every membership write -- now pushes the scoped rows too, as a
third in-place ITEMS-region push alongside the Overview and Artifacts ones
(a single reactive assignment, never a rebuild, so an in-flight create form is
not even queried).

### Verification

* New file `Tests/UI/test_watchlists_rail_counts_and_scope.py` (11 tests).
* Mutation-verified: 6 mutations, each reverted individually -> RED ->
  restored byte-exact (md5).
* Gates: `Tests/Watchlists/` + `test_watchlists_destination_shell.py` +
  `test_watchlists_inspector.py` + `test_watchlists_source_row_click_selects.py`
  + `test_watchlists_check_now_failure.py` +
  `test_watchlists_overview_loading_state.py` **530 passed**;
  `test_destination_visual_parity_correction.py` +
  `test_watchlists_source_create_form.py` + `test_destination_shells.py` +
  `test_destination_headers.py` **252 passed, 1 skipped**.

### Live verification (235x52, fresh profile, real HN feed)

```
scope = AI Research News, before assign
  header  Local Watchlists snapshot: AI Research News (0 sources)
  table   (no rows)                 <- the UAT saw an Unassigned source here

after Add source (scope unchanged)
  header  Local Watchlists snapshot: AI Research News (1 source)
  table   HN Front Page  rss  active

rail before Check now         rail 3s after Check now
  Counts: unread items          Counts: unread items
   All sources  0                All sources  20
   Unassigned  0                 Unassigned  0
   AI Research News  0           AI Research News  20

after Ingest        All sources 18 / AI Research News 18
```

### Files

* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
  `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`,
  `tldw_chatbook/css/features/_watchlists.tcss` (+ the regenerated bundle).
* `Tests/UI/test_watchlists_rail_counts_and_scope.py` (new).
