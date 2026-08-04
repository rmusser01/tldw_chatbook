---
id: TASK-2304
title: Rail counts and the scoped sources table tell the truth
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

UAT: the rail counts stayed frozen at 0 across create → assign → check while
the centre header simultaneously read "(1 source)" — two counts of the same
fact disagreeing in one frame. Separately, with scope "AI Research News
(0 sources)" the sources table still listed an Unassigned source: the table
ignores the scope the header claims (or the header counts a filter the table
does not apply). What the rail number even counts is not discoverable.

UAT findings F15 (high), F16 (high).

## Acceptance Criteria (the what)

- [ ] Rail counts update when sources/watchlist membership change (create,
      assign, remove, delete) without requiring a tab switch or restart.
- [ ] The sources table and the header agree: selecting a watchlist scope
      shows exactly that watchlist's sources, and the header count matches
      the visible rows.
- [ ] What the rail count counts is visually self-evident or labeled.
- [ ] Regression tests cover count-updates-on-assign and scope-filtering.

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
