---
id: TASK-2304
title: Rail counts and the scoped sources table tell the truth
status: To Do
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
