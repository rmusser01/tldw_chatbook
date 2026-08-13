---
id: TASK-15775
title: Watchlists screen reactive default disagrees with the shipped first-run region layout
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - perf
  - watchlists
priority: low
---

## Description

Found and measured during task-15462's profiling investigation (input-latency
burn-down), deliberately not shipped there because the cost/benefit did not
justify the risk of reordering config-writing code. `WatchlistsCollectionsScreen`'s
reactive default is `region_layout = RegionLayout()` (nothing collapsed),
while the shipped first-run default is `_FIRST_RUN_DEFAULT =
RegionLayout(collapsed={RIGHT_RAIL})`. Every single screen visit therefore
composes the expanded Inspector rail (13 widgets), and `on_mount`'s
`_apply_layout(load_region_layout())` immediately tears it down and mounts
the one-line collapsed header in its place.

Task-15462 measured this precisely: 13 widgets discarded, 1 mounted, ~5-10 ms
of a ~450 ms push (1-2%). A prototype (seeding `region_layout` from
`load_region_layout()` before compose, rather than after) was built and
verified to remove the swap entirely — `_apply_layout` sees `equal=True`,
zero `_swap_region_widget` calls, identical final widget counts — but a
paired A/B could not reliably detect the wall-clock difference (6/12 pairs
faster, median delta -1 ms). Not shipped because `load_region_layout()`
performs a one-time synchronous migration write, and moving it into screen
construction changes its ordering relative to `_last_persisted_collapsed`
priming, which `_schedule_layout_persist`'s no-op guard depends on — a real
migration-ordering risk task-15462 chose not to take inside a profiling task.

## Acceptance Criteria

- [ ] `WatchlistsCollectionsScreen`'s `region_layout` reactive default agrees
      with (or is seeded from) the persisted/first-run layout before compose,
      so a normal visit does not compose-then-discard the Inspector rail
- [ ] `load_region_layout()`'s one-time migration write and
      `_last_persisted_collapsed` priming are proven correctly ordered
      relative to the earlier construction point (task-15462's flagged risk),
      with a test that would fail if the ordering regressed
- [ ] `_apply_layout` performs zero `_swap_region_widget` calls on a normal
      screen visit (regression test, mirroring task-15462's prototype
      verification)
- [ ] Existing Watchlists screen-navigation and layout-persistence suites
      stay green
