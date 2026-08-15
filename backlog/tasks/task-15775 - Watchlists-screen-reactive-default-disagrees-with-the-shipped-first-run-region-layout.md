---
id: TASK-15775
title: Watchlists screen reactive default disagrees with the shipped first-run region layout
status: Done
assignee:
  - '@claude'
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

- [x] `WatchlistsCollectionsScreen`'s `region_layout` reactive default agrees
      with (or is seeded from) the persisted/first-run layout before compose,
      so a normal visit does not compose-then-discard the Inspector rail
- [x] `load_region_layout()`'s one-time migration write and
      `_last_persisted_collapsed` priming are proven correctly ordered
      relative to the earlier construction point (task-15462's flagged risk),
      with a test that would fail if the ordering regressed
- [x] `_apply_layout` performs zero `_swap_region_widget` calls on a normal
      screen visit (regression test, mirroring task-15462's prototype
      verification)
- [x] Existing Watchlists screen-navigation and layout-persistence suites
      stay green

## Implementation Plan

1. Re-read `WatchlistsCollectionsScreen` at current HEAD (dev has moved past
   task-15462; confirm `region_layout`, `on_mount`, `_apply_layout`,
   `_schedule_layout_persist` and `WatchlistsWorkbench`'s scoped-rebuild
   machinery from task-15461/15476/15478 are unchanged in shape from the
   task's description).
2. Prefer "derive from the same source of truth" per the owner's
   durable-over-quick ruling: seed `region_layout` (via `set_reactive`,
   mirroring `WatchlistsWorkbench.__init__`'s own pattern) from a SINGLE
   `load_region_layout()` call made in `__init__`, before `compose_content`
   ever runs. Prime `_last_persisted_collapsed` from the SAME call's result,
   on the next line, so the two cannot drift apart (closes task-15462's
   flagged ordering risk without introducing a second call).
3. Update `on_mount` to reuse `self.region_layout` instead of calling
   `load_region_layout()` a second time -- `_apply_layout` stays, now a
   true no-op reconciliation pass (Textual's reactive skips the watcher on
   an unchanged value, so this is zero `_swap_region_widget` calls on a
   normal visit).
4. Write regression tests: construction-time ordering (AC#2, no mount
   needed), a real end-to-end pilot test proving zero swaps on a genuine
   fresh config (AC#1+#3), a mocked-value pilot test for the shipped
   first-run default and for an explicit non-default user choice (AC#3,
   "must not pin the default harder"), and a no-redundant-persist check
   (AC#2 continued).
5. Born-red every new test against the pre-fix code (file-swap, not git
   checkout, to avoid discarding uncommitted work), confirm the churn
   assertions fail, then restore the fix and confirm green.
6. Run the broader Watchlists suite (screen, region_layout, region_layout_store,
   scoped_rebuilds, workbench, destination_shell, rail_counts, content_pane,
   inspector, run_detail, items_pane, pagination) to confirm no regressions
   (AC#4).
7. ruff check + format on touched files only; update the task file with
   Implementation Notes and mark Done.

## Implementation Notes

**Disagreement mechanism.** `WatchlistsCollectionsScreen.region_layout`'s
class-level reactive default was `reactive(RegionLayout())` (nothing
collapsed). `region_layout_store._FIRST_RUN_DEFAULT` (the value
`load_region_layout()` returns on a never-saved config) is
`RegionLayout(collapsed={RIGHT_RAIL})`. `compose_content` reads
`self.region_layout` to build the initial `WatchlistsWorkbench`, and
compose always runs before the `Mount` event, so every cold open composed
the workbench with RIGHT_RAIL fully expanded (13 widgets). `on_mount` then
called `load_region_layout()` and `_apply_layout(...)`, pushing the loaded
layout into the already-mounted workbench; `WatchlistsWorkbench.watch_region_layout`
saw RIGHT_RAIL's collapse state change and called `_swap_region_widget`,
tearing the just-built Inspector pane down for a one-line collapsed
header — a guaranteed compose-then-discard on every fresh install.

**Fix shape: derive, not just agree.** Per the owner's durable-over-quick
ruling, `__init__` now calls `load_region_layout()` exactly once, seeds
`region_layout` from that SAME result via `set_reactive` (mirroring
`WatchlistsWorkbench.__init__`'s own seeding pattern) before
`compose_content` ever runs, and primes `_last_persisted_collapsed` from
the identical call's result on the next line — so the two values cannot
drift apart (closing the ordering risk task-15462 flagged). `on_mount`
reuses `self.region_layout` instead of calling `load_region_layout()` a
second time; `_apply_layout` still runs there, now a true no-op
reconciliation (Textual's reactive skips the watcher on an unchanged
value), so `WatchlistsWorkbench._swap_region_widget` is called zero times
on a normal visit. `Tests/Watchlists/test_watchlists_cold_open_layout.py`
covers both the construction-time ordering and the DOM/swap-count
evidence.

**Before/after churn evidence.** A `_SwapCounter` wraps
`WatchlistsWorkbench._swap_region_widget` (call-through, so it only
observes). Born-red: file-swapped the pre-fix module in and reran the new
suite — 5 of 6 new tests failed (the sixth pins pre-existing, already-correct
no-op persistence behaviour and legitimately passes both before and after).
Restored the fix — all 6 pass. Both config states hold with zero swaps:
the shipped first-run default (RIGHT_RAIL collapsed) and a real, end-to-end
fresh config through a genuine mounted pilot (`test_cold_open_with_a_fresh_real_config_shows_the_collapsed_rail_with_no_swap`),
plus an explicit non-default user choice (LEFT_RAIL collapsed, RIGHT_RAIL
expanded) — proving the fix derives from `load_region_layout()` rather than
hard-coding either default.

**Trap found:** `Tests/conftest.py`'s autouse `isolate_test_environment`
fixture blanket-patches `watchlists_collections_screen.load_region_layout`
to `lambda: RegionLayout()` whenever the screen module is already imported,
so pre-task-2513 screen tests don't have to care about collapse state. A
pilot-mounted test that doesn't override this back (to a fixed value or to
the real function) silently measures the stub, not real config — burned an
hour chasing a "the fix doesn't work" ghost before finding the patch at
`Tests/conftest.py:967`. Every new test that cares what `load_region_layout`
actually returns now explicitly overrides it, matching the established
`test_persisted_layout_is_applied_on_mount` idiom.

**Tests run:** `Tests/Watchlists/test_watchlists_cold_open_layout.py` (new,
6/6 passed, born-red confirmed against the pre-fix module). Broader suite,
4 foreground batches: `test_watchlists_collections_screen.py` +
`test_region_layout.py` + `test_region_layout_store.py` +
`test_watchlists_workbench.py` + the new file (163 passed);
`test_watchlists_destination_shell.py` (79 passed, 1 failed —
`test_a_background_tree_reload_repaints_the_artifacts_scope_note`,
confirmed flaky/unrelated: passes in isolation, exercises a rename/tree-reload
timing path with no connection to region_layout); `test_watchlists_scoped_rebuilds.py`
+ `test_watchlists_rail_counts_and_scope.py` + `test_watchlists_content_pane.py`
(100 passed); `test_watchlists_inspector.py` + `test_watchlists_run_detail.py`
+ `test_watchlists_items_pane.py` + `test_watchlists_pagination.py`
(116 passed). Total 458 passed, 1 confirmed-flaky. `ruff check` clean on
both touched files; `ruff format` applied to the new test file only —
the pre-existing screen module has extensive prior format debt unrelated
to this change (no `[tool.ruff]` config pins a line length; the repo does
not run `ruff format` as a whole-file gate), and my inserted regions
already match `ruff format`'s output, so reformatting the other ~11,000
lines was out of scope and not attempted.

**Files modified:**
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — `__init__`
  seeds `region_layout`/`_last_persisted_collapsed` from one
  `load_region_layout()` call; `on_mount` reuses `self.region_layout`
  instead of loading a second time; updated docstrings/comments.
- `Tests/Watchlists/test_watchlists_cold_open_layout.py` (new) — 6 tests
  covering AC#1-#3.
