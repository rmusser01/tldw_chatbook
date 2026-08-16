---
id: TASK-15778
title: 'Watchlists: batch the cold Read-tab region swap and drop the pre-mount seeding recompose'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - perf
  - watchlists
priority: low
---

## Description

Two related residuals recorded "for the controller to file" in task-15461's
Implementation Notes (input-latency burn-down's Watchlists scoped-rebuild
work). Both are about construction-order cost inside the same region-build
plumbing task-15461 converted from whole-screen recomposes to scoped
rebuilds.

1. **Cold Read-tab wall-clock did not improve despite halving DOM work.**
   Measured 75 -> 110 ms (best-of-two) for the one section switch
   (`section: Read`, cold) that has to re-mount the CONTENT region. Every
   other measured section switch improved with task-15461's scoped rebuild;
   this one regressed on wall clock because the scoped path does the CONTENT
   remount as its own discrete remove/mount pair rather than inside one
   batched recompose. Task-15461's own notes point at Textual's `batch()` as
   "the obvious next move."
2. **`_build_detail_pane`'s pre-mount seeding costs one pane recompose per
   region build.** `[] != [row]` on a freshly constructed pane triggers an
   extra recompose; pre-existing and unchanged by task-15461, invisible on
   an empty fixture (which is why it surfaced only once task-15461's review
   asked for a seeded row). Fixing it means seeding with `set_reactive`
   instead of a plain assignment, which is not safe blind: `RunsPane`'s
   seeding ORDER is load-bearing (`selected_run` clears the detail, so the
   detail must be set after it) — any fix must preserve that ordering
   per-pane, not apply one blanket change.

## Acceptance Criteria

- [ ] The cold `section: Read` switch's CONTENT-region remount is batched
      (e.g. via Textual's `batch()`) into the same pass as its other DOM
      work, and the wall-clock regression measured in task-15461 is closed
      (before/after recorded)
- [ ] `_build_detail_pane`'s pre-mount seeding uses `set_reactive` (or an
      equivalent that avoids the extra recompose) without breaking any
      pane's load-bearing seeding order — `RunsPane`'s `selected_run`-before-
      detail ordering explicitly verified by test
- [ ] Every other pane that uses `_build_detail_pane` is checked for its own
      seeding-order dependencies before the change lands, not just
      `RunsPane`
- [ ] `_build_content_pane`'s `item` seeding (the CONTENT region built by the
      same cold Read-tab swap; `item` is `recompose=True` and pays the same
      extra recompose whenever an item is selected) gets the identical
      treatment, with its own seeding-order/watcher audit
- [ ] `Tests/Watchlists/test_watchlists_scoped_rebuilds.py` and the sources/
      rules pane suites stay green

## Implementation Plan

1. Re-verify both premises at HEAD (`41a2f8a00`, which includes task-15775's
   seed-before-compose and task-15776's row collapse): confirm neither
   residual was already fixed — (a) no `batch_update`/`batch()` anywhere in
   the workbench's swap path, so the cold Read switch still pays one
   layout/paint pass per remove/mount cycle; (b) `_build_detail_pane` still
   seeds recompose=True reactives by plain assignment, so a seeded row still
   queues one extra pane recompose per region build. Evidence via an
   instrumented probe (isolated HOME/XDG config, seeded fixture): layout-pass
   count + wall clock for the cold Read switch, and per-pane recompose counts
   per section switch.
2. AC#1: wrap `WatchlistsWorkbench.apply_section_view`'s mounted-path DOM
   work in `self.app.batch_update()` so the CONTENT mount, the ITEMS region
   rebuild and the header rebuild are reconciled in ONE layout/paint pass
   instead of one per remove/mount cycle. Focus restoration
   (`_restore_focus_after_swap`) and the persisted-layout contract are
   untouched — the batch only defers repaints, it does not reorder the DOM
   work.
3. AC#2/#3: audit every pane branch in `_build_detail_pane` for
   recompose=True reactives and watcher side effects, then convert exactly
   the recompose=True seeding assignments to `set_reactive` (their pre-mount
   watchers are all `is_mounted`-gated no-ops — verified per pane). Keep the
   non-recompose assignments as plain assignments so their load-bearing
   watcher side effects survive: `RunsPane.selected_run` must keep firing
   `watch_selected_run` (clears the detail — hence detail-after-selection
   order — and starts the run poll for a mid-flight run). `RulesPane.
   edit_rule` gets a pre-mount `set_reactive` path for `show_rule_form`.
4. Born-red tests (new file `Tests/Watchlists/test_watchlists_cold_read_swap.py`,
   reusing the 15775 `_SwapCounter`/15461 `_RebuildCounter` patterns):
   batch-active-during-swap pin (deterministic, red pre-fix), zero-pane-
   recompose-per-switch pins for the seeded panes (red pre-fix at 1),
   `RunsPane` ordering + poll-side-effect pins (red under a wrong-order or
   set_reactive-blind mutation), and the cold Read switch verified under
   both the first-run default layout and a user-customized layout.
5. Before/after swap-count + layout-pass + timing measurements on a cold
   Read-tab open, recorded in Implementation Notes.
6. Run the Watchlists suites (scoped_rebuilds, cold_open_layout, workbench,
   sources/rules/runs/artifacts/items pane suites, destination shell), ruff
   check + format on touched files, commit.
