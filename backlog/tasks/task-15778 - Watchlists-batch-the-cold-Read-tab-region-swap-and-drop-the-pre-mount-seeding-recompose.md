---
id: TASK-15778
title: 'Watchlists: batch the cold Read-tab region swap and drop the pre-mount seeding recompose'
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

- [x] The cold `section: Read` switch's CONTENT-region remount is batched
      (e.g. via Textual's `batch()`) into the same pass as its other DOM
      work, and the wall-clock regression measured in task-15461 is closed
      (before/after recorded — with an honest refutation: the audited
      mechanism behind the regression never existed at this HEAD; see
      Implementation Notes)
- [x] `_build_detail_pane`'s pre-mount seeding uses `set_reactive` (or an
      equivalent that avoids the extra recompose) without breaking any
      pane's load-bearing seeding order — `RunsPane`'s `selected_run`-before-
      detail ordering explicitly verified by test
- [x] Every other pane that uses `_build_detail_pane` is checked for its own
      seeding-order dependencies before the change lands, not just
      `RunsPane`
- [x] `_build_content_pane`'s `item` seeding (the CONTENT region built by the
      same cold Read-tab swap; `item` is `recompose=True` and pays the same
      extra recompose whenever an item is selected) gets the identical
      treatment, with its own seeding-order/watcher audit
- [x] `Tests/Watchlists/test_watchlists_scoped_rebuilds.py` and the sources/
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

## Implementation Notes

**Premise re-verification at HEAD `41a2f8a00`** (includes 15775's
seed-before-compose and 15776's row collapse — neither touched these two
residuals):

| premise | state at HEAD | evidence |
|---|---|---|
| swap unbatched | still true — no `batch_update`/`batch()` anywhere in the swap path | grep + probe |
| pre-mount seeding recompose | still true | instrumented section sweep: SourcesPane 1, RulesPane 1, OverviewPane 1, ArtifactsPane 2 recomposes per data-carrying switch; ContentPane 1 on a cold Read switch with a selected item |
| "the regression is BECAUSE the swap is unbatched" | **refuted** | 0 in-swap layout passes and 0 compositor refreshes measured on a cold Read switch, batch NEUTERED (A/B on the same HEAD) — the whole swap runs inside `_drain_surface_refresh`'s single `call_next` callback (15461's own `run_worker`→`call_next` move), so the pump never idles mid-swap and the paused update timer never fires |

**Fix 1 — batch (AC#1).** `WatchlistsWorkbench.apply_section_view` now runs
its DOM work (region sync, section-pane rebuild, header rebuild) inside
`App.batch_update()`. Given the refutation above this is a contract, not a
measured win: it makes the one-pass property structural (survives a future
factory that awaits, or the drain moving off a single callback) instead of
an accident of the drain's scheduling. Wall clock recorded, honest: cold
Read swap window 32–37 ms pre vs 37–39 ms post (noise); dispatch→settled
113–270 ms both arms, indistinguishable on a busy machine. The 15461-era
75→110 ms number does not reproduce as a batching problem at this HEAD.

**Fix 2 — seeding (AC#2/#3/#4).** `_build_detail_pane` and
`_build_content_pane` seed `recompose=True` reactives with `set_reactive`,
per-reactive after a per-pane watcher audit — NOT blanket:

* Converted (all pre-mount watchers `is_mounted`-gated no-ops, verified by
  reading each): OverviewPane `data`/`watchlist_count`; SourcesPane
  `sources`/`show_create_form`/`create_draft_source_type`; RunsPane `runs`;
  RulesPane `rules`/`show_rule_form` (+ `edit_rule` gained a pre-mount
  `set_reactive` route, same body, `is_mounted`-switched); NotificationsPane
  `notifications`/`selected_notification`; ArtifactsPane — all 20 seeded
  reactives; ContentPane `item` (no watcher at all).
* Deliberately left plain: every non-recompose reactive. RunsPane's
  `selected_run` is the load-bearing case — its watcher clears the stale
  detail (hence the detail-after-selection order, pinned by test) AND
  starts the status poll for a still-running run (pinned by test; a blind
  `set_reactive` would freeze a mid-flight run's status on every region
  rebuild). ArticleListPane audited: zero `recompose=True` reactives,
  nothing to convert (documented in the branch).

**Measured before/after** (isolated per-test config, seeded fixture,
`_RebuildCounter` on `Widget.recompose`/`mount`; ms noisy on a busy
machine, counts deterministic):

| interaction | pane recomposes | mounts | ms (dispatch→settled) |
|---|---|---|---|
| cold Read, item selected | ContentPane 1 → 0 | 81 → 69 | ~126–220 → ~132–148 (noise) |
| switch → sources | 1 → 0 | — | — |
| switch → rules (loader pre-landed) | 1 → 0 | — | — |
| switch → overview | 1 → 0 | — | — |
| switch → artifacts | 2 → 1 (the 1 = briefings loader landing post-mount, honest data arrival) | — | — |
| switch → runs / items | 0 → 0 (runs list empty; ArticleListPane has no recompose reactives) | — | — |

**Born-red.** All new pins in
`Tests/Watchlists/test_watchlists_cold_read_swap.py` were run against the
pre-fix production files (file-swap from git, tests kept): 13 red — the
batch pin (both config states: first-run default AND a user-customised
layout), every data-carrying `_recompose_required` seeding pin, the
edit-form prefill pin, the ContentPane pin, the warm-revisit end-to-end
pin, and the tightened scoped_rebuilds warm assertion (`<= 1` → `== 0`).
The 3 that stayed green are the designed behaviour-preservation pins:
`[items]` (never had the defect), the RunsPane ordering pin and the
RunsPane poll pin — each of those was then redded by its targeted mutation
(detail-before-selection reorder; blind `set_reactive` on `selected_run`),
Edit-based restores, tree verified clean after.

**Tests.** `Tests/Watchlists/` full directory: 688 passed. Watchlists UI
suites (`test_watchlists_content_pane.py`, `test_watchlists_rail_counts_
and_scope.py`, `test_watchlists_destination_shell.py`): 163 passed. ruff
check clean on all touched files; the new test file ruff-formatted (the
four touched production files were already unformatted at HEAD, so no
whole-file reformat churn).

**Files.** `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`tldw_chatbook/UI/Watchlists_Modules/rules_pane.py`,
`tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (comment only),
`Tests/Watchlists/test_watchlists_cold_read_swap.py` (new),
`Tests/Watchlists/test_watchlists_scoped_rebuilds.py`.
