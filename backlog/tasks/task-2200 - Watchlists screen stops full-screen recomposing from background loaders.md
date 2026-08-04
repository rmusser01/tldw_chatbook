---
id: TASK-2200
title: Watchlists screen stops full-screen recomposing from background loaders
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ui
  - tech-debt
dependencies: []
priority: medium
---

## Description (the why)

The Watchlists screen's background loaders request full-screen recomposes:
`_apply_local_wc_snapshot` (watchlists_collections_screen.py:983) and
`_load_tree_data` (:912) call `refresh(recompose=True)` at screen level, tearing
down and rebuilding the entire centre — including panes that are mid-recompose
of their own.

This is the architectural root cause behind TASK-1960's crash class, confirmed
empirically during that investigation: `App._prune` stamps `_pruning` over a
`walk_children` snapshot, `Widget.mount` silently mounts nothing on a pruned
widget, and `MessagePump._pre_process`'s `finally` marks it mounted anyway — so
any widget whose `_on_mount` queries its own children can crash when a screen
recompose lands mid-mount. TASK-1960 shipped `PruneSafeSelect` as the
mechanism-level guard for `Select`, but the guard covers one widget class, not
the hazard: any other composite widget mounted by these rebuilds is exposed the
same way.

A second latent defect is masked by the same behaviour: `SourcesPane`'s
form-close recompose silently mounts *nothing* when the screen prunes it
mid-flight — invisible today only because the screen's own recompose immediately
rebuilds the pane. If a future change stops the screen rebuilding the pane on
that path, form-close visibly breaks.

This is also TASK-1541's standing recommendation (recorded there as "open rec,
documented, not filed"): replace recompose-the-world refresh paths with the
targeted in-place update discipline the screen already uses elsewhere
(`update_item_status_cell`, `refresh_header_content`).

## Acceptance Criteria (the what)

- [x] Completing a background load (`_load_tree_data`) or applying a local
      watchlists snapshot (`_apply_local_wc_snapshot`) no longer rebuilds the
      whole screen: the affected panes/regions are updated in place, and
      unrelated in-flight pane recomposes (e.g. the Sources create-form
      open/close) are not torn down by it.
- [x] The rendered result after a background refresh is equivalent to what the
      full recompose produced today, covering the overview/first-run/loading
      states (the TASK-1347-strengthened tests stay green).
- [x] The TASK-1960 e2e reproduction
      (`test_a_source_can_be_created_end_to_end_through_the_form`) stays green
      10/10 in isolation and in the poisoned order after
      `Tests/UI/test_watchlists_content_pane.py` — now with the destroyer
      removed rather than merely guarded against.
- [x] The masked SourcesPane defect is addressed or made impossible: closing the
      create form yields a correctly-populated pane without depending on a
      screen-level rebuild to paper over an empty recompose.
- [x] `PruneSafeSelect` remains in place as defense-in-depth (this task removes
      the known destroyer; it does not un-fix TASK-1960).

## Implementation Plan

1. Map every recompose site on `watchlists_collections_screen.py` and classify
   each as user-gesture-driven (keep) or background-loader-driven (convert).
2. Give the screen one owner for in-place workbench surface rebuilds: a
   record-intent / drain-serially helper (`_request_surface_refresh` +
   `_drain_surface_refresh`) over `WatchlistsWorkbench.refresh_region_content`
   / `refresh_header_content`, so two loaders landing together coalesce
   instead of cancelling each other mid remove-then-mount.
3. Convert `_apply_local_wc_snapshot`: request FEEDS + centre-header rebuilds
   and patch the Inspector's snapshot-derived widgets (`State:` line, the
   Console attach button's `disabled`/tooltip) in place.
4. Convert `_load_tree_data`: request LEFT_RAIL + FEEDS + centre-header
   rebuilds and push `breadcrumb_labels` / `watchlist_count` into the live
   Inspector and Overview panes.
5. Convert `overview_data` from a screen-level `recompose=True` reactive to a
   plain reactive plus a watcher that pushes into the live `OverviewPane`,
   the Inspector's `profile_state`, and the two summary `Static`s.
6. Route `watch_tree_scope`'s existing FEEDS/header refreshes through the same
   drain, so nothing else can swap those regions concurrently.
7. Add regression tests for AC#1 (an in-flight create form survives a
   background load) and AC#4 (form-close yields a correct pane with the same
   pane instance, i.e. no screen rebuild papering over it); update the tests
   whose assertions pinned the old full-recompose behaviour.
8. Run the verification gates: overview/first-run/loading-state, region
   gating, content pane, source create form (full), sources pane, inspector,
   the AC#3 10x/5x e2e repeats, `--collect-only`, and mutation-revert every
   behavioural change.

## Implementation Notes

The three background destroyers are gone. The screen keeps exactly one
`refresh(recompose=True)`, in `watch_active_section` — a user gesture that
genuinely changes which regions mount (`_hidden_centre_regions`) and whether
the workbench gets a `header=` factory at all.

### Every recompose site, classified

| Site | Driver | Verdict |
|---|---|---|
| `overview_data` reactive (`:401`) | background (`_refresh_overview_data`, its only writer) | **converted** — plain reactive + `watch_overview_data` |
| `_load_tree_data` (`:913`) | background worker | **converted** — `_apply_tree_data_to_live_surfaces` |
| `_apply_local_wc_snapshot` (`:984`) | background worker + the snapshot timeout | **converted** — `_apply_snapshot_to_live_surfaces` |
| `watch_active_section` (`:2963`) | user gesture (tab switch) | **kept** — changes which regions exist |
| `region_layout` (`:410`) | user gesture (`z`/`Z`/`[`/`]`) | **kept, out of scope** — and it is *not* `recompose=True` on the screen; the reactive with that flag is `WatchlistsWorkbench.region_layout`, pushed by `_apply_layout` |

### One owner for DOM swaps

`refresh_region_content`/`refresh_header_content` are remove-then-mount pairs
with an `await` between the halves (Textual's `NodeList._ensure_unique_id`
refuses to mount the replacement while the old widget still holds the id).
Three call sites now want those swaps, so `exclusive=True` per surface — the
shape `watch_tree_scope` used — becomes unsound: whichever request lands
second cancels the first *between* its `remove()` and its `mount()`, leaving a
bordered empty box. `_request_surface_refresh`/`_drain_surface_refresh` record
intent and drain serially instead, never cancelling (TASK-1541's lesson,
applied to DOM swaps rather than durable writes). `watch_tree_scope`'s two
refreshes were routed through the same queue for the same reason.

### What each converted loader now updates

* **Snapshot** — FEEDS + centre header (the loading/error/empty/summary marker
  lives in exactly those two places), plus two in-place patches on the
  Inspector: the `State:` line (`Static.update`) and the Console attach
  button's `disabled`/tooltip.
* **Tree data** — the rail, FEEDS + header (the scope heading resolves a
  watchlist name out of `_tree_watchlists`, so a rename would otherwise sit
  stale), the Inspector's `breadcrumb_labels`, and the Overview's
  `watchlist_count` (TASK-998's first-run copy).
* **`overview_data`** — `OverviewPane.data` and `InspectorPane.profile_state`
  (both pane-scoped `recompose=True` reactives; the pane swaps between three
  whole layouts, so there are no cells to patch), plus the Inspector's two
  count `Static`s.

`overview_data`'s fate was a real question, not a formality: TASK-1960 proved
it does **not** fire on the post-submit path (an equal dict), but it fires
whenever counts genuinely change, and `_update_item_status`'s `refresh=False`
path and `_save_noise_selectors` both exist *only* to dodge the screen
recompose it used to cause. So it was the third destroyer, and it is converted.

### Two things the full recompose was carrying that nothing else was

Found by running the suites, not by reading:

1. **The backend switch.** `watch_runtime_backend` clears
   `selected_source`/`selected_run`/`selected_notification` on the screen, and
   the snapshot refresh's recompose is what pushed those clears into the
   mounted pane. `_reseed_live_detail_pane` now does it directly (also called
   from `_delete_source`/`_delete_run`, which clear the same state).
2. **The Console-follow row.** It is polled from an app-level adapter *at
   render time only*, so an adapter that failed once recovered on whichever
   recompose came next. `_resolve_console_follow_drift` re-polls and compares
   against the DOM; the right rail is rebuilt only when it genuinely differs.
   Deliberately conditional: the rail hosts the noise-selector editor, and
   rebuilding it on every background load would destroy a half-typed selector
   set — the same harm this task removes from the Sources create form. The
   RIGHT_RAIL factory had to stop closing over `compose_content`'s captured
   values for that rebuild to mean anything.

### AC#4

Addressed by removal. With the screen recompose gone from both loaders,
nothing prunes `SourcesPane` mid-form-close, so the pane's own recompose
mounts its children normally — and the pane the user is looking at is the one
the assertions now target (`test_closing_the_create_form_repopulates_the_same_pane`
asserts pane identity *first*, then that the same instance has its table and
no form, so a screen-level rebuild can no longer satisfy it).

### Verification

* AC#3: **10/10** e2e in isolation, **5/5** of
  (`test_watchlists_content_pane.py` + the e2e) in one invocation, that order.
* `Tests/Watchlists/` 384 → **480 passed** together with the watchlists UI
  files; `test_watchlists_destination_shell.py` + `overview_loading_state` +
  `inspector` **110 passed**; `test_destination_shells.py` +
  `test_destination_visual_parity_correction.py` + `source_create_form`
  **243 passed, 1 skipped**; `test_console_live_work_handoffs.py` +
  `test_no_side_effecting_predicates.py` **49 passed**; shell/navigation/
  maturity sweep **124 passed**; `--collect-only Tests/UI Tests/Watchlists
  Tests/Widgets` **8750 collected**, no errors.
* 15 mutations, each reverted individually → RED → restored byte-exact
  (md5-verified, `git status --short` unchanged between).

### Files

* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — the whole
  change.
* `tldw_chatbook/UI/Watchlists_Modules/items_pane.py` — one stale comment.
* Tests: `Tests/UI/test_watchlists_destination_shell.py` (+6 tests, and the
  rule-edit test's `is not rules_pane` assertion inverted to `is` — it used to
  *require* the destroyer), `Tests/UI/test_watchlists_source_create_form.py`
  (+2: AC#1 draft survival, AC#4 pane identity),
  `Tests/Watchlists/test_watchlists_collections_screen.py` (the rename test now
  also asserts the mounted Inspector's breadcrumb).
