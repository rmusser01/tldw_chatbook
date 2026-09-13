---
id: TASK-15461
title: 'Watchlists: one scoped rebuild per section or tree interaction'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `watch_active_section` (`UI/Screens/watchlists_collections_screen.py:3882`) does a whole-screen `refresh(recompose=True)` (~450-650 widgets), after which `_load_active_section_data()` triggers the freshly-built pane's own recompose reactive one frame later — 2+ full rebuilds per section tab click. One tree-node click cascades into 4 recomposes (inspector `:3580` -> tree `:3633` -> sources push -> items worker). `watchlists_workbench.py:87` `region_layout` rebuilds all four region panes per z/Z/[/] keypress or chevron click (blast radius documented at `:103-123`).

Fix direction: region-scoped updates — rebuild only the section/region that changed; split the tree-click chain into targeted syncs; coalesce the select->clear->reload pipeline (the artifacts pane's 3-stage recompose pipeline, `artifacts_pane.py:655-788`, is the same disease). Coordinate with task-15460. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A section switch performs one scoped rebuild of the changed section only (evidence)
- [x] #2 A tree-node click causes at most one update per affected pane; briefing arrow-key selection no longer triggers 3 recomposes
- [x] #3 z/Z/[/] layout keys rebuild only the toggled region; all behavior unchanged (tests)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first. Instrumented probe (isolated HOME/XDG/`TLDW_CONFIG_PATH`,
   real seeded watchlist + briefings) counting `Widget.recompose()` calls and
   mounted widget instances per interaction, plus the synchronous latency.
   Record the BEFORE numbers before touching anything.
2. Pin behaviour: characterisation tests for section switching, tree-click
   fan-out, z/Z/[/] and briefing arrow-key selection. Green before changes.
3. `WatchlistsWorkbench.region_layout`: drop `recompose=True`. Replace it with
   a reconcile that swaps ONLY the regions whose rendered form actually
   changed (collapsed header <-> expanded body) and patches the
   sole-centre CSS marker in place on the ones that did not.
4. Add a workbench `apply_section_view()` that mounts/unmounts hidden centre
   regions, applies the tab-adjusted layout and rebuilds only the ITEMS
   region + the centre header -- the two surfaces `active_section` actually
   feeds.
5. `watch_active_section`: stop calling whole-screen
   `refresh(recompose=True)`. Queue the scoped swap on the EXISTING
   `_request_surface_refresh` drain (so it can never interleave with the
   header/rail swaps that queue there), and sync the backend header bar in
   place.
6. Artifacts pane: fold the screen's "clear the previous briefing's
   scripts/citations/audio" writes into the pane itself, applied with
   `set_reactive` inside `watch_selected_briefing`, so select+clear is ONE
   recompose instead of two.
7. Re-measure with the same probe; record AFTER numbers. Re-run the pinned
   suites unmodified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`WatchlistsWorkbench.region_layout` is no longer `recompose=True`, and
`watch_active_section` no longer calls whole-screen `refresh(recompose=True)`.
Both are replaced by swaps scoped to what actually changed.

**Layout keys (AC#3).** `watch_region_layout` swaps only the regions whose
rendered form moved -- collapsed one-line header <-> expanded body. A region
that stays expanded keeps its widget instance and has its
`watchlists-region-sole-centre` marker toggled in place, which is the only
other thing a layout change can move for it. `ContentPane.expanded` gained a
watcher that relabels the Expand/Restore button in place, because soloing
CONTENT does not change CONTENT's own form (it collapses ITEMS, the sibling),
so the pane survives and nothing else would repaint the label.

**Section switch (AC#1).** `WatchlistsWorkbench.apply_section_view` mounts or
unmounts the centre regions the new tab hides/shows, applies the tab-adjusted
layout in the same pass, and rebuilds the ITEMS region plus the centre header
-- the only surfaces `active_section` feeds. Both rails, the screen, the
navigation bar and the footer are untouched. The backend Select and its
local-only label are patched in place by `_sync_backend_header_bar`.

Two seams had to move for this to be correct rather than merely smaller, and
both are recorded in `backlog/docs/lessons-testing-evidence.md`:

* The surface-refresh drain now schedules with `call_next` instead of
  `run_worker`. A worker is invisible to `Pilot.pause`'s `_wait_for_screen`
  (and to the app's own idle handling), which `refresh(recompose=True)` -- a
  `call_next` callback internally -- was not. Ten tests began failing
  load-dependently until this moved back onto the pump. It is also strictly
  safer: the private worker group existed only so `exclusive=True` callers
  could not cancel the drain mid-swap, and a pump callback cannot be.
* `_reseed_active_section_pane` re-applies the section's rows after the mount.
  `refresh_region_content` deliberately calls the region factory *before*
  detaching the old pane (so a raising factory leaves the screen intact), which
  opens a window Textual's own `recompose` does not have; a loader landing in
  it wrote rows nobody was left to render (a permanently empty Alert-rules
  table over a populated `_loaded_rules`). The re-apply is free when nothing
  moved, and it is what keeps a switch to ONE pane build: the loader's own
  later push finds its values already in place.

**Briefing selection (AC#2).** The previous briefing's scripts/audio/citations
are now cleared inside `ArtifactsPane.watch_selected_briefing` with
`set_reactive`, folding select+clear into the one recompose the selection had
already queued instead of the screen adding a second from its message handler.

**Tree click (AC#2, first half).** Measured at ≤1 update per affected pane
*before* this task -- Textual coalesces a burst of `recompose=True` writes on
one widget into a single rebuild via `_recompose_required`, so the audit's
"4 recomposes" is 4 different widgets, one each. Nothing needed fixing; the
new test is a regression guard, and honestly labelled as one (it is the single
test in the new file that does not go red against the pre-task code).

**Measured** (isolated HOME/XDG/`TLDW_CONFIG_PATH`, seeded watchlist with a
source, 3 items and 3 briefings; `Widget.recompose`/`Widget.mount` counted;
ms = dispatch to message-pump drained, best of two runs):

| interaction | recompose | mounted widgets | ms |
|---|---|---|---|
| section: Sources | 2 -> 1 | 128 -> 71 | 154 -> 118 |
| section: Runs | 1 -> 0 | 84 -> 27 | 66 -> 30 |
| section: Rules | 1 -> 0 | 75 -> 18 | 120 -> 24 |
| section: Notifications | 1 -> 0 | 80 -> 23 | 56 -> 27 |
| section: Artifacts | 3 -> 2 | 176 -> 118 | 88 -> 48 |
| section: Read (warm) | 1 -> 0 | 111 -> 54 | 76 -> 48 |
| section: Overview | 2 -> 1 | 90 -> 33 | 49 -> 28 |
| `[` collapse rail | 1 -> 0 | 59 -> 1 | 114 -> 108 |
| `[` expand rail | 1 -> 0 | 95 -> 19 | 128 -> 107 |
| `]` collapse Inspector | 3 -> 1 | 123 -> 24 | 232 -> 107 |
| `]` expand Inspector | 1 -> 0 | 77 -> 1 | 121 -> 86 |
| `z` collapse ITEMS | 1 -> 0 | 37 -> 1 | 113 -> 86 |
| `Z` solo CONTENT | 1 -> 0 | 37 -> 0 | 115 -> 88 |
| tree click | 1 -> 1 | 26 -> 26 | 31 -> 30 |

Two honest caveats. `section: Read` (cold) is the one interaction whose
wall-clock did not improve (75 -> 110 ms on the best of two) despite halving
the DOM work: it is the tab that has to mount the CONTENT region back, and the
scoped path does that as its own remove/mount pair rather than inside one
batched recompose. And the ms column is two samples on a busy machine; the
mount/recompose counts are the deterministic evidence, the timings are
supporting.

**Behaviour pinned, not moved.** Three existing tests used "press `[`" as a
convenient way to force a pane rebuild -- which is precisely what AC#3
abolishes. Their contracts (a rebuilt pane is re-seeded with its filter/
selection/detail) are unchanged; only the trigger moved, to a real region
collapse/expand or to `refresh_region_content` directly, with the reason
recorded at each site. `test_bracket_toggle_preserves_inspector_selection` was
strengthened rather than retargeted: it now asserts the Inspector is never
rebuilt at all. One test in the artifacts suite drove
`handle_briefing_selected` directly to prove synchronous clearing; it now
drives the selection, which is the synchronous route the clearing moved to.

**Review round (post-approval).** Two Importants and four minors, all closed:

* **Focus.** A tab click focuses the tab `Button`, the swap rebuilds the header
  it lives in, and Textual's `Screen._reset_focus` re-homed focus onto a LEFT
  RAIL tree node -- so the next `z` collapsed AND persisted the rail. Measured
  A/B (pre-task it was focus-`None` and `z` was refused).
  `_restore_focus_after_swap` puts focus back on the rebuilt
  `#wl-tab-<section>`, which restores the refusal by the honest route
  (`_focus_in_centre_header`). Gated on `_swap_will_destroy_focus` so a
  section change driven from elsewhere never steals focus. Two tests.
* **Stale comments.** Eleven sites still asserted `region_layout` "is
  `recompose=True`" -- several load-bearing for the factory/seeding rationale.
  All corrected to the real trigger (a region is rebuilt when its own form
  changes, or on a section switch for ITEMS), keeping the historical note
  where it explains why the state is mirrored at all.
* **Minors.** The pane-build test now seeds a real alert rule (the assertion
  could not discriminate on an empty fixture -- and with a row it immediately
  exposed a residual, below); `_reseed_active_section_pane` has an in-suite
  pin that reconstructs the mid-swap landing window deterministically and
  reds when the reseed is neutered; the task-627 mouse-capture note is on
  `_swap_active_section`.

**Residuals** (for the controller to file):

1. **Cold Read tab wall-clock.** 75 -> 110 ms best-of-two despite halving the
   DOM work; it is the only tab that re-mounts the CONTENT region, and the
   scoped path does that as a discrete remove/mount pair rather than inside
   one batched recompose. Textual's `batch()` is the obvious next move.
2. **Two artifacts tests flickered once** (`test_the_kill_switch_on_leaves_
   the_cadence_picker_unchanged`, `test_keep_button_disabled_without_a_
   complete_selection_or_chachanotes`) in an intermediate run, before the
   `call_next` fix, and have passed in every run since including three
   full-directory runs. Probably the same wait-window issue; flagged rather
   than asserted.
3. **Artifacts pane recomposes wholesale on a briefing selection** (1, down
   from 2). Because it recomposes, the briefings `DataTable` is destroyed and
   loses focus, so a SECOND arrow key does nothing until the user re-focuses
   the table. Measured on the pre-task code too -- pre-existing, not caused
   here, but a real defect worth its own task.
4. **`_build_detail_pane`'s pre-mount seeding costs one pane recompose** per
   region build (`[] != [row]` on a freshly constructed pane). Pre-existing
   and unchanged by this task; invisible on an empty fixture, which is why it
   surfaced only once the review asked for a seeded row. Removing it means
   seeding with `set_reactive`, which is not safe blind: `RunsPane`'s seeding
   ORDER is load-bearing (`selected_run` clears the detail, so the detail must
   be set after it).

**Files.** `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`,
`.../artifacts_pane.py`, `.../content_pane.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`;
`.../sources_pane.py`, `.../rules_pane.py`;
`Tests/Watchlists/test_watchlists_scoped_rebuilds.py` (new, 17 tests, 12 of
them red against the mechanism they pin), `Tests/Watchlists/
test_watchlists_artifacts_pane.py`, `Tests/UI/test_watchlists_content_pane.py`,
`Tests/UI/test_watchlists_destination_shell.py`,
`Tests/UI/test_watchlists_run_detail.py`,
`backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
