---
id: TASK-15461
title: Watchlists: one scoped rebuild per section or tree interaction
status: In Progress
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
- [ ] #1 A section switch performs one scoped rebuild of the changed section only (evidence)
- [ ] #2 A tree-node click causes at most one update per affected pane; briefing arrow-key selection no longer triggers 3 recomposes
- [ ] #3 z/Z/[/] layout keys rebuild only the toggled region; all behavior unchanged (tests)
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
