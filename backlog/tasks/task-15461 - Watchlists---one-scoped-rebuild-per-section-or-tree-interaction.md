---
id: TASK-15461
title: Watchlists: one scoped rebuild per section or tree interaction
status: To Do
assignee: []
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
