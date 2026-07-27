---
id: TASK-995
title: >-
  The Sources toolbar renders no controls, so a new user cannot add a source
status: To Do
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: critical
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On the Sources section, the toolbar's first filter strip renders as a single bare bar with **no visible controls**. It holds the search input, the type / status / active filters, the **`New Source`** button and `Filters` (`sources_pane.py:132-158`) — only its top border row is drawn.

Captured live at 235x52 on `origin/dev` `dbbb7de84`, clean profile:

```
  Sources

  ▊▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
      Preview        Check now      Import OPML     Export OPML
  Name  Type  Status  Last scraped  Active
```

**This blocks the primary new-user path.** A user who lands on Watchlists, creates a watchlist, and clicks `Create source` arrives here with no visible way to add one. Watchlists cannot be populated through the UI at all.

**Cause, isolated during the UAT.** `.destination-filter-strip` is `height: 1` (`css/layout/_panes.tcss:31-36`), but a bordered `Input`/`Select` is three rows, so only the top border survives. The Rules section's strip, which contains only `Button`s, renders correctly (`Refresh  New Rule` both visible) — so this is specific to strips carrying Inputs or Selects, not to the class itself.

Third occurrence of the one-row-container / three-row-children pattern on this project, after `WatchlistsTabStrip` and `LabModeStrip` (task-875). Note `.destination-filter-strip` is shared chrome — check which other screens put Inputs or Selects in one before changing the class itself, and prefer a scoped rule if the blast radius is wide.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The search input, all three filters, `New Source` and `Filters` are visible and usable on the Sources section
- [ ] #2 A new user can create a source from a clean profile without using OPML import
- [ ] #3 Every other screen using `.destination-filter-strip` is checked for the same clipping, and any found are listed here or fixed
- [ ] #4 A test asserts the controls are actually rendered — against the production stylesheet in the full shell, and proven to fail against current code
- [ ] #5 The Sources table keeps the row space it gained in task-897
<!-- AC:END -->
