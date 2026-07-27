---
id: TASK-995
title: >-
  The Sources toolbar renders no controls, so a new user cannot add a source
status: Done
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
- [x] #1 The search input, all three filters, `New Source` and `Filters` are visible and usable on the Sources section
- [x] #2 A new user can create a source from a clean profile without using OPML import
- [x] #3 Every other screen using `.destination-filter-strip` is checked for the same clipping, and any found are listed here or fixed
- [x] #4 A test asserts the controls are actually rendered — against the production stylesheet in the full shell, and proven to fail against current code
- [x] #5 The Sources table keeps the row space it gained in task-897
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
The UAT diagnosis was half the story. There were **two independent causes**,
and fixing the height alone would have left the toolbar just as unusable.

**1. Height.** `.destination-filter-strip` is `height: 1`, a bordered
`Input`/`Select` is three rows, so only the top border painted. Fixed by
passing `compact=True` to those widgets — Textual's own supported one-row
form, already used ~300 times in this codebase (including for `Input`/`Select`
in `mcp_server_mutations.py`). Done in Python rather than by un-pinning the
strip class: `.destination-filter-strip` is shared chrome, and growing it to
three rows would have cost the Sources table four of its sixteen rows at
160x42, re-opening TASK-897.

**2. Width, which nobody had noticed.** Nothing sized these controls, so
`Input`'s Textual default (`width: 100%`) and `_conversations.tcss`'s bare
global `Select { width: 100%; }` each claimed the *entire* strip and stacked.
Measured before the fix, in a pane 93 columns wide on a 160-column terminal:

    sources-search-input   Region(x=31,  width=91, height=3)
    sources-type-select    Region(x=122, width=91, height=3)
    sources-status-filter  Region(x=213, width=91, height=3)
    sources-active-filter  Region(x=304, width=91, height=3)
    sources-new-button     Region(x=395, width=16, height=1)

`New Source` was 300 columns off the right edge of the terminal. Explicit
widths now live in `features/_watchlists.tcss`; they have to be in the bundle,
not in a widget's `DEFAULT_CSS`, because Textual always ranks a CSS_PATH rule
above `DEFAULT_CSS` regardless of specificity — the same lesson recorded for
`#mcp-tools-filter-server-slot Select`. The override also zeroes the
`margin-bottom: 1` that the same global rule adds.

After: at 160x42 the row measures search 19 / type 12 / status 16 / active 12
/ New Source 16 / Filters 16 = exactly the 91-column interior, and paints
`Search sources...  All ▼ All statuses ▼ All ▼  New Source  Filters`. At
235x52 the search box takes 94.

**AC#3 — the sweep.** `.destination-filter-strip` has eleven users. Schedules
(`#schedules-filter-strip`), Workflows (`#workflows-mode-strip`), the empty-
state action row and the Runs / Rules / Notifications toolbars hold only
`Static`s and `Button`s and were never affected — a `Button` renders its label
fine at `height: 1`, which is why the Rules strip looked correct during the
UAT. The two that did carry an `Input` or `Select`, and so had exactly this
defect, are `#items-toolbar` and `#watchlists-header-bar`; both are fixed here
and covered by a test rather than left for the next UAT to rediscover.

**AC#5.** `#sources-toolbar` keeps `height: auto` and both its rows stay one
row tall, so the table's share is unchanged;
`test_watchlists_sources_toolbar_does_not_starve_its_table` stays green.

**Tests.** `test_watchlists_sources_toolbar_controls_are_actually_visible` was
supplied red and extended with horizontal-containment assertions (regions
alone cannot catch an off-screen control, and `render_strips` alone cannot
distinguish it from a clipped one). `test_watchlists_other_filter_strip_
controls_are_visible` is new. Both were re-proven red against the unfixed
code at both 160x42 and 235x52 before being taken green.

**Not fixed, noted.** The type and active filters both display `All` when
unset, so the row now shows two identically-labelled dropdowns. That is the
pre-existing option wording, not this defect, and is left alone.

Modified: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`,
`tldw_chatbook/UI/Watchlists_Modules/items_pane.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`tldw_chatbook/css/features/_watchlists.tcss`,
`tldw_chatbook/css/tldw_cli_modular.tcss` (generated),
`Tests/UI/test_destination_visual_parity_correction.py`.
<!-- SECTION:NOTES:END -->
