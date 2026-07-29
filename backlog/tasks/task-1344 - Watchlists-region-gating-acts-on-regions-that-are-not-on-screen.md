---
id: TASK-1344
title: Watchlists region gating acts on regions that are not on screen
status: To Do
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The spec says "Only Read uses the three-pane split. Sources, Runs, Rules, and Artifacts take the
full centre width." Three related gaps remain after Phase D.

**FEEDS is always mounted** regardless of the active tab (unconditional in `_build_list_pane`),
the same violation Phase D fixed for CONTENT. Pre-existing from Phase C.

**The CONTENT gate collapses rather than unmounts.** `#wl-header-content` still measures
`height=1` on Sources at 160x42, so the region is not literally absent — it contributes a header
row. It no longer taxes the layout (that regression was fixed), but it is not "full centre width"
in the spec's sense.

**`Z` (solo) on CONTENT off the Read tab is ungated** (`watchlists_collections_screen.py:1578`).
Phase D gated the chevron and `z` so neither can persist a CONTENT collapse from a tab where the
region is invisible, and `collapsed_for_persistence()` returns the pre-solo baseline so solo cannot
corrupt persisted state. But solo still collapses FEEDS and ITEMS around a region the user cannot
see, leaving no expanded centre region — recoverable only by clicking a header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 FEEDS occupies the centre only on the Read tab, matching the CONTENT gating Phase D added
- [ ] #2 Solo on a region that is not visible on the active tab is refused, with a notify, exactly as the chevron and z toggles now are
- [ ] #3 A test asserts that no sequence of tab switches and region toggles leaves zero expanded centre regions
- [ ] #4 A decision is recorded on whether gated regions should unmount or keep a one-row header, and the implementation matches it
<!-- AC:END -->
