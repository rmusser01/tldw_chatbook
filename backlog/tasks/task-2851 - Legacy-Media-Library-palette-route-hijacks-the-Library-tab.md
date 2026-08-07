---
id: TASK-2851
title: Legacy Media Library palette route hijacks the Library tab
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - navigation
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-02, Assessment B anomaly 1, observed at dev `6ffa56516`).

The command palette exposes "Media & Content: Open Media Library", which opens the OLD Media
Library screen (left nav: Media Types / All Media / Analysis Review / Collections/Tags /
Multi-Item Review) rendered UNDER the active ⌃3 Library tab (toast: "Opened Media Library").
After landing there, selecting the palette's "Tab Navigation: Switch to Library" still displayed
the legacy Media Library content under the Library tab until app restart.

Several legacy screens were already retired via `_SCREEN_ALIASES` in
`UI/Navigation/screen_registry.py` (Notes/Skills/Prompts/Search alias to Library). This route
escaped that sweep: two different surfaces both answer to "Library", and the stale one wins a
sticky fight with tab activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No palette entry opens the legacy Media Library screen as a dead-end twin: the entry is removed, or it deep-links into the canonical Library Media canvas
- [ ] #2 Activating the Library tab (palette "Switch to Library" or tab click) always re-asserts the canonical LibraryScreen, even after any legacy/deep-link route
- [ ] #3 A regression test covers the palette route and the tab re-assertion
- [ ] #4 Live TUI verification: palette route + subsequent "Switch to Library" both land the canonical Library, no restart needed
<!-- AC:END -->
