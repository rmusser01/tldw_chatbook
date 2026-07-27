---
id: TASK-997
title: >-
  Watchlist tree row draws its expand chevron on its own line above the name
status: To Do
assignee: []
created_date: '2026-07-27 22:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A watchlist row in the tree renders its expand chevron on a separate, indented line above the name instead of beside it. Captured live at 235x52 on `origin/dev` `dbbb7de84` after creating one watchlist from a clean profile:

```
│ Unassigned  0            │
│       ▸                  │
│ Morning AI Brief  0      │
```

Expected: `▸ Morning AI Brief  0` on one row.

It costs a rail row per watchlist and reads as a stray glyph, which matters more as the tree fills up — the rail is 26 columns and the tree is the screen's primary navigation.

Evidence: `Docs/superpowers/qa/watchlists-uat-2026-07-27/notes.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The chevron and the watchlist name render on the same row
- [ ] #2 One watchlist occupies one row in the collapsed state
- [ ] #3 A test asserts the rendered row text against the production stylesheet, proven to fail against current code
- [ ] #4 Expanding still shows the watchlist's sources indented beneath it
<!-- AC:END -->
