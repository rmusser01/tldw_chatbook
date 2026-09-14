---
id: TASK-32557
title: >-
  Library Notes: "Add from files…" is painted "Add from" beside the grip at
  60x24
status: In Progress
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 16:00'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, compact terminal. D14. Task-32360 fixed mid-word clipping on the rail-only stage below 64 columns; task-32127's whole-word pin is at 235 wide.

**What happened.** At 60x24 the Notes list toolbar's first row paints `New  Sort: Newest  Select  Add from   s` — "Add from files…" cut to "Add from" against the grip (B 52 line 10). The footer compacts correctly and the status drops its prefix as documented. Captures: B 52.

**Cause.** INFERRED. Docs contradicted: "Nothing is ever painted as half a word."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x24 every Notes toolbar label paints whole or is elided with an ellipsis
- [ ] #2 A test at 60 columns pins the toolbar labels
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: fresh 60x24 is clean; resizing a merged 235-wide list down to 60 paints 'New  Sort: Newest  Select  Add from   s' (the critique's exact line).
2. Trace: _effective_pane_width prefers the screen-contract pane_width, which lags one resize behind the resolved layout, so on_resize re-decides from the stale wide value and keeps the merged shape.
3. Prefer the canvas's own measured width; pane_width stays the pre-measurement fallback.
4. Pin the resize round trip through the production canvas.
<!-- SECTION:PLAN:END -->
