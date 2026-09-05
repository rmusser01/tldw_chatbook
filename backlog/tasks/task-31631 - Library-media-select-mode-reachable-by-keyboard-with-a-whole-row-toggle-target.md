---
id: TASK-31631
title: >-
  Library media select mode - reachable by keyboard with a whole-row toggle
  target
status: To Do
assignee: []
created_date: '2026-09-05 06:18'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P1: after pressing s from the rail, F6, Down and Space all no-op with no focus indicator painted anywhere; only a mouse click on the one-cell ☐ glyph seeds focus, after which the keys work. Row-title clicks do nothing in select mode, and Done takes the exact slot sort: occupied. Cause: focus sits on the pane grip after the recompose that enters select mode (task-31567 is the general fix; this task is the select-mode contract).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entering select mode by key or click puts focus on the selected or first row with a visible focus ring, so Down and Space work immediately
- [ ] #2 Clicking anywhere on a row toggles its selection in select mode
- [ ] #3 Done does not occupy the position sort: held in browse mode
- [ ] #4 Painted tests at 235x52 and 100x30 cover the keyboard path from the rail
<!-- AC:END -->
