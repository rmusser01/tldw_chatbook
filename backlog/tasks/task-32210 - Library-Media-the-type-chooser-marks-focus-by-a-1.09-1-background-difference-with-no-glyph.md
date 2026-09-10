---
id: TASK-32210
title: >-
  Library Media: the type chooser marks focus by a 1.09:1 background difference
  with no glyph
status: To Do
assignee: []
created_date: '2026-09-10 14:52'
labels:
  - library
  - media
  - accessibility
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the type chooser the focused row is background rgb(30,30,30) against rgb(39,39,39) elsewhere (darker than its neighbours, no glyph); only the active value has `✓`. The footer has committed the user to a keyboard interaction whose cursor is invisible in a plain-text capture. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The focused chooser row carries the `█` left-edge bar the lists use; `✓` keeps marking the active value
- [ ] #2 A painted-cell test pins the bar on the focused row and its absence on the others
<!-- AC:END -->
