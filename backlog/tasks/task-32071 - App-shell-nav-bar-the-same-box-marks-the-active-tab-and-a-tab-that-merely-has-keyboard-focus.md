---
id: TASK-32071
title: >-
  App shell nav bar: the same box marks the active tab and a tab that merely has
  keyboard focus
status: To Do
assignee: []
created_date: '2026-09-08 18:26'
labels:
  - app-shell
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tabbing out of a Library canvas into the nav bar boxes '⌃1 Home' exactly as when Home is the active screen, so a keyboard user cannot tell 'focused' from 'active' and may press Enter expecting a Library action. Outside Library but surfaced by its Tab order. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused nav tab is visually distinct from the active tab by shape, not colour alone
<!-- AC:END -->
