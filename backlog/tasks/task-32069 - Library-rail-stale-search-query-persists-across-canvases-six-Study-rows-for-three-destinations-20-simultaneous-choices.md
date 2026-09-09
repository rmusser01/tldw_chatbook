---
id: TASK-32069
title: >-
  Library rail: stale search query persists across canvases; six Study rows for
  three destinations; 20 simultaneous choices
status: To Do
assignee: []
created_date: '2026-09-08 18:25'
labels:
  - library
  - rail
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail search box keeps the last query when switching canvases with no clear affordance; the Study section spends six rows ('see what carries over' under each) on three destinations; the full rail presents 15 destinations plus 5 utility controls at once. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail search box offers a clear affordance or resets when the canvas changes
- [ ] #2 The Study section uses three rows, with the carry-over hint in the staging canvas
- [ ] #3 A recorded decision on rail density (which rows show when a source is empty)
<!-- AC:END -->
