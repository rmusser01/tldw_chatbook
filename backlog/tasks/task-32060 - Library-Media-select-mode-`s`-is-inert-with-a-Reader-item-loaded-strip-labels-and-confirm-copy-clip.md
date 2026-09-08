---
id: TASK-32060
title: >-
  Library Media select mode: `s` is inert with a Reader item loaded; strip
  labels and confirm copy clip
status: To Do
assignee: []
created_date: '2026-09-08 18:24'
labels:
  - library
  - media
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With focus on an Items row and an item loaded in the Reader, `s` did nothing and the footer dropped 's select', forcing the mouse; the strip reads '○ Export / ○ Review / ○ Delete' and the delete confirm sentence is cut at the pane edge at the 36-cell Items floor. The pinning test covers only the no-item case. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `s` enters select mode from any focused Items row regardless of the Reader state, and the footer advertises it
- [ ] #2 The delete confirm sentence wraps instead of clipping at the Items floor
- [ ] #3 Overlap with task-32045 (zero-selection reason) and task-15140 (toolbar overflow) is reconciled in the notes
<!-- AC:END -->
