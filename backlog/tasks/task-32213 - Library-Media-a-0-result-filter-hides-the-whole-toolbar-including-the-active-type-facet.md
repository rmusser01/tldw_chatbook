---
id: TASK-32213
title: >-
  Library Media: a 0-result filter hides the whole toolbar including the active
  type facet
status: To Do
assignee: []
created_date: '2026-09-10 14:53'
labels:
  - library
  - media
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With `type: pdf` and a filter that matches nothing the canvas keeps only the title, the filter box and the miss sentence; `type: pdf`, `sort:`, `Export…`, `Trash`, `Select` and `Review these` are gone, so the type that produced the empty page cannot be seen or reset from the canvas (guide: 'A filtered empty page keeps its submitted type, query, or collection visible until you choose the reset action'). Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The toolbar (at least the type facet, sort and Trash) stays visible on a 0-result page
- [ ] #2 The miss sentence names the active type when one is set, with a reset action
<!-- AC:END -->
