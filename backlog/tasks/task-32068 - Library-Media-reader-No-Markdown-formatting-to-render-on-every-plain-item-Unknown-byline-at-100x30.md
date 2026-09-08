---
id: TASK-32068
title: >-
  Library Media reader: 'No Markdown formatting to render' on every plain item;
  'Unknown' byline at 100x30
status: To Do
assignee: []
created_date: '2026-09-08 18:25'
labels:
  - library
  - media
  - copy
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every plain-text item shows 'No Markdown formatting to render — showing the stored text' above the body, and at 100x30 an item without an author shows an 'Unknown' byline although the guide says the byline appears only when an author exists. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 19.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Markdown notice appears once in Info rather than on every render
- [ ] #2 No byline is painted for items without an author at any width
<!-- AC:END -->
