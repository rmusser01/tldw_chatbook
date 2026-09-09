---
id: TASK-32065
title: >-
  Library Media below 64 columns: the stage shows an empty Reader with no items
  and no '‹ Library' return
status: To Do
assignee: []
created_date: '2026-09-08 18:25'
labels:
  - library
  - media
  - layout
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 60x24 activating Media paints 'Select a media item to read it here.' with two '›' grips, no items list and no return control, although library.md says the single-stage mode returns via '‹ Library'. The media page notes a pending follow-up for this exit. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Below 64 columns the Media stage shows the Items list
- [ ] #2 A '‹ Library' (or '< Library' in ASCII mode) control is present and works by keyboard
<!-- AC:END -->
