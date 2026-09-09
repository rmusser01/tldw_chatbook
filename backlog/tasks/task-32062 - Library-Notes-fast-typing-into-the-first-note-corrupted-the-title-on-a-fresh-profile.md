---
id: TASK-32062
title: >-
  Library Notes: fast typing into the first note corrupted the title on a fresh
  profile
status: To Do
assignee: []
created_date: '2026-09-08 18:24'
labels:
  - library
  - notes
  - bug
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typing 'My first note', Tab, and a body within ~0.4 s produced the stored title 'Mhello from jordan, testing the libraryy first note' with an empty body; not reproducible on a populated profile. The likely cause is the graduation recompose ('Library tools are now available.' plus pane collapse) firing mid-typing and resetting the focused Input. Data-affecting. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing into a focused editor is never interrupted by a recompose or focus reset
- [ ] #2 A test types rapidly during the first-content graduation and asserts title and body land in the right fields
<!-- AC:END -->
