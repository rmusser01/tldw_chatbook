---
id: TASK-32052
title: >-
  Library New-note canvas: nothing focused on entry; Enter/Down no-op; Tab
  leaves the screen
status: To Do
assignee: []
created_date: '2026-09-08 18:22'
labels:
  - library
  - notes
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing `n` opens the New-note canvas with the footer saying 'enter create note', but nothing has focus: Enter and Down do nothing, the first Tab walks into the top nav bar where Enter switches to Home, and 22 Tabs later focus is in the rail search box, never on Blank note. Creating the first note is mouse-only. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entering the New-note canvas focuses the Blank note control so Enter creates a note immediately
- [ ] #2 Up/Down move between Blank note and the templates with a visible cursor
- [ ] #3 Tab from any Library canvas stays inside the Library screen; the nav bar is reached only via its documented keys
- [ ] #4 The footer hint on the New-note canvas is truthful for the focused control
<!-- AC:END -->
