---
id: TASK-32063
title: >-
  Library Notes/Import status prose: duplicated status line, run-on header, and
  a graduation notice painted as a bare line
status: In Progress
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 19:41'
labels:
  - library
  - notes
  - copy
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Library notes · Library database · Ready · Next: Create a note or add from files.' appears in the list pane and again as the canvas header; the Add-from-files header is a 130-character run-on plus a second header; 'Library tools are now available.' stays as an unframed line all session and also fires on a populated profile's first visit (any transition into GRADUATED, including UNKNOWN to GRADUATED on the first source read). Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 14.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each Notes canvas paints one status line, with 'Next:' only when it names a control on screen
- [ ] #2 The graduation notice is a toast, fired only on a real compact-to-graduated transition
- [ ] #3 The Add-from-files header is one sentence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: work pane restates the list pane's authority sentence; Add from files stacks four headers; the graduation notice fires on a populated profile's first visit.
2. Give LibraryNotesCanvas an overridable authority prefix; the work pane returns empty.
3. Drop the work pane's own authority line for modes whose child canvas paints one.
4. 'Next:' only when it names a control.
5. One-sentence Add-from-files header.
6. Gate the graduation notice on STARTER -> GRADUATED.
<!-- SECTION:PLAN:END -->
