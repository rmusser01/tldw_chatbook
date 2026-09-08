---
id: TASK-32051
title: 'Library: Escape cannot leave a focused text box; the next key is typed into it'
status: To Do
assignee: []
created_date: '2026-09-08 18:22'
labels:
  - library
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With focus in the rail 'Search Library…' box or the Search/RAG query box, Escape is a no-op (footer stays 'typing in field') and the next printable key is inserted as text (`i` landed in the search box instead of opening Import). Only Tab/F6 leave the box, which is undocumented and breaks the keyboard-only journey. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape in the rail search box and in the Search/RAG query box moves focus to the canvas (or the documented target) without inserting text
- [ ] #2 The footer hint reflects the new focus after Escape
- [ ] #3 The next printable key after Escape performs its canvas action (for example `i` opens Import)
- [ ] #4 The behaviour is documented in library.md's Keyboard section
<!-- AC:END -->
