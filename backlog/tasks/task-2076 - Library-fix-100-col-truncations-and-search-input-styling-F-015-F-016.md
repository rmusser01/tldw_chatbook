---
id: TASK-2076
title: 'Library: fix 100-col truncations and search input styling (F-015, F-016)'
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100 cols 'Conversations (0)' ellipsizes its count and the search placeholder truncates to 'Search'; the input renders as a borderless black void with stray artifacts. Evidence: library-100x30.png, library_screen.py:15581. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Row counts are never ellipsized at 100 cols,Search placeholder renders fully,Input reads as a field, not a void,Rendered-layout test at 100x30
<!-- AC:END -->
