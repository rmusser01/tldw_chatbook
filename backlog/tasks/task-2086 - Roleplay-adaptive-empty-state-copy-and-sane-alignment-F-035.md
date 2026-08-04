---
id: TASK-2086
title: 'Roleplay: adaptive empty-state copy and sane alignment (F-035)'
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Empty copy renders 'use New or Import' even with characters present, centered in a huge void (reads broken, right-aligned at some widths). Evidence: personas_screen.py:263-269,870-874. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty copy adapts to whether the library has items,Copy alignment matches app conventions (left/centered deliberately),Tests updated
<!-- AC:END -->
