---
id: TASK-24605
title: Inspect rail fold indicator does not render at 80 columns
status: To Do
assignee: []
created_date: '2026-08-30 00:54'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The fold hint renders at 235 and 120 columns but never at 80, where measurement showed only 2 of 11 sections visible and 9 below the fold. At maximum scroll the last visible content was 'Artifacts: Connected -' clipped mid-sentence with no closing border and no hint. This is the exact failure the fold-indicator convention exists to prevent. Fixed non-scrolling chrome is 8 rows, 62 percent of the usable rail height at that size, so the hint row loses the space contest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The fold hint renders at 80x24 whenever Inspect rail content overflows
- [ ] #2 Rail content is never clipped mid-sentence as the only signal that more exists
- [ ] #3 Fixed non-scrolling chrome is reduced at narrow widths so scrollable content is not a minority of the rail
- [ ] #4 A test asserts hint presence at 80x24 with overflowing content and absence at scroll end
<!-- AC:END -->
