---
id: TASK-38411
title: 'Console: the collapsed paste chip wraps over three rows at 80x24'
status: To Do
assignee: []
created_date: '2026-10-31 02:37'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since the 2026-10-30 owner decision the chip names its key: 'Pasted text | N characters | Ctrl+E expand'. At the default 80x24 terminal the composer's draft area is about 26 columns wide, and the longer chip wraps over three rows of it, pushing the typed question out of view. Live capture during TASK-37870. A shorter chip form when the draft area is narrow would keep the token on one row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80x24 a collapsed paste chip takes one row of the draft area and still names Ctrl+E, pinned by a Pilot test at that size
- [ ] #2 At widths where the full chip fits, its copy is unchanged
<!-- AC:END -->
