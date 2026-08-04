---
id: TASK-2081
title: 'Roleplay: stop the library toolbar clipping Import at supported widths (F-030)'
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 only New and Sort survive; Import, Duplicate, Tag are clipped while empty-state copy still says 'use New or Import'. Compact threshold is 90 so it never engages at 100. Evidence: roleplay-100x30.png, personas_screen.py:368. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All toolbar actions are reachable at 100x30 (wrap or overflow menu with New pinned),Rendered-layout regression test at 100x30 and 80x24
<!-- AC:END -->
