---
id: TASK-32066
title: >-
  Library landing canvas still paints at compact widths where the guide says it
  hides
status: In Progress
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 18:47'
labels:
  - library
  - docs
  - layout
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 the landing canvas ('Search everything…', counts, From your Library, Quick actions) is painted next to the 22-column rail; library.md says the landing is hidden at compact widths and the rail owns navigation. Harmless visually; docs and code disagree. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 library.md and the landing behaviour at compact widths agree (either hide the canvas or document that it stays)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test at (100,30): the landing canvas still paints beside the rail.
2. Hide the canvas host while the landing canvas is mounted below LIBRARY_NOTES_COMPACT_BREAKPOINT, so the existing RAIL_ONLY width contract gives the rail the columns.
3. Keep the wide route unchanged (regression test at 170x48).
4. Stamp Docs/User_Guide/library.md.
<!-- SECTION:PLAN:END -->
