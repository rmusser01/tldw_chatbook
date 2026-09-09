---
id: TASK-32066
title: >-
  Library landing canvas still paints at compact widths where the guide says it
  hides
status: Done
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 21:47'
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
- [x] #1 library.md and the landing behaviour at compact widths agree (either hide the canvas or document that it stays)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test at (100,30): the landing canvas still paints beside the rail.
2. Hide the canvas host while the landing canvas is mounted below LIBRARY_NOTES_COMPACT_BREAKPOINT, so the existing RAIL_ONLY width contract gives the rail the columns.
3. Keep the wide route unchanged (regression test at 170x48).
4. Stamp Docs/User_Guide/library.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REVERSED DIRECTION after evidence. AC#1 asks for docs and behaviour to AGREE, 'either hide the canvas or document that it stays'. I first implemented the hide (which the brief asked for), and the whole-file test_library_shell.py run then failed three geometry tests that pin the opposite -- including test_library_returning_landing_geometry_keyboard_and_compact_focus_stability, which keeps a focused Continue button through a resize to 100x30 and would lose that focus into a hidden pane.

git log settled it: commit 1a6c293761 'fix(library): keep compact landing alongside rail' (2026-08-26) DELIBERATELY took the landing out of compact single-stage and added those pins. library.md's sentence is the stale artifact -- it was never updated when that fix landed, and the critique itself rated the disagreement 'Harmless visually'.

So the code is unchanged and Docs/User_Guide/library.md now describes what the app does: at compact widths the landing stays beside the rail, both panes kept, focus preserved across the resize; only Notes' own workflow routes fold to one pane. A test pins the two-pane result at 100x30 so the guide and the behaviour cannot drift apart again. Confirmed live at 100x30. Files: Docs/User_Guide/library.md, Tests/UI/test_library_crit8_polish_shell.py (no production change).
<!-- SECTION:NOTES:END -->
