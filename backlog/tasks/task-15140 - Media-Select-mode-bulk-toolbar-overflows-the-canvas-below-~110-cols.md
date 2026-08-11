---
id: TASK-15140
title: Media Select-mode bulk toolbar overflows the canvas below ~110 cols
status: To Do
assignee: []
created_date: '2026-08-11 13:33'
labels:
  - library
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by task-14900's AC#3 pins and A/B-verified pre-existing at base 345da0422 (independently re-measured at takeover: the toolbar's right edge is a fixed 111 cells at every compact width probed -- 100/108/110/112/114/119 -- so it overflows any terminal of 110 cols or narrower and fits from 111 up; byte-identical x/width per button at base and at the task-14900 tree). On a 100-col terminal the Media list's Select-mode bulk toolbar (count + Select all + Clear + Export selected + Delete selected, one non-wrapping Horizontal) lays 'Delete selected' at x=90 width=21, so only its first ~9 cells render; the tail is clipped at the canvas edge (observed live at 100x30: the strip ends in '○ Delet'). Keyboard press still works and the visible sliver is clickable, but the label is unreadable. Needs a real narrow-width treatment (shortened compact labels through the shared marker/label seams, or a two-row toolbar) -- a label fork must keep the in-place patchers (_apply_library_row_toggle, _patch_library_disabled_marker_label) in lockstep per the recompose-discipline rule. task-14900 pinned the current behavior in Tests/UI/test_library_media_side_by_side.py rather than blessing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Below the breakpoint every Select-mode bulk action renders fully on-screen at a 100-col terminal
- [ ] #2 Disabled markers and in-place label patching stay consistent with any compact label variant
<!-- AC:END -->
