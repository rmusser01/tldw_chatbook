---
id: TASK-32060
title: >-
  Library Media select mode: `s` is inert with a Reader item loaded; strip
  labels and confirm copy clip
status: Done
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 20:59'
labels:
  - library
  - media
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With focus on an Items row and an item loaded in the Reader, `s` did nothing and the footer dropped 's select', forcing the mouse; the strip reads '○ Export / ○ Review / ○ Delete' and the delete confirm sentence is cut at the pane edge at the 36-cell Items floor. The pinning test covers only the no-item case. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `s` enters select mode from any focused Items row regardless of the Reader state, and the footer advertises it
- [x] #2 The delete confirm sentence wraps instead of clipping at the Items floor
- [x] #3 Overlap with task-32045 (zero-selection reason) and task-15140 (toolbar overflow) is reconciled in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: s from a focused Items row with an item loaded (100x30) is inert; the confirm copy is cut mid-word at the Items pane's narrowest.
2. Fix _library_media_list_surface_active: focus inside the Items pane owns the list behaviours regardless of the Reader's exit availability.
3. Remove LibraryMediaCanvas's min_width entirely: any floor above the slot the pane hands the canvas (pane width minus its 4 cells of padding) overflows it and clips every child instead of ellipsising. The 1fr width is bounded by the pane already.
4. GREEN + covering files; docs stamp; live-verify at 100x30.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two one-place fixes, both root-cause.

1. AC#1 -- `_library_media_list_surface_active` (library_media_controller.py) asked the Reader's exit availability BEFORE it asked where focus was, so in every layout that keeps a real Reader exit (Library collapsed, Items beside the Reader -- 100x30) a focused Items row was not 'the list surface': `s` was inert and the footer's `s select` chip vanished with it. Focus inside the Items pane now wins outright; the exit check only decides the no-focus case. One predicate, so the key gate, the footer chip and the select-mode footer set all move together.

2. AC#2 -- the confirm copy already wrapped (task-30043), but `LibraryMediaCanvas` pinned `min_width = 36` while the Items pane hands the canvas PANE WIDTH MINUS 4 (the pane's own padding): a 36-cell pane gives 32, and the resolver's real floor (a 32-cell pane, ITEMS_MIN_WIDTH) gives 28. A floor above the slot overflows it and clips every child by the pane edge with no ellipsis ('Delete 2 selected items? You c'), which is the same defect task-30043 fixed one step up (40 -> 36). The min_width is REMOVED, not lowered -- `ITEMS_MIN_WIDTH` (32) would still overflow the 28-cell slot at the floor, and the `1fr` width that replaced the old `13fr` is bounded by the pane already, so the floor that once guarded that resolution has nothing left to guard. Measured at panes of 32/36/40 the canvas is now 28/32/36 and rows ellipsise. (Fix round 1: this note previously described lowering the floor to an imported ITEMS_MIN_WIDTH -- an earlier version of the change that did not ship.)

AC#3 (overlap): the strip's '○ Export / ○ Review / ○ Delete' zero-selection reason belongs to task-32045 and the wide-layout toolbar overflow to task-15140 -- neither is touched here, so both can land without conflicting with this task's entry gate and canvas floor.

Files: tldw_chatbook/UI/Library_Modules/library_media_controller.py, tldw_chatbook/Widgets/Library/library_media_canvas.py, Tests/UI/test_library_crit8_polish_media.py (new), Tests/UI/test_library_media_render_fixes.py (36-cell floor pin extended to the ellipsis), Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
