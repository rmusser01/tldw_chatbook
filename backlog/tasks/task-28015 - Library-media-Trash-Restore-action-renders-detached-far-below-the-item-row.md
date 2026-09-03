---
id: TASK-28015
title: Library media Trash - Restore action renders detached far below the item row
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:08'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). Layout defect confirmed: the Trash view renders the item row near the top (pane rows 10-11) and pins its controls (pager + "Restore Delete permanently") to the panel bottom (~rows 47-49) with ~36 empty rows between - Restore is visually detached from the row it acts on. Two updates since filing: "Delete permanently" now EXISTS (task-15130's gap shipped), and the Trash ENTRY button itself is one of the clipped-invisible toolbar buttons (see the toolbar-clipping task from the same run); no palette command reaches Trash either ("trash" query returns only tab-nav entries).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restore is visually associated with the trashed item list (no dead gap)
- [ ] #2 A keyboard path from a trash row to Restore exists
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (not started — layout geometry + a11y): AC#1 (Restore not detached from the row list) is caused by the deliberate 'height: 1fr' on #library-media-trash-list (library_media_trash_canvas.py:361) which fills all available height even with few rows, docking the pager+Restore controls at the panel bottom with a big gap. The 1fr was chosen (comment :353-360) so a 200-item trash page does not push controls off a 24-row terminal. Fix is 'auto up to available' (height:auto + max_height bounded so it scrolls only when tall) -- same class as deferred 28010, needs LIVE geometry verification (the L3a clipping lesson / geometry-pilot rule), not a blind CSS edit. AC#2 (keyboard path from a trash row to Restore) needs a binding/focus-order addition. Recommend pairing with 28010's viewer scroll-model work.
<!-- SECTION:NOTES:END -->
