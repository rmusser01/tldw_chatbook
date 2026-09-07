---
id: TASK-28015
title: Library media Trash - Restore action renders detached far below the item row
status: Done
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-04 20:48'
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
- [x] #1 Restore is visually associated with the trashed item list (no dead gap)
- [x] #2 A keyboard path from a trash row to Restore exists
- [x] #3 The Trash header no longer clips (Local Trash · 1 i) at the shell's pane width — critique #4 B cap_102
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: actions row under the last trash row (list→pager→actions pinned order, total gap ≤ 4); header paints 'Local Trash · 1 item' in full
2. GREEN: list height:1fr → auto with a post-layout cap (Textual resolves max-height:1fr against the container, so CSS alone cannot bound it); re-cap when the status fold appears; back button min_width 0 + singular 'item'
3. Live 235x52 + 100x30; resize + Tab-order tests
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Trash list was height:1fr, pinning the pager and Restore/Delete permanently at the pane bottom under ~36 blank rows. It is now height:auto with a cap measured after layout (_cap_trash_list: available minus the fixed-height siblings; non-oscillating because every sibling is fixed-height); the review found the cap was measured before the status fold showed, leaving Restore clipped one row past the pane in the full-page + folded-status posture — _update_status_fold now re-caps when it actually flips the fold. Header paints 'Local Trash · 1 item' in full (back button min_width 0, singular). Pinned: actions under the last row, resize 235x52→100x30 recaps, Tab from a row reaches Restore without passing an Input, full page + folding status keeps Restore inside. Deferred: one uncapped first frame before on_mount's hook; a decorative assertion.
<!-- SECTION:NOTES:END -->
