---
id: TASK-169
title: Prune legacy notes-window CSS and finish notes dead-code sweep
status: Done
assignee: []
created_date: '2026-07-11 22:03'
updated_date: '2026-07-11 23:52'
labels:
  - follow-up
  - tech-debt
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The F2 dead-code sweep pruned the notes-workbench selectors but left the pre-workbench legacy notes-window CSS: features/_notes.tcss, .notes-content-header/-label in _sidebars.tcss, and the Constants.py embedded duplicate. These have no live widget users, but #notes-window is still named in app.py's ALL_MAIN_WINDOW_IDS — adjudicate and prune. Also resolve the build_css.py 'Missing module: features/_evaluation_v2.tcss' warning surfaced during the sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Orphaned legacy notes-window CSS removed (or its live user documented)
- [ ] #2 ALL_MAIN_WINDOW_IDS no longer references a dead window id
- [ ] #3 build_css.py no longer warns about a missing _evaluation_v2.tcss module
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the quick-wins batch (branch claude/followups-quickwins). See Docs/superpowers/plans/2026-07-11-followups-quickwins.md.
<!-- SECTION:NOTES:END -->
