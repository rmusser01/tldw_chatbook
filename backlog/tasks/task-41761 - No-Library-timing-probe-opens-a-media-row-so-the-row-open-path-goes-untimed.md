---
id: TASK-41761
title: "No Library timing probe opens a media row, so the row-open path goes untimed"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- performance
priority: low
references:
- Helper_Scripts/library_click_probe.py
- Helper_Scripts/library_switch_teardown_probe.py
- backlog/docs/library-decomposition-recipe.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The media series' timing criterion (TASK-36952.3.8 AC #5 and parent TASK-36952.3 AC #4) used the recipe's two instruments:
- `Helper_Scripts/library_click_probe.py` (recipe §9);
- `Helper_Scripts/library_switch_teardown_probe.py` (recipe §26).

Both switch rail modes, and neither presses a media row. The row-open path, `handle_library_media_row` -> `_refresh_library_media_detail` -> the viewer sync, was therefore never timed, although slices 3.3 and 3.6 moved it into the controller. The ruling that chose these probes named the cost: "a media-row click regression goes unmeasured". The 3.8 notes record the gap (2026-11-18). The owner waived AC #5 on the rail-switch data alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Helper_Scripts probe opens a media row headless and reports the press-to-painted-detail time with its recompose, full-update and mount counts, in the click probe's report format
- [ ] #2 The recipe says when a move must run it, alongside §9 and §26
- [ ] #3 A paired, order-swapped base and head run of it on one dev commit is recorded in the notes as a baseline, with n and the load
<!-- AC:END -->
