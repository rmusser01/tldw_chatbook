---
id: TASK-42459
title: "A Trace reader paused on the oldest rows widens the rendered window with every live event, and it never shrinks after f"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- trace
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/trajectory_screen.py
- Tests/UI/test_trajectory_live.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42414's second review. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Past 500 records each new turn drops the oldest rows, so a paused reader on them used to drift. TASK-42414 fixed that by growing the window to keep the reader's anchor row, as it already did for the cursor row: `_apply_live_snapshot` sets `_visible_count = max(_visible_count, window)` from `_window_for_key` (`UI/Screens/trajectory_screen.py:845-868`). The cost is that while that reader stays paused, every arriving event widens `_visible_count`, and live rebuilds render synchronously every 0.5 s. Nothing shrinks the window again after `f` resumes follow: it is reset to `min(_total_records, PAGE_SIZE)` only on a fresh load (line 400). So on a long live run the paused render keeps growing, and stays large after the reader returns to the tail. No lag has been measured; this task decides whether a bound is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A measurement over a long live run with a reader paused on the oldest rows (window size and render time per rebuild, recorded in the notes) decides whether the paused window needs an upper bound; if it does, the window has one and the anchor row stays in view
- [ ] #2 After `f` resumes follow, the rendered window returns to the page size at the tail
- [ ] #3 A test covers the shrink after `f` (and the bound, if one is added); it fails on the current screen code
- [ ] #4 The existing paused-reader and live Trace tests pass unedited
<!-- AC:END -->
