---
id: TASK-42414
title: "The Trace screen follow-suspend test checks follow before its scroll lands and fails UI Fast Lane 39"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
labels:
- follow-up
- ci
- tests
- flaky
priority: high
references:
- Tests/UI/test_trajectory_live.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** `Tests/UI/test_trajectory_live.py::test_scrolling_up_suspends_follow_until_f_resumes` is the most frequent red in the required UI Fast Lane. "UI Fast Lane (39)" failed in 11 PR runs from 2026-11-21 21:32Z to 2026-11-22 17:17Z (44952102973, 44953045328, 44954347282, 44973100546, 44974726809, 44974735998, 44974782181, 44976516789, 44978286545, 45027696251, 45048643145). The logs read (44973100546, 44978286545, 45027696251) all fail this node, as did the 2026-11-21 runs TASK-33620.15.61's notes list (44936495470, 44948612804, 44953045328). The file is a whole-file census entry, and the whole file (with the rest of the 205-test set checked) passes locally at the base.

**The failure.** `test_trajectory_live.py:263`, `assert screen._follow is False` → `assert True is False`. The test calls `table.scroll_to(y=0, animate=False)`, waits one `pilot.pause()`, then calls `screen._sync_follow_from_scroll()` (`UI/Screens/trajectory_screen.py:868-886`). That method returns early while `table.max_scroll_y <= 0`, and otherwise reads `table.scroll_y`. On a loaded runner one pause is not enough for the table's layout and the scroll to land, so the sync still sees the tail and follow stays on. The later `assert table.scroll_y == 0` and the `f` step have the same single-pause shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test waits on the table's geometry (scrollable and at the position it scrolled to) before each follow check, with a load-tolerant deadline, instead of a single pause
- [ ] #2 The new wait is shown needed: with the table's scroll delayed by a temporary edit (reverted), the old test fails at line 263 and the new one passes
- [ ] #3 The test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
- [ ] #4 The screen's own follow behaviour is unchanged: the file's other tests pass unedited
<!-- AC:END -->
