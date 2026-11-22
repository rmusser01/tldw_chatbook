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

**Why.** `Tests/UI/test_trajectory_live.py::test_scrolling_up_suspends_follow_until_f_resumes` is the most frequent red in the required UI Fast Lane. In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), "UI Fast Lane (39)" failed in 39 jobs, from 2026-11-20 09:47Z to 2026-11-22 20:49Z, and every one of them failed on this node alone (for example 44973100546, 44978286545, 45027696251, 45048643145). The file is a whole-file census entry, and the whole file (with the rest of the 205-test set checked) passes locally at the base.

**The failure.** `test_trajectory_live.py:263`, `assert screen._follow is False` → `assert True is False`. The test calls `table.scroll_to(y=0, animate=False)`, waits one `pilot.pause()`, then calls `screen._sync_follow_from_scroll()` (`UI/Screens/trajectory_screen.py:868-886`). That method returns early while `table.max_scroll_y <= 0`, and otherwise reads `table.scroll_y`. On a loaded runner one pause is not enough for the table's layout and the scroll to land, so the sync still sees the tail and follow stays on. The later `assert table.scroll_y == 0` and the `f` step have the same single-pause shape.

**Test drift or a product race.** It may not be only the test. When the screen re-renders while following, it queues `table.call_after_refresh(table.scroll_end, animate=False)` (`UI/Screens/trajectory_screen.py:858-866`). If such a pending scroll lands after the reader's own scroll up, the table snaps back to the tail: a real snap-back a user would see, not a test timing problem. The fix has to establish which it is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes record whether the failure is test timing or a pending tail scroll that lands after the reader scrolled up, with the evidence for it
- [ ] #2 A scroll up suspends follow and keeps the reading position on a loaded runner: the test passes without depending on how many refreshes run between its steps, and if the cause is the pending tail scroll, the product no longer snaps back
- [ ] #3 The change is shown needed: with the table's layout or scroll delayed by a temporary edit (reverted), the old test fails at line 263 and the new one passes
- [ ] #4 The test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
- [ ] #5 The file's other tests pass unedited
<!-- AC:END -->
