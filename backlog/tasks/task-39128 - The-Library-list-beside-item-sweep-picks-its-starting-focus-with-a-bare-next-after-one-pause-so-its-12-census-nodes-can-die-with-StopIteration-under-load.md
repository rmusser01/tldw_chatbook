---
id: TASK-39128
title: "The Library list-beside-item sweep picks its starting focus with a bare next() after one pause, so its 12 census nodes can die with StopIteration under load"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- flaky
- library
priority: low
references:
- Tests/UI/test_library_adaptive_reader_closeout.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-39052
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in TASK-39052's notes (round-11 branch fu/r11-library). Line numbers are dev's.

- `_open_the_list_beside_the_open_item` (`Tests/UI/test_library_adaptive_reader_closeout.py:1851`) takes its starting focus with a bare `next(...)` over the open item's painted work-pane controls (`:1889`), right after a single `pilot.pause()`.
- Under load no work-pane control has painted yet, so the node fails with `StopIteration` (surfacing as `RuntimeError: coroutine raised StopIteration`) instead of waiting.
  - A round-11 run on the library branch recorded this failure.
  - TASK-39052's notes record `[media-50]` failing on base at load 80 and passing on a rerun.
  - 5 of 5 alone runs passed on dev at load 33-44.
- The helper serves both `test_opening_the_list_beside_an_open_item_leaves_no_work_sliver` and `test_every_other_list_destination_gives_an_opened_list_the_stage` (`:1984`). So all 12 census nodes at `scripts/ui_pr_gate_census.txt:2467-2478` carry the same race.
- This is the shape TASK-38959 fixed elsewhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the work pane's first paint held back by a temporary edit, the old helper fails with StopIteration, and the new one waits within a deadline and passes
- [ ] #2 All 12 census nodes that use the helper pass 10 of 10 at `-n 4` at load 40 or more, with times and load recorded
- [ ] #3 No check is weakened: the painted-cells requirement and the sliver and focus assertions stay
<!-- AC:END -->
