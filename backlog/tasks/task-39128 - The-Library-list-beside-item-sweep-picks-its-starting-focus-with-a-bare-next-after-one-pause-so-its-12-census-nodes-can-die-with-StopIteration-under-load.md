---
id: TASK-39128
title: "The Library list-beside-item sweep picks its starting focus with a bare next() after one pause, so its 12 census nodes can die with StopIteration under load"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-10 17:12'
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
- [x] #1 With the work pane's first paint held back by a temporary edit, the old helper fails with StopIteration, and the new one waits within a deadline and passes
- [x] #2 All 12 census nodes that use the helper pass 10 of 10 at `-n 4` at load 40 or more, with times and load recorded
- [x] #3 No check is weakened: the painted-cells requirement and the sliver and focus assertions stay
<!-- AC:END -->

## Implementation Plan

1. Hold the work pane's first paint back with a temporary test edit (the helper's `_work_cells` reads 0 for the first 0.5 s after the item takes the stage) and show the bare `next(...)` dies with StopIteration.
2. Replace the bare `next(...)` with a deadline wait (`_wait_for_condition`) for a painted work-pane control in the focus chain, then take that control; keep every painted-cells, sliver and focus assertion.
3. Under the same temporary hold, show the new helper waits and passes; revert the hold.
4. Run the 12 census nodes 10 times at `-n 4` at load 40 or more, recording times and load.

## Implementation Notes

**Approach.** `_open_the_list_beside_the_open_item` (`Tests/UI/test_library_adaptive_reader_closeout.py`) no longer takes its starting focus with a bare `next(...)` after one pause. It waits, with the file's own deadline helper (`_wait_for_condition`, 15 s), until the focus chain holds a work-pane control with painted cells, then takes that control. Every check stays: the painted-cells requirement (`_work_cells(target) > 0`), the work-pane floor, the no-unseen-control sliver check and both focus assertions are unchanged. Test-only change.

**AC#1, held first paint** (run.sh, one node, `[prompts]`, load 23-39). A temporary edit made `_work_cells` read 0 for the first 0.5 s after the item took the stage (reverted; `grep R13L` is empty). Old helper: `RuntimeError: coroutine raised StopIteration` at the bare `next` (1 failed). New helper under the same hold: `[prompts]` and `[media-40]` (ten widths, each held) 2 passed.

**AC#2, 10 of 10.** The 12 census nodes (`scripts/ui_pr_gate_census.txt`, the eight `test_opening_the_list_beside_an_open_item_leaves_no_work_sliver` cases and the four `test_every_other_list_destination_gives_an_opened_list_the_stage` cases), run together at `-n 4` ten times in a row: 10 of 10 runs, 12 passed each, 156-236 s per run, 1-minute load 46.7-94.0 at the starts of the runs (every run started at load 46 or more).

**Files.** `Tests/UI/test_library_adaptive_reader_closeout.py`.

