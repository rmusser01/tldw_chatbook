---
id: TASK-41529
title: "TASK-23149 and TASK-32310 acceptance criteria are partly met or stale after round 15"
status: Done
assignee: ['@claude']
updated_date: 2026-11-19 15:42
created_date: 2026-11-17 05:27
labels:
- follow-up
- backlog-hygiene
priority: low
references:
- backlog/tasks
dependencies:
- TASK-23149
- TASK-32310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (round 15). TASK-23149 should be narrowed to its two fake-row nodes, since 83c1eb5458 covers the rest. TASK-32310 AC#1 is probably met by d10bfafebb. Both need a read of the code before the criteria change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TASK-23149's criteria name only the two fake-row nodes
- [x] #2 TASK-32310 AC#1 is ticked with the commit as evidence, or the gap is written down
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read TASK-41127's commits on dev: find the merged equivalents of 83c1eb5458 and d10bfafebb (patch-id) and what each changed.
2. TASK-23149: run its fake-row and compose nodes at the base; narrow the criteria to the nodes still red, with a dated note naming the commit that met the rest.
3. TASK-32310 AC#1: run the rail-count node alone and in its file in default and reversed order (serial, run.sh); tick it with the commit as evidence, or write the gap down.
4. Task-file check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both tasks now say what is left. Neither is marked Done: TASK-23149's two nodes are still red, and TASK-32310's AC#2 is open.

**The commits.** Neither 83c1eb5458 nor d10bfafebb is on dev. Their rebased copies are: cbf233b179 for 83c1eb5458, and eeeacefa14 for d10bfafebb. Each pair has the same `git patch-id --stable`. Both copies came in through PR #3603 (merge b60535b101), and both are ancestors of c176f553b3. The task notes cite the dev shas and name the originals.

**AC#1, TASK-23149.** Run at c176f553b3 through run.sh: `Tests/UI/test_screen_navigation.py -k "focus_library_list_entry or compose"`, `-n 4`, load 20-25. Result: 2 failed, 4 passed.
- Red: `test_focus_library_list_entry_prefers_still_checked_row_in_select_mode` and `test_focus_library_list_entry_checked_row_preference_is_media_only`, both `AttributeError: … 'is_attached'` at `UI/Navigation/base_app_screen.py:208`. 29a03afb5a (TASK-34000.26) added that guard.
- Green: the two compose tests (cbf233b179 met the old third criterion) and the third fake-row test, which stubs `set_focus`.

A temporary `is_attached = True` on the fake row (reverted) moved both failures to `has_class`, at `library_screen.py:4546`: the focus write refreshes the footer, and its Enter-label lookup reads the focused widget's classes. The criteria now name only those two nodes. They require the fake row to carry what the focus path reads (`focusable`, `is_attached`, `has_class`, and any further attribute the run shows), and focus to be asserted by the widget `set_focus` targets. A dated note records the old criteria and why each changed.

**AC#2, TASK-32310 AC#1.** Ticked, with b6a388bee8 and eeeacefa14 (d10bfafebb) as the evidence. The test sets up its own app and captures. Runs at c176f553b3, serial, load 39-58:
- alone: 3/3 passed;
- the file in default order: 15 passed;
- the file in reversed node order: 15 passed.

Its AC#2 is untouched, so TASK-32310 stays To Do.

TASK-34916 and TASK-34917 were not edited. `scripts/check_backlog_task_files.py` passes.

Files: `backlog/tasks/task-23149 - …`, `backlog/tasks/task-32310 - …`.
<!-- SECTION:NOTES:END -->
