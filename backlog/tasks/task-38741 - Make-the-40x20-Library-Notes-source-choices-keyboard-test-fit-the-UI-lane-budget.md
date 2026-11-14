---
id: TASK-38741
title: "Make the 40x20 Library Notes source-choices keyboard test fit the UI lane budget"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 03:09
updated_date: 2026-11-05 22:19
labels:
- follow-up
- tests
- ci
- library
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-38695 put the File Notes workspace tests on the UI PR lane node by node, but left `Tests/UI/test_library_file_notes_workspace.py::test_library_notes_source_choices_render_and_switch_by_keyboard[size0]` off it: the test passes but takes 31-34 s locally (serial, `--timeout=180`), over the census's 30 s per-entry budget. At 40x20 it presses Tab up to 240 times to reach the source choices, so its cost is the keyboard walk, not the behaviour it checks. Until it fits, the 40x20 keyboard path for switching note sources is gated by no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The [size0] case still proves the source choices render and can be switched by keyboard alone at 40x20
- [x] #2 It runs in under 20 s serially locally, measured over 3 runs and recorded in the notes
- [x] #3 It is on the UI PR lane, appended at the census tail
<!-- AC:END -->

## Implementation Plan

1. Time `[size0]` serially on this branch, after TASK-38757, and on the base workspace source. Find where its time goes: the Tab walks.
2. If the walks still cost too much, shorten them without dropping any keyboard step. Otherwise leave the test as it is.
3. Append the node at the UI PR gate census tail and raise the floor.

## Implementation Notes

- The test is unchanged. It still renders both source choices at 40x20, reaches Database and Files with Tab, switches with Enter, and comes back with Shift+Tab and Enter (AC#1).
- The cost was the focus chain, not the test. At 40 columns the navigator left the work pane a 1-cell sliver, and every control in it stayed in the focus chain, so the three Tab walks crossed them all. TASK-38757 (this branch) hides that sliver. A probe copy of the test counted 11, 7 and 4 presses for its three walks (database, search, files), and the whole case took 6.4 s.
- Timing (AC#2), `run.sh ... pytest '<node>[size0]' -q -p no:randomly --durations=3 --timeout=180`, serial, one process per run:
  - this branch, 3 runs: 4.88 / 5.09 / 5.29 s call, 6.6-6.95 s per session, 8.5-9.1 s wall (load 7.9-8.5 on 18 CPUs);
  - base workspace source (`29592ecc9b`), 2 runs: 33.66 / 32.57 s call, which matches TASK-38695's 31-34 s.
- The node is appended at the census tail (AC#3), and the floor is raised to match. It depends on TASK-38757's fix, so both must land together.
- Files: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

