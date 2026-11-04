---
id: TASK-38722
title: "The Console rail-prune tests run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/Chat/test_console_rail_state_prune.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-36518
- TASK-36532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03.

`Tests/Chat/test_console_rail_state_prune.py` has 9 test functions. They pin `_prune_console_rail_preferences`, the launch-time prune of Console rail preferences that TASK-36518 reworked. No PR lane runs the file:
- it is not in `scripts/ui_pr_gate_census.txt`, which takes only `Tests/UI`;
- it is not a target in any workflow.

Only `test.yml` (on a push to main or on dispatch) and the scheduled `nightly-deep.yml` collect it, with the whole `Tests/` tree. It passed at the base in one `-n 4` run through the scratch-profile wrapper.

A Tests/Chat file needs a workflow lane target. The owner allowed one in `derived-artifacts.yml` for TASK-36532 (`dreams-boundary`, :1092-1116).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file runs on a PR lane as a whole-file target, with its measured time in that lane recorded in the task notes
- [ ] #2 It passes 3 of 3 runs in that lane's configuration before it is added
<!-- AC:END -->
