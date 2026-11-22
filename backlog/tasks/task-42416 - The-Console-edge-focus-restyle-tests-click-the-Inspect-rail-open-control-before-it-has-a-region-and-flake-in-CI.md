---
id: TASK-42416
title: "The Console edge-focus restyle tests click the Inspect rail open control before it has a region and flake in CI"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_edge_focus_restyle_scope.py
- Tests/UI/test_console_left_rail.py
- Tests/UI/test_console_inspector_navigation.py
- Tests/UI/test_console_right_rail.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths start at the repository root.

**Why.** Both tests in `Tests/UI/test_console_edge_focus_restyle_scope.py` start with `assert await pilot.click("#console-inspector-rail-open")` (lines 83 and 158), straight after `make_console_pilot(size=(160, 45))` (from `Tests/UI/test_console_left_rail.py:42`) yields, with no wait for layout. `Pilot.click` returns False when the target has no region yet, so on a loaded runner the first assert fails before the test reaches what it pins. The file runs in "PR Fast Lane (admission-sensitive 2)" (`derived-artifacts.yml:818`). In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), that lane failed in 9 jobs, from 2026-11-22 04:05Z to 17:57Z, and every one failed on this file: runs 44974742439, 44974765316, 44974790480, 44977790794 attempts 2 and 3 (the two merge-queue attempts of #3653 it sank, jobs 137095986403 and 137178739290), 45016281720 attempt 2, 45034674868 attempt 2, 45050954944 and 45050972963. The same step logs `FAILED Tests/QA/test_project_chat_uat.py::test_project_chat_uat[fenced]` lines; those come from that step's nested negative-control pytest run and are expected, not failures of the job.

TASK-41632 fixed the same kind of fixed-wait flake in the `_open_console_inspector` helpers of two other files; this file was not one of them. `Tests/UI/test_console_inspector_navigation.py:116` (`_open_inspector`) and `Tests/UI/test_console_right_rail.py` click the same control the same way, without checking the result.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests wait until the Inspect rail's open control is displayed and has a region, with a load-tolerant deadline (or use an existing shared rail-open helper that does), before clicking it
- [ ] #2 Each test then confirms the rail opened before it measures restyles
- [ ] #3 With the control's first layout delayed by a temporary edit (reverted), the old tests fail at the click and the new ones pass
- [ ] #4 The file passes in a soak of at least 20 runs under parallel load, recorded in the notes
<!-- AC:END -->
