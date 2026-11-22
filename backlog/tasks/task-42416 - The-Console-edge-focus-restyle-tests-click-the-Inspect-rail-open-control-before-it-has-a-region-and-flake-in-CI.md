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
priority: medium
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

**Why.** Both tests in `Tests/UI/test_console_edge_focus_restyle_scope.py` start with `assert await pilot.click("#console-inspector-rail-open")` (lines 83 and 158), straight after `make_console_pilot(size=(160, 45))` (from `Tests/UI/test_console_left_rail.py:42`) yields, with no wait for layout. `Pilot.click` returns False when the target has no region yet, so on a loaded runner the first assert fails before the test reaches what it pins. This sank two merge-queue attempts of #3653 in "PR Fast Lane (admission-sensitive 2)" (run 44977790794, jobs 137095986403 and 137178739290), and it also failed attempt 2 of run 45016281720. The file runs in that admission step (`derived-artifacts.yml:818`).

TASK-41632 fixed the same kind of fixed-wait flake in the `_open_console_inspector` helpers of two other files; this file was not one of them. `Tests/UI/test_console_inspector_navigation.py:116` (`_open_inspector`) and `Tests/UI/test_console_right_rail.py` click the same control the same way, without checking the result.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests wait until the Inspect rail's open control is displayed and has a region, with a load-tolerant deadline (or use an existing shared rail-open helper that does), before clicking it
- [ ] #2 Each test then confirms the rail opened before it measures restyles
- [ ] #3 With the control's first layout delayed by a temporary edit (reverted), the old tests fail at the click and the new ones pass
- [ ] #4 The file passes in a soak of at least 20 runs under parallel load, recorded in the notes
<!-- AC:END -->
