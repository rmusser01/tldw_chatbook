---
id: TASK-42415
title: "The right-rail live-work geometry test asserts a quiet rail after a pause and fails admission-sensitive 4 when an outer reconcile is scheduled again"
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
- Tests/UI/test_console_right_rail.py
- tldw_chatbook/UI/Console_Modules/right_rail.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), while checking which tests the required "PR Fast Lane (admission-sensitive 4)" job fails on now. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `.github/`.

**Why.** Every admission-4 failure from 2026-11-21 22:32Z to 2026-11-22 14:10Z is this one node, `Tests/UI/test_console_right_rail.py::test_live_work_widget_swaps_cover_real_twenty_twenty_one_geometry[readiness-to-pending-250-10-not_configured-15-21]`, with `AssertionError: (15, 46, 15, False, True)`. Runs: 44957243776, 44974726809, 44974749607, 44979357368, 45031398878. It passes locally at the base. The 2026-11-21 admission-4 reds that TASK-33620.15.61's notes list (`test_console_reference_custody.py` ×4, `test_change_turn_tracking.py` ×2) have not failed since about 21:17Z that day and pass locally.

**The failure.** The test waits with `_wait_for_right_rail_condition` until `initial_geometry_is_stable()` holds, then `await pilot.pause()` and asserts the same condition again (`test_console_right_rail.py:596-603`). The tuple's last field is `rail._outer_reconcile_scheduled`, so between the wait and the assert the rail scheduled another outer reconcile, while demand (15), width (46), viewport (15) and hint (False) were all as expected. On a loaded runner a late layout or refresh pass schedules a reconcile after the condition first held. The test also clicks `#console-inspector-rail-open` (line 546) without checking the click landed, which is the shape behind another flake, `test_console_edge_focus_restyle_scope.py`.

Whether the late reconcile is expected (a second pass after the swap) or a wasted one is not known; the fix has to say which.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The source of the outer reconcile scheduled after the initial geometry first holds is identified and recorded in the notes
- [ ] #2 If that reconcile is redundant, the product stops scheduling it (shown red first); if it is expected, the test waits for the rail to be quiet with nothing scheduled instead of asserting quiet after one pause
- [ ] #3 The test confirms the Inspect rail opened before reading its geometry
- [ ] #4 Both directions of the parametrized test pass in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
