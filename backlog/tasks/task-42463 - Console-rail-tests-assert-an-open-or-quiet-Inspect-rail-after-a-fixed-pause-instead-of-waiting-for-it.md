---
id: TASK-42463
title: "Console rail tests assert an open or quiet Inspect rail after a fixed pause instead of waiting for it"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- console
- tests
- hardening
priority: low
references:
- Tests/UI/test_console_rail_reconciliation.py
- Tests/UI/test_console_right_rail.py
- Tests/UI/test_console_shell_regions.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42415's sibling check, widened by the filing's review. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths start at the repository root.

**Why.** Round 22 fixed two Console rail flakes by replacing a fixed pause with a confirmed wait. TASK-42415 waits for the Inspect rail to stay quiet, with nothing scheduled, across consecutive checks. TASK-42416 added `click_inspect_rail_open` (`Tests/UI/test_console_shell_regions.py:154`), which waits until the control is clickable and names the widget the click point reaches. The same shapes remain in two files that run in the same PR lane step (`.github/workflows/derived-artifacts.yml:961-962`):
- `Tests/UI/test_console_right_rail.py` clicks `#console-inspector-rail-open` 21 times. Only the click at line 553 then waits for a rail condition. Fifteen are followed by one bare `pilot.pause()` (lines 416, 453, 494, 864, 1033, 1062, 1134, 1164, 1188, 1220, 1278, 1312, 1367, 1454 and 1504; among them `test_sources_use_exact_twenty_line_content_ceiling` and `test_conditional_demotion_does_not_steal_newer_context_focus`), and five by a fixed `pilot.pause(0.2)` or `pilot.pause(0.3)` (lines 1400, 1434, 1626, 1672 and 1718). None of them confirms the rail opened before going on.
- `Tests/UI/test_console_rail_reconciliation.py`: three tests assert `inspector._outer_reconcile_scheduled is False` (with display or scroll checks) after a fixed `_settle(pilot, passes=8)` or `passes=10`, at about lines 1441, 3185 and 3261. TASK-42415's notes judged these lower risk: they request the reconcile synchronously, so the vacuous wait it fixed does not apply, and none failed in its 140-run CI scan.

No CI failure has been seen for any of these. This is hardening against the failure shape round 22 fixed twice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every site in `test_console_right_rail.py` that clicks the open-rail control and then waits only a fixed pause (found by a grep, and listed in the notes) waits instead, with a load-tolerant deadline, until the rail is open
- [ ] #2 Each of the three reconciliation sites waits for the rail to stay quiet across consecutive checks, or the notes record why its fixed settle cannot be read too early
- [ ] #3 A click on the open-rail control that does not open the rail fails at the click, naming the widget the click reached, not at a later assert
- [ ] #4 The tests keep their assertions, and both files pass in repeated runs under parallel load (at least 10 runs each, or a soak recorded in the notes)
<!-- AC:END -->
