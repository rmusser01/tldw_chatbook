---
id: TASK-42463
title: "Console rail tests assert an open or quiet Inspect rail after a fixed pause instead of waiting for it"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 09:41
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
- `Tests/UI/test_console_rail_reconciliation.py`: three tests assert `inspector._outer_reconcile_scheduled is False` (with display or scroll checks) after a fixed `_settle(pilot, passes=8)` or `passes=10`, in `test_inspector_geometry_only_generation_reconciles_without_owner_pass`, `test_inspector_section_collapse_still_runs_the_full_reconcile` and `test_scroll_cost_probe_still_detects_pre_split_routing` (about lines 1515, 3280 and 3356 since TASK-42474 on `fu/r22-railrecon` 6c03bf6c6d; they were 1441, 3185 and 3261). TASK-42415's notes judged these lower risk: they request the reconcile synchronously, so the vacuous wait it fixed does not apply, and none failed in its 140-run CI scan.

No CI failure has been seen for any of these. This is hardening against the failure shape round 22 fixed twice.

**Update 2026-11-23 (TASK-42474).** TASK-42474 fixed the same shape in two other tests in `test_console_rail_reconciliation.py` with `_wait_for_inspector_quiet` (rail open with nothing scheduled, the owner-pass count stable and past a threshold, and the Console's own startup syncs quiet, on two consecutive polls). The sites above, and more, reach the rail through `_open_production_inspector` (about lines 1245-1254): it presses the open control and settles eight bare pauses. Its press branch is the single fix point. Its callers that then assert a quiet scheduler or an owner-pass count are exposed to the startup syncs TASK-42474 found landing late:
- `test_inspector_geometry_only_generation_reconciles_without_owner_pass`
- `test_inspector_owner_demand_latches_while_geometry_generation_pending`
- `test_inspector_owner_demand_survives_hint_toggle_continuation`
- `test_inspector_unmount_before_callback_clears_pending_generation`
- `test_pure_inspector_scroll_never_relayouts_the_rail`
- `test_inspector_section_collapse_still_runs_the_full_reconcile`
- `test_scroll_cost_probe_still_detects_pre_split_routing`

The scheduler-capture tests patch `call_after_refresh` and expect their own request to be the first scheduled, so a startup sync still scheduling a pass breaks them. The no-owner-pass halves (the geometry-only test's asserts at about lines 1542-1546, and the pure-scroll test's unchanged owner-pass count at about line 3232) need a variant of the wait without the `owner_passes_after` threshold: they assert that no owner pass happens, so the wait must not require one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every site in `test_console_right_rail.py` that clicks the open-rail control and then waits only a fixed pause (found by a grep, and listed in the notes) waits instead, with a load-tolerant deadline, until the rail is open
- [ ] #2 `_open_production_inspector` waits for the rail and the Console's startup syncs to stay quiet across consecutive checks, so each of its callers listed above starts from a quiet rail; or the notes record, per caller, why its fixed settle cannot be read too early
- [ ] #3 A click on the open-rail control that does not open the rail fails at the click, naming the widget the click reached, not at a later assert
- [ ] #4 The tests keep their assertions, and both files pass in repeated runs under parallel load (at least 10 runs each, or a soak recorded in the notes)
<!-- AC:END -->
