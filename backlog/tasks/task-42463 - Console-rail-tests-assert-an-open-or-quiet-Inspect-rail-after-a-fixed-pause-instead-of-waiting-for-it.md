---
id: TASK-42463
title: "Console rail tests assert an open or quiet Inspect rail after a fixed pause instead of waiting for it"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 03:22
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
- [x] #1 Every site in `test_console_right_rail.py` that clicks the open-rail control and then waits only a fixed pause (found by a grep, and listed in the notes) waits instead, with a load-tolerant deadline, until the rail is open
- [x] #2 `_open_production_inspector` waits for the rail and the Console's startup syncs to stay quiet across consecutive checks, so each of its callers listed above starts from a quiet rail; or the notes record, per caller, why its fixed settle cannot be read too early
- [x] #3 A click on the open-rail control that does not open the rail fails at the click, naming the widget the click reached, not at a later assert
- [x] #4 The tests keep their assertions, and both files pass in repeated runs under parallel load (at least 10 runs each, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep every open-rail click in `Tests/UI/test_console_right_rail.py` followed only by a fixed pause; route each through TASK-42416's `click_inspect_rail_open` (load-tolerant wait until the rail is open), extended with a click offset and a failure that names the widget the click reached.
2. Make `_open_production_inspector` wait (TASK-42474's quiet wait, on TASK-42475's accessor) for the rail-open's owner pass and the Console's startup syncs, on consecutive polls.
3. Give the quiet wait a no-owner-pass mode and an awaited-state predicate, and replace the fixed settles before the quiet asserts in the geometry-only, section-collapse, scroll-cost and pure-scroll tests.
4. Name the race with forced delays (a delayed rail open; a delayed rail-open owner pass) that turn the old fixed pauses red and leave the new waits green; soak both files 20 times under load.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Race.** Each site read an open or quiet Inspect rail after a fixed pause. No CI failure has been seen for these sites (this task's filing found none in a 140-run scan), so this is hardening. Two forced delays show the fixed pauses can be read too early; each temporary edit was reverted, and each old file ran as a copy of the base file next to the new one:
- **Rail open 0.5 s late** (`on_console_inspector_rail_open` deferred through `set_timer(0.5, ...)`). The base `test_console_right_rail.py` failed 11 tests at their first post-open assert; the new file passed every test (11 failed, 63 passed in total). Examples: `test_sources_use_exact_twenty_line_content_ceiling`, `test_more_toggle_persists_without_programmatic_repost` and `test_inspector_root_pins_project_and_authority_above_outer_body[size0/1]`.
- **The open's outer reconcile held 3 s** (`_run_scheduled_outer_reconcile` re-deferring for 3 s from its first run with the rail displayed). The base copy failed 11 of its 12 `_open_production_inspector` callers and named-site tests. The scheduler-capture tests failed `assert 0 == 1` and `assert 0 == 2`: their request coalesced into the still-pending open. Others failed `_outer_reconcile_scheduled` (`assert True is False`), the counterfactual test `assert 10 == 9`, and collapse and pure-scroll `assert False is True`. The new file passed 12 of 12. With a 0.6 s hold the base copy passed at load about 45, so at that load the eight-pause settle's margin lies between 0.6 s and 3 s.

**AC #1, `test_console_right_rail.py`.** `grep -n 'console-inspector-rail-open\|pilot.click(open_button'` on base b31e3e9241 found 23 open-rail clicks:
- 15 followed by one bare `pilot.pause()`: lines 416, 453, 494, 864, 1033, 1062, 1134, 1164, 1188, 1220, 1278, 1312, 1367, 1454 and 1504;
- 7 followed by `pause(0.2)` or `pause(0.3)`: 959 (far-end click on the Button), 1001 (Button object), 1400, 1434, 1626, 1672 and 1718;
- line 553, which already waited for the rail but whose click could fail later.

Every one now calls TASK-42416's `click_inspect_rail_open`. It waits up to 15 s until the control is displayed, has a region and is the widget at its click point, clicks, then waits until the rail is displayed with a region. The far-end site passes its offset.

**AC #2, `_open_production_inspector`.** It presses the open control only if the rail is closed. It then waits with `_wait_for_inspector_quiet` for the rail to be laid out (`rail.region.width > 0`) and for the rail, its sections and the Console's startup work (TASK-42475's accessor) to be quiet on two polls in a row. The first version also required one owner pass after the press, as TASK-42474's tests do. Two callers then timed out with `Inspector still busy: ['owner pass']` (`test_bounded_rail_shell_regions_are_compositor_contained[160x45]` and `test_production_content_demand_counts_internal_box_only`). Both open the rail after `_open_all_production_context_sections`, and no owner pass came, so the helper requires none (lesson addendum in `lessons-testing-evidence.md`). `_wait_for_inspector_quiet` gained:
- `owner_passes_after=None`, which requires no owner pass while still waiting for the count to stop moving;
- `until`, the state an action produces, for an action that leaves the quiet state unchanged until its deferred work starts (TASK-42415's trap).

The fixed settles before the quiet asserts now use the wait:
- the geometry-only test: its request with the owner-pass threshold, and its layout half with no owner pass and `until` the hint shows;
- the pure-scroll test: its request, and its wheel burst with no owner pass and `until` the outer body reaches the bottom;
- the section-collapse test: its request, then the collapse and the re-expand with no owner pass and `until` the layout probe records a refresh;
- the scroll-cost probe: its request.

Two fixed pauses stay: the geometry-only test's final two-pause stability recheck, and the scroll-cost probe's later settles, which count layout refreshes in a mutation arm rather than assert a quiet rail.

**AC #3.** `click_inspect_rail_open` (shared helper in `Tests/UI/test_console_shell_regions.py`; the only shared-helper change) now takes `offset`. A click that misses fails with `the click at <point> reached <widget>`. A click that does not open the rail fails at the click with `Inspect rail never opened: region=...; the click at <point> reached <widget>`. Tested with the open handler made a no-op:
- the base tests failed later: `assert False is True` (`#console-inspector-more-body.display`) and `assert 0 == 20` (sources viewport height);
- the new tests failed in the helper: `Inspect rail never opened: region=Region(x=0, y=0, width=0, height=0); the click at Offset(x=150, y=6) reached Button(id='console-inspector-rail-open', ...)`.

`_open_production_inspector` presses rather than clicks, and its wait fails naming `rail closed`. Its importers, `test_console_edge_rail_geometry.py` and `test_console_edge_focus_restyle_scope.py`, pass with `test_console_shell_regions.py` (62 passed).

**AC #4, soak through `run.sh` at `-n 4`.**
- `test_console_right_rail.py` with `test_console_rail_reconciliation.py`: 20 of 20 runs green, 92 passed each, load 38-143 (several agents' runs plus this branch's other checks).
- The "Run admission-sensitive suites 4" step, which runs both files, serial with the lane flags: 600 passed.
- `test_console_shell_regions.py` with the helper's two importers: 62 passed.
- Final tip, after TASK-42475's review round: both files with `test_console_shell_regions.py` and `test_console_credential_poll_config.py`, 20 of 20 runs green (139 passed each) at load 18-52.

Every assertion is kept.

**Files.** `Tests/UI/test_console_right_rail.py`, `Tests/UI/test_console_rail_reconciliation.py`, `Tests/UI/test_console_shell_regions.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
