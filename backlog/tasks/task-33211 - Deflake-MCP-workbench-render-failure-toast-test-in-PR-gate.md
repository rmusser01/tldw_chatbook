---
id: TASK-33211
title: Deflake MCP workbench render-failure toast test in PR gate
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 08:20'
updated_date: '2026-09-30 16:20'
labels:
  - ci-throughput-3-candidate
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_mcp_workbench.py::test_render_failure_in_show_tool_test_result_notifies_instead_of_only_logging` failed in the required PR Fast Lane on docs-only PR #2875 (run 36393146890, 2026-09-28: `expected an error toast naming the tool, got: []`, 1 failed / 1170 passed). It passed 5/5 locally on the same tree and passed the same lane on #2763/#2719/#1651 at the same time, so it is load-dependent. Likely race: the test clicks Run, then `app.workers.wait_for_complete()` and a single `pilot.pause()`; if the click has not started the worker yet the wait returns immediately, and one pause is not enough for the toast under CI load. A flaky test in the required gate costs a full CI re-run per flake. Candidate for CI throughput sub-project 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test waits for the outcome it asserts (bounded poll for the toast) instead of assuming one pause suffices
- [x] #2 The test still fails when the production toast is removed (negative control recorded)
- [x] #3 The test passes repeatedly under CI-like parallel load (e.g. many runs under xdist)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace Run click -> worker -> _show_tool_test_result -> toast; establish the race with evidence (reproduce under load or instrument ordering)
2. Make the test wait for the asserted outcome (bounded poll for the error toast) instead of one wait_for_complete + pause
3. Negative control: remove the production toast, confirm the test fails
4. Repeat under parallel load (xdist) to show it no longer flakes
5. Close duplicate TASK-33080 against this fix
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (evidence, not the guessed timing race): 60 parallel copies of the test under xdist at load 17-26 reproduced the CI signature ('got: []') in 1-7 of every 60 runs. Instrumenting the failures showed `pilot.click(run_button)` returned **False** in every one -- Run was not under the pointer -- so no press, no run (service prepare/test calls 0), no toast; the button, preview, form value and app were all fine. Opening the test panel moves Run (row 12 -> 5 at 120x40) while its preview worker and focus scroll settle, and the test aimed after a single pause. (The single wait_for_complete() after the click also never saw the run worker -- 0 workers in all 60 runs, passing or not -- which is why the toast wait is now a bounded poll.)

Fix (test-only, Tests/UI/test_mcp_workbench.py): after opening the panel, wait for workers and scheduled animations before typing and aiming; assert the Run click landed (a future miss names itself); poll up to 40 pauses for the error toast.

Evidence: before, 33 failures in 660 parallel runs (5%); after, 600/600 passed at load 23-26. Negative control with the production toast removed: 120/120 fail on 'expected an error toast', 0 missed clicks. Whole file: 350 passed. Also resolves duplicate TASK-33080 (its AC#1 root-cause evidence is above).
<!-- SECTION:NOTES:END -->
