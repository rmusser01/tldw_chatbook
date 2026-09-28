---
id: TASK-33211
title: Deflake MCP workbench render-failure toast test in PR gate
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 08:20'
updated_date: '2026-09-28 17:44'
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
- [x] #1 The test's Run click deterministically reaches the tool-test worker (no click lost to an in-flight scroll), so the toast assertion tests the product, not the timing
- [x] #2 The test still fails when the production toast is removed (negative control recorded)
- [x] #3 The test passes repeatedly under CI-like parallel load (e.g. many runs under xdist)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce under CI-like load (throwaway parametrized wrapper, xdist -n 20)
2. Instrument the click to find where the chain breaks
3. Fix at the cause; verify with stress rounds + negative control
4. Sweep sibling raw clicks on the same button
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the test clicked Run with a raw `pilot.click(run_button)` right after clicking Test Tool, whose focus scroll animates the inspector. Under load the scroll is still in flight, the button moves (instrumented: region y=8 -> y=5) and the click misses (`pilot.click` returned False, Run enabled, service test_calls == 0) -- the worker never ran, so no toast. Commit 86efdced97 (2026-09-22) had moved 20 sibling tests onto `_click_test_run(pilot)` (settle scheduled animations, assert the click landed) but missed this one.

Fix: one line -- use `_click_test_run(pilot)`. AC #1 restated from the pre-investigation guess (poll for the toast) to the actual outcome, since polling could never have helped when the worker never started.

Evidence: reproduced locally 4/80 failures under xdist -n 20 (CI: 3/84 runs, all this test); after the fix 240/240 passes (3 rounds of 80). Negative control: disabling the product toast fails the fixed test with the original assertion. Whole module 350/350. Sibling raw clicks in Tests/UI/test_mcp_inspector.py (6) mount the inspector standalone with no scroll container: 240/240 under the same load, left unchanged.

Modified: Tests/UI/test_mcp_workbench.py.
<!-- SECTION:NOTES:END -->
