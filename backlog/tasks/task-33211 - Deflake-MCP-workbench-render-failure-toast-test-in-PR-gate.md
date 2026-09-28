---
id: TASK-33211
title: Deflake MCP workbench render-failure toast test in PR gate
status: To Do
assignee: []
created_date: '2026-09-28 08:20'
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
- [ ] #1 The test waits for the outcome it asserts (bounded poll for the toast) instead of assuming one pause suffices
- [ ] #2 The test still fails when the production toast is removed (negative control recorded)
- [ ] #3 The test passes repeatedly under CI-like parallel load (e.g. many runs under xdist)
<!-- AC:END -->
