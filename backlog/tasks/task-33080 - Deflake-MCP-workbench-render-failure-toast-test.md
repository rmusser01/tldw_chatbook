---
id: TASK-33080
title: Deflake MCP workbench render-failure toast test
status: Done
assignee: []
created_date: '2026-09-27 18:30'
updated_date: '2026-09-30 16:18'
labels:
  - ci
  - flaky-test
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_mcp_workbench.py::test_render_failure_in_show_tool_test_result_notifies_instead_of_only_logging` failed once in CI's PR Fast Lane on PR #2849 (2026-09-27 ~16:52Z) with `AssertionError: expected an error toast naming the tool, got: []`, even though the PR touched no MCP code. It passed twice locally on the same commit and passed on the next CI run. The failure matters beyond one PR: the required "Derived artifacts" check now requires PR Fast Lane to succeed, so one flake blocks every merge into dev until it is re-run.

The test clicks the tool's Run button, waits for workers to finish, and expects an error toast. An empty notification list means the render-failure path never ran or its toast landed after the assertion. Two unverified guesses: the click did not reach the button under CI load, or the result is delivered through a path that `app.workers.wait_for_complete()` does not wait on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The root cause of the empty-toast failure is identified and recorded, with evidence (a reproduction under load or a traced event ordering)
- [x] #2 The test waits on the condition it asserts rather than on timing, and still fails if the render-failure toast is removed from production code
- [x] #3 The test passes 50 consecutive runs under CPU load (for example with pytest-repeat, or parallel runs on a loaded runner)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Duplicate of TASK-33211 (filed a day later for the same test); fixed there. Root cause with evidence is in TASK-33211's notes: `pilot.click` on Run returned False in every reproduced failure (the button moves while the test panel's preview and scroll settle), not a worker-wait race.
<!-- SECTION:NOTES:END -->
