---
id: TASK-35305
title: 'A CI check keeps bootstrap_profile tests out of the PR Fast Lane''s sandboxed step'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- follow-up
- ci
- tests
dependencies: []
references:
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_console_p0_regression_gate.py
- .github/workflows/derived-artifacts.yml
- backlog/tasks/task-32873 - Fix-five-red-test_console_runtime_ownership-tests-on-dev-fleet-wake-rename-tombstone-order.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32873's rule is that a `bootstrap_profile` suite runs in the PR Fast Lane's admission-sensitive invocation (`ADMISSION_TARGETS`, job `pr-fast-lane-admission`), never in the sandboxed step (`FAST_LANE_TARGETS`): its profile enrollment poisons the sandboxed suites that share its pytest process. PR #3144's review round 1 put bootstrap wizard nodes in the sandboxed step anyway, and nothing failed until a reviewer read the list.

The detection exists: Tests/CI/test_console_p0_regression_gate.py's `_uses_bootstrap_profile` reads the conftest's filename set and node-level marks, and `test_bootstrap_profile_p0_tests_stay_out_of_sandboxed_invocations` applies it, but only to the Console P0 table. Nothing applies it to `FAST_LANE_TARGETS`. Run over `FAST_LANE_TARGETS` at origin/dev 111f8e48fc it flags 4 of 92 entries, all whole files: Tests/MCP/test_control_plane_bridge.py, Tests/MCP/test_live_server_request_wiring.py, Tests/UI/test_mcp_workbench.py and Tests/UI/test_mcp_tools_mode.py. Whether each of those is a real violation or safe for a reason the rule does not see is not known yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/CI fails when a `FAST_LANE_TARGETS` entry (a file or a node id) selects a `bootstrap_profile` test, and the message names the entry and the admission-sensitive step
- [ ] #2 Each of the four entries flagged at origin/dev 111f8e48fc is moved to `ADMISSION_TARGETS`, narrowed to nodes that are not bootstrap_profile, or kept on a reviewed allowlist with its reason
- [ ] #3 A negative control (a bootstrap_profile node added to `FAST_LANE_TARGETS`) fails the new check, recorded in the notes
<!-- AC:END -->
