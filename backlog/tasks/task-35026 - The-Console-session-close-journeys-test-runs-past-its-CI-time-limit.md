---
id: TASK-35026
title: The Console session-close journeys test runs past its CI time limit
status: In Progress
created_date: 2026-10-15 15:48
labels:
- tests
- ci
- console
updated_date: 2026-10-18 09:40
assignee:
- '@claude'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_session_tab_close.py::test_session_close_pending_race_and_fleet_journeys took about 183 seconds of test time in a required-check run on 2026-10-14 (run 38613866408), against the 180-second limit of its lane; its own timeout fired, app code swallowed the exception and the test passed, and the run only reported because the private-profile wrapper's longer wait covered it. Its passing runs had already been the slowest of the private-profile tests (81 to 127 s over 30 earlier samples). A test that routinely runs at its limit becomes a timeout failure on the next slow runner, which the merge queue counts against a pull request. Found in the review of PR #3117.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's runs in CI stay well inside its lane's limit over 20 pull-request runs
- [x] #2 What it proves is unchanged (the journeys it covers still run)
- [x] #3 If the test is split, every part runs in a required lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Time each of the test's five scenarios on its own.
2. Split them into two private-profile tests of about the same time, scenarios unchanged.
3. Check the file's lanes keep running both parts; measure CI after merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Split test_session_close_pending_race_and_fleet_journeys into test_session_close_pending_journeys (the two background pending-close scenarios) and test_session_close_race_geometry_and_fleet_journeys (chat-create race, 80x24 geometry, fleet retry), sharing a _run_scenarios helper that keeps the old loop (fresh patch context per scenario, the same drains and gc). Every scenario, its arguments and its order within its test are unchanged.

Timing (local, each scenario as its own private-profile test, child start-up included): 24.5, 12.5, 8.1, 14.6 and 13.8 s; the old test 54-58 s. After the split: pending journeys 25-39 s, race/geometry/fleet about 38 s, so each half carries about half of the old time. On CI's slower runners the old test took 81-127 s and once about 183 s; halves of that are well inside the 180 s limit.

Lanes: the file runs whole in the UI Fast Lane (census entry Tests/UI/test_console_session_tab_close.py, shard 8 of 10) and in the Console P0 gate (Tests/CI/test_console_p0_regression_gate.py pins the file), so both parts run in required lanes (AC#3); no census, workflow or node-id pin names the old test.

Verifying the split turned up a product race in one scenario: 1 failure in 6 loaded runs of the pending-journeys test, a Console worker raising NoMatches for the project-instruction status button. That is TASK-35413, fixed in the same PR; 8 of 8 loaded runs passed after it.

AC#1 stays open: it needs 20 pull-request runs after merge. Verify with the CI jobs' logs: both new tests' durations well under 180 s.

Files: Tests/UI/test_console_session_tab_close.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
