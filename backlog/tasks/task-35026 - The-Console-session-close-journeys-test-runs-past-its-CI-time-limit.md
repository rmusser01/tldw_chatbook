---
id: TASK-35026
title: The Console session-close journeys test runs past its CI time limit
status: To Do
created_date: 2026-10-15 15:48
labels:
- tests
- ci
- console
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_session_tab_close.py::test_session_close_pending_race_and_fleet_journeys took about 183 seconds of test time in a required-check run on 2026-10-14 (run 38613866408), against the 180-second limit of its lane; its own timeout fired, app code swallowed the exception and the test passed, and the run only reported because the private-profile wrapper's longer wait covered it. Its passing runs had already been the slowest of the private-profile tests (81 to 127 s over 30 earlier samples). A test that routinely runs at its limit becomes a timeout failure on the next slow runner, which the merge queue counts against a pull request. Found in the review of PR #3117.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's runs in CI stay well inside its lane's limit over 20 pull-request runs
- [ ] #2 What it proves is unchanged (the journeys it covers still run)
- [ ] #3 If the test is split, every part runs in a required lane
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
