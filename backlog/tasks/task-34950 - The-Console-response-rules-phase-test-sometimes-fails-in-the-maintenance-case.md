---
id: TASK-34950
title: The Console response-rules phase test sometimes fails in the maintenance case
status: Done
created_date: 2026-10-14 13:47
labels:
- tests
- flaky
- console
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-14 15:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session[maintenance] failed in the required check on PR #3097's merge-queue run on 2026-10-13 19:37Z and passed on the same-commit re-run, and twice more on the branch of PR #3020 the same day (18:33Z and 19:20Z). PR #3020 then changed this test and the Console's credential polling during config maintenance (commits ab2c485f97 at 19:07Z and 90bf1daeee at 20:05Z) and reached dev at 22:58Z; the #3097 failure ran on a base without them. A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes 20 consecutive CI runs after PR #3020's changes, or, if it fails again, the cause is identified in the test or the product
- [x] #2 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Match each CI failure's line and tree to the test and code versions. 2. Check the current test pins the race. 3. Count attempts since PR #3020 reached dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed by PR #3020, no code change here. Every CI failure ran a tree without one of #3020's fixes, and the failing line numbers match the older test: the TimeoutError at line 412 (run 38447749999, and run 38454979335 attempt 2, which is PR #3097's 19:37Z queue run; #3097 merged without ab2c485f97) is the credential timer blocking on the cold rebuild lock during maintenance; the AssertionError at line 422 (run 38454091728) is the pre-90bf1daeee assertion on a chip that lagged the phase after Stop. The current test ticks the armed credential timer itself: with #3020's two chat_screen.py changes reversed (temporarily, nothing committed), both [maintenance] and [rebuild] fail with 'credential timer blocked the UI loop', 2 of 2. Since #3020 reached dev (2026-10-13 22:58Z): 20 completed attempts; the only failure of this test was in a run of perf/34386-34388-hot-path-efficiency that failed 56 tests on its own change (its _console_credential_poll_timer was None); 10 of 10 local runs of all four parameters pass.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
