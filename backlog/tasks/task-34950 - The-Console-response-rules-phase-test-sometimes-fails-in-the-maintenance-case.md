---
id: TASK-34950
title: The Console response-rules phase test sometimes fails in the maintenance case
status: To Do
created_date: 2026-10-14 13:47
labels:
- tests
- flaky
- console
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session[maintenance] failed in the required check on PR #3097's merge-queue run on 2026-10-13 19:37Z and passed on the same-commit re-run, and twice more on the branch of PR #3020 the same day (18:33Z and 19:20Z). PR #3020 then changed this test and the Console's credential polling during config maintenance (commits ab2c485f97 at 19:07Z and 90bf1daeee at 20:05Z) and reached dev at 22:58Z; the #3097 failure ran on a base without them. A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes 20 consecutive CI runs after PR #3020's changes, or, if it fails again, the cause is identified in the test or the product
- [ ] #2 If the product was at fault, a test that fails deterministically without the fix pins it
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
