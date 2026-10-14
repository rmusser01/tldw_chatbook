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
Tests/UI/test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session[maintenance] failed once in the required check on PR #3097's merge-queue run on 2026-10-13 and passed on the re-run; it also failed in a run of another branch that failed on its own changes, which says nothing about it. A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the intermittent failure is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive runs in CI, or fails only for a real defect
- [ ] #3 If the product was at fault, a test that fails deterministically without the fix pins it
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
