---
id: TASK-35444
title: The Console response-rules phase test still fails in the maintenance case
status: To Do
created_date: 2026-10-18 10:10
labels:
- tests
- flaky
- console
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session[maintenance] runs in a required UI Fast Lane shard. TASK-34950 fixed one cause of its intermittent failure, but it failed again on 2026-10-17 with a TimeoutError in run 39050477414 (attempt 1, PR #3113, which changed no Console or UI file), after PR #3137 had changed the same test's held-config setup. A failure on unchanged code counts against a pull request in the merge queue, and two in a row evict it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the remaining timeout is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive pull-request runs in CI
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
