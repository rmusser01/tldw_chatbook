---
id: TASK-34814
title: Stopping the Console transcript poll while it is removing rows cancels the
  removal
status: To Do
created_date: 2026-10-12 05:26
labels:
- console
- stability
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console's transcript poll runs its paint inside the timer's own task. When the timer is stopped while a paint is waiting for transcript rows to be removed, the cancellation reaches the tasks of the rows being removed. In a test this surfaces as a cancellation error at teardown (one Console blocked-send test failed 8 times in 137 local runs under heavy load, on the branch and on the code before it alike, and never in the saved CI logs; its file is not in the PR gate). The same stop happens in the running app when the user leaves the Console while a paint is in that state; what the user would see then was not checked. Found while investigating flaky tests for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stopping the transcript poll during a row removal lets the removal finish or ends it without an error
- [ ] #2 Leaving the Console while a paint is removing rows leaves the transcript consistent on return
- [ ] #3 A test stops the timer during a removal deterministically and fails before the fix
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
