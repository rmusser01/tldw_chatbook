---
id: TASK-34916
title: The required PR Fast Lane runs within a minute of its 35-minute cap
status: To Do
created_date: 2026-10-12 22:22
labels:
- ci
- tests
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR Fast Lane is one of the jobs behind the required check, and it has a 35-minute cap. Over 34 pull-request runs measured on 2026-10-12 its median was 31.6 minutes and its slowest 34.0; one run on PR #3091 took 34 minutes 10 seconds. When a job reaches its cap GitHub cancels it, and a cancelled job leaves no test report, so the next few slow tests added to the lane will turn into failures with no explanation, which the merge queue then counts against the PR. The owner's rule for CI time is to add a shard rather than raise the minutes. Measured during the review of PR #3091 (TASK-34789).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The PR Fast Lane's slowest run over at least 20 pull-request runs leaves at least 5 minutes under its cap
- [ ] #2 No test is dropped from the required check to achieve it
- [ ] #3 The contract test that pins the lanes and their caps reflects the new layout
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
