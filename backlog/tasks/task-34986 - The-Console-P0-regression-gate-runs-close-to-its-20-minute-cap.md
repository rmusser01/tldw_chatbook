---
id: TASK-34986
title: The Console P0 regression gate runs close to its 20-minute cap
status: To Do
created_date: 2026-10-14 16:13
labels:
- ci
- tests
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console P0 regression gate is one of the jobs behind the required check, with a 20-minute cap. Over the PR runs measured on 2026-10-14 its median was 13.0 minutes and its slowest 18.7; it took 16 minutes 30 seconds on PR #3106's first run. When it reaches its cap GitHub cancels it and leaves no test report, and the merge queue counts that against the PR. TASK-34916 and TASK-34917 made room in the PR Fast Lane and the UI shards but did not cover this gate. The owner's rule for CI time is to add a shard rather than raise the minutes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The gate's slowest run over at least 20 pull-request runs leaves at least 4 minutes under its cap
- [ ] #2 No test is dropped from the required check to achieve it
- [ ] #3 The contract tests that pin the gate reflect the new layout
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
