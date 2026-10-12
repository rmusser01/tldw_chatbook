---
id: TASK-34929
title: The Dreams tracking tests run in no pull-request lane
status: To Do
created_date: 2026-10-12 23:18
labels:
- ci
- tests
- dreams
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Dreams/test_track_service.py, about 200 tests of the Dreams tracking and switch logic, is in no group the required check runs; only the scheduled full-tree workflow runs it. Seven of its tests went red with the PR #2921 merge and stayed red unnoticed (fixed in the follow-up PR to #2921). One of its tests, test_overlapping_reconciles_report_settled_once_at_the_end, also waits on fixed 5 s and 10 s bounds and runs five times slower under load. The PR Fast Lane has almost no time left (TASK-34916), so adding the file needs room made first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Dreams tracking tests run on every pull request that changes Dreams or Subscriptions code
- [ ] #2 No required job comes closer to its time cap as a result
- [ ] #3 The overlapping-reconciles test waits on events rather than fixed bounds, or its bound has a measured margin under load
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
