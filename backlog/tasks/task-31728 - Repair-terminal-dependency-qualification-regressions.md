---
id: TASK-31728
title: Repair terminal dependency qualification regressions
status: Done
assignee: []
created_date: '2026-09-05 04:50'
updated_date: '2026-09-05 04:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore terminal dependency-qualification coverage against the current probe and capability contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced terminal qualification failures pass
- [x] #2 Affected terminal qualification module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current dependency-qualification module failures. 2. Trace each failure to the current probe or capability contract and apply the smallest justified fix. 3. Run focused regressions and the full affected module. ADR required: no. ADR path: N/A. Reason: this is localized regression maintenance for existing terminal qualification behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replayed the historical CI failure cluster on the current branch. None of the old
failures reproduce: the full terminal dependency-qualification module passes with
200 tests. No source or test changes were necessary. ADR required: no; this task
records verification only.
<!-- SECTION:NOTES:END -->
