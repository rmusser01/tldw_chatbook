---
id: TASK-38596
title: Reject calculator keyword arguments instead of dropping them
status: To Do
created_date: 2026-11-02 14:28
labels:
- bug
- tools
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CalculatorTool silently discards keyword arguments so round with ndigits returns a numerically wrong answer from an always-on auto-approved tool. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Non-empty call keywords raise a clear ValueError
- [ ] #2 round and min with keyword arguments no longer return wrong numbers
- [ ] #3 Test covers keyword rejection
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
