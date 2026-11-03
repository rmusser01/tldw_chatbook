---
id: TASK-38584
title: Pair A/B evaluation samples by sample id
status: In Progress
created_date: 2026-11-02 14:26
labels:
- bug
- evals
priority: high
updated_date: 2026-11-02 14:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A/B merged views and confidence intervals zip results from the two concurrently running models by completion order rather than sample id so answers to different questions get compared. See bug-sweep-2026-11-02-dev.md section 1.4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Merged view and confidence intervals pair results by sample id
- [ ] #2 Mismatched or truncated run counts are handled explicitly
- [ ] #3 Regression test covers out-of-order completion
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
