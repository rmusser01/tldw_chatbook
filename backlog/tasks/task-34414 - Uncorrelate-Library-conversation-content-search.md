---
id: TASK-34414
title: Uncorrelate Library conversation content search
status: To Do
created_date: 2026-10-07 02:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 1 / F2: library conversation search uses a correlated EXISTS with leading-wildcard LIKE on messages content scanning the whole corpus per candidate row - the Console seam already removed this exact shape
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Results and hit projections byte-identical to baseline,Correlated EXISTS replaced by uncorrelated IN subquery,Trace evidence shows no correlated EXISTS,Note search LIKE branch decision documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 2 (T2)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
