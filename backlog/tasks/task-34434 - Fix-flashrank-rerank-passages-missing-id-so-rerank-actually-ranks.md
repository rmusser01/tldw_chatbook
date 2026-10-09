---
id: TASK-34434
title: Fix flashrank rerank passages missing id so rerank actually ranks
status: To Do
created_date: 2026-10-08 00:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34413: rerank_results builds passages without id and validates ranked.index against real flashrank dicts (which return input dicts with score added) - AttributeError is swallowed and every real rerank silently falls back to original unranked order. The feature has likely never worked. Fix the validation/passages to the real flashrank contract with a live-path test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real flashrank output re-orders passages,Golden + fake-factory tests updated to the real contract,Fallback semantics preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 1 parked finding + task-1-report.md
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
