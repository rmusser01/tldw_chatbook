---
id: TASK-34427
title: Sargable SQL trio with timestamp normalization ADR-215
status: In Progress
created_date: 2026-10-07 02:42
dependencies:
- TASK-34419
updated_date: 2026-10-07 13:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 6 / F11: julianday keyset ordering datetime(next_review) filtering and json_extract visibility plus NOCASE sort all defeat existing indexes - fixing requires normalizing stored timestamp formats with a schema migration
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-215 written before code,Format audit test documents the split,Migration normalizes timestamps and bumps schema version,Writers emit canonical format only,EXPLAIN QUERY PLAN shows index use for all three paths,Ordering identical on mixed-format fixture,Migration idempotent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 15 (T15)
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
