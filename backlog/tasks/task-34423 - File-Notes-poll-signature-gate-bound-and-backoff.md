---
id: TASK-34423
title: File Notes poll signature gate bound and backoff
status: In Progress
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 10:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 / F10: the 1.5s workspace poll does an unbounded full-tree walk with two lstats per file plus a full replica read per tick and session changes accumulate unboundedly
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unchanged vault performs zero replica reads per tick,Reconcile fires exactly once per real change,Session change list capped at 500,Poll backoff to 6s when idle,Existing sync tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 11 (T11)
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
