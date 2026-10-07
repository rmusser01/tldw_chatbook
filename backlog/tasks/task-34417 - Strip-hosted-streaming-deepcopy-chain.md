---
id: TASK-34417
title: Strip hosted-streaming deepcopy chain
status: To Do
created_date: 2026-10-07 02:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 / F8a: each streamed chunk on six hosted providers is deep-copied 5-9 times and JSON round-tripped twice
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mutation isolation test green,Per-chunk deepcopy count drops to 0-1,Perf microbenchmark recorded,Hosted provider streaming tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 5 (T5)
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
