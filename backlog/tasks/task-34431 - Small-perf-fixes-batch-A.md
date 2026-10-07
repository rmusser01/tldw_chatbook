---
id: TASK-34431
title: Small perf fixes batch A
status: To Do
created_date: 2026-10-07 02:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 7 / F19a-c: fork path double-fetches the full conversation CitationTraceBuilder re-encodes all payloads per record and lore listing runs N+1 entry-count queries
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fork performs one full-conversation read,Citation trace performs N encodings for N records with cap intact,Lore counts one GROUP BY query for 10 books
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 19 (T19)
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
