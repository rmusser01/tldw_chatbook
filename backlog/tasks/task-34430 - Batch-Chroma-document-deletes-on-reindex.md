---
id: TASK-34430
title: Batch Chroma document deletes on reindex
status: To Do
created_date: 2026-10-07 02:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 6 / F17: re-ingest deletes stale chunks per document with a where scan per doc making bulk reindex O(N times C)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 5000 changed docs produce at most 10 delete calls,Feature detect falls back to per-doc loop on old Chroma,Tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 18 (T18)
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
