---
id: TASK-34420
title: Persistent embedding content-hash cache ADR-213
status: In Progress
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 08:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 / F6: no content-hash embedding cache exists - the wrapper cache check is dead code against a model-keyed dict so any re-index re-embeds everything and the skip check is N+1
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-213 written before code,embedding_cache table with migration and version bump,Unchanged-content re-ingest performs zero provider embed calls after restart,Batched skip check one query per batch,Hit-rate metric reports true values
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 8 (T8)
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
