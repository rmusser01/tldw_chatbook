---
id: TASK-35378
title: Enforce server Canvas lifecycle conflicts and retired-owner barriers
status: To Do
created_date: 2026-10-18 05:55
dependencies:
- TASK-35377
labels:
- canvas
- sync
- server
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make delete, restore and purge converge safely across offline devices while retaining conflicting source histories for explicit review. Apply the approved ADR-239 lifecycle rules to the inert server domains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Same-epoch delete dominates update visibility in both arrival orders while preserving immutable candidates and durable delete-versus-update conflict records.
- [ ] #2 Restore requires exact current tombstone provenance and a new epoch; late old-epoch writes cannot change lifecycle, and parent-conversation deletion/restoration preserves independently deleted Canvas state.
- [ ] #3 Hard purge removes source-bearing pending/conflict copies and retains dataset-lifetime content-free retired-owner barriers; stale devices cannot recreate retired conversation IDs, and backup/erasure limits are documented.
- [ ] #4 Server tests cover concurrent siblings, rename/update, delete/update, restore/replay, equivalent deletes, quotas and purge retries without source leakage or HTML execution; only canonical bounded reason codes reach diagnostics.
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
