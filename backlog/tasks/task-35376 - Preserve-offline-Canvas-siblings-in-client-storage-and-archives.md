---
id: TASK-35376
title: Preserve offline Canvas siblings in client storage and archives
status: In Progress
created_date: 2026-10-18 05:54
dependencies:
- TASK-31003
labels:
- canvas
- sync
- client
priority: medium
assignee:
- '@codex'
updated_date: 2026-10-18 05:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the local artifact contract to preserve concurrent offline revisions without changing immutable source, ownership or branch projection. Implement the approved ADR-239 revision-order and archive compatibility foundation while Canvas synchronization remains disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A genuine current-schema database upgrades atomically and matches a fresh database; original Canvas values, owner/source protections and exactly-one-root enforcement are preserved, with rollback and retry covered.
- [ ] #2 Complete valid graphs permit equal-sequence siblings while retaining positive contiguous distinct sequence values, strictly lower same-Canvas parents, quotas and deterministic sequence/UUID ordering. Ordinary local settlement still allocates max(sequence)+1 and rejects stale authored sequence claims.
- [ ] #3 A new explicit Canvas-bearing Chatbook archive contract round-trips tied graphs and exact source/profile/origin metadata; existing valid 3.0 archives remain readable, non-Canvas exports retain their existing format, and legacy readers reject new archives before partial import.
- [ ] #4 Targeted repository, branch-resolution, migration, archive, backup-validation and packaging qualification plus documentation cover the changed contract; no transport, runtime capability or host-profile migration is activated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; existing accepted ADR path: backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md. Reason: direct implementation of its approved sibling-order/schema/archive contract; no duplicate ADR. 1. Follow Docs/superpowers/plans/2026-10-17-canvas-sync-client-foundation.md under isolated synthetic state. 2. Add RED/GREEN schema/repository/recovery ownership and sequence controls, then implement migration and typed graph admission without changing authored settlement safety. 3. Add RED/GREEN major-version archive and canonical-order/legacy-refusal controls, implement all creator/importer callers and exact compatibility. 4. Independently review each task and the full foundation, run only affected tests/static/package checks, update living documentation and record evidence before closing.
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
