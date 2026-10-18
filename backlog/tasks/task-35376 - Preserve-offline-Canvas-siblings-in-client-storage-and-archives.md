---
id: TASK-35376
title: Preserve offline Canvas siblings in client storage and archives
status: Done
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
updated_date: 2026-10-18 06:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the local artifact contract to preserve concurrent offline revisions without changing immutable source, ownership or branch projection. Implement the approved ADR-239 revision-order and archive compatibility foundation while Canvas synchronization remains disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A genuine current-schema database upgrades atomically and matches a fresh database; original Canvas values, owner/source protections and exactly-one-root enforcement are preserved, with rollback and retry covered.
- [x] #2 Complete valid graphs permit equal-sequence siblings while retaining positive contiguous distinct sequence values, strictly lower same-Canvas parents, quotas and deterministic sequence/UUID ordering. Ordinary local settlement still allocates max(sequence)+1 and rejects stale authored sequence claims.
- [x] #3 A new explicit Canvas-bearing Chatbook archive contract round-trips tied graphs and exact source/profile/origin metadata; existing valid 3.0 archives remain readable, non-Canvas exports retain their existing format, and legacy readers reject new archives before partial import.
- [x] #4 Targeted repository, branch-resolution, migration, archive, backup-validation and packaging qualification plus documentation cover the changed contract; no transport, runtime capability or host-profile migration is activated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; existing accepted ADR path: backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md. Reason: direct implementation of its approved sibling-order/schema/archive contract; no duplicate ADR. 1. Follow Docs/superpowers/plans/2026-10-17-canvas-sync-client-foundation.md under isolated synthetic state. 2. Add RED/GREEN schema/repository/recovery ownership and sequence controls, then implement migration and typed graph admission without changing authored settlement safety. 3. Add RED/GREEN major-version archive and canonical-order/legacy-refusal controls, implement all creator/importer callers and exact compatibility. 4. Independently review each task and the full foundation, run only affected tests/static/package checks, update living documentation and record evidence before closing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the approved local sibling-graph foundation under accepted ADR-239 (backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md), following Docs/superpowers/plans/2026-10-17-canvas-sync-client-foundation.md. Schema 82 transactionally replaces sequence uniqueness with canonical sequence/UUID ordering and a native single-root constraint, preserving immutable values and serial authored max+1 allocation. Repository graph validation is bounded by revision count. Chatbook 4.0 / Canvas extension 2.0 round-trips tied graphs; valid legacy 3.0/1.0 and no-Canvas exports remain supported. Exact frozen core/hybrid recovery catalogs retain all four v81 predecessors and qualify installed v82 migration. Changed DB/migration/recovery, Canvas repository/archive, Chatbook model/creator/importer, focused tests, index census classification and living architecture/archive documentation; packaging discovery needed no new registry. Review found and fixed legacy V1/V2 Canvas-member dropping with refusal before extraction/database access, plus date-labeled an earlier pre-approval closeout. Independent task, scoped correction, whole-branch and final prose re-reviews are complete, with no remaining findings. Evidence: 103 final model/archive tests, seven fresh real-package/installed checks, 30 independent final-parser confirmations, ten migration controls, eleven recovery controls and derived preflight checks passed. Broader affected run was 264 passed / one migration-source census failure reproduced identically on exact base for seven unrelated SQL files. Scoped Ruff has 762 identical inherited per-file diagnostics, zero introduced; twelve baseline-clean files format successfully and four retain pre-existing formatting debt. No global-green claim. Root checked package/source bytes and all 5504 Backlog records. Qualification and limitations: Docs/Canvas/SIBLING_GRAPH_FOUNDATION_QUALIFICATION_2026-10-17.md; raw local receipts /private/tmp/chatbook-canvas-sync-foundation.KPZ9cL. Scope clarifications were exact installed 81/82 hybrid recovery, count-bounded contiguous validation and correctness-only index census classification; no old catalog or guard relaxation. Sync remains disabled: no transport, runtime authority, server write, normal-profile migration, dependency install, full-suite, hosted/platform/server qualification, push or merge. TASK-35377–35383 remain To Do; schema/archive allocations must be rechecked before integration.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and independently reviewed the local Canvas sibling-revision storage, archive and recovery foundation. The approved sync contract is recorded separately from the still-unimplemented server/client transport and real-server qualification tasks. Local commits only; targeted evidence and inherited failures are documented.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
