---
id: TASK-33647
title: Keep legacy trace batches within the storage-admission budget
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 15:14'
updated_date: '2026-10-02 15:26'
labels:
  - ci
  - performance
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The read-only completion precheck introduced for parked trace maintenance adds a second admission when a batch still needs normalization. Keep the existing storage-unit budget without weakening the idle read-only path, mutation rechecks or finite worker ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh file-backed normalization batch and completed idle batch each pay at most two storage admissions through the existing finite worker helper.
- [x] #2 Idle completion uses no immediate transaction, and normalization still rechecks maintenance and migration state under the immediate transaction.
- [x] #3 The existing Console storage-unit ratchet and relevant maintenance and worker-ownership checks pass without changing ceilings, warmup or measured work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact CI admission breach and pin the first incomplete fresh-worker callback with a real file-backed regression. 2. Reuse the existing core repository operation across the read-only probe and write fallback while leaving run_owned_db_call and physical compaction unchanged. 3. Run the unchanged storage-unit ratchet, read-only/refusal/rollback/parking/worker-ownership checks, static and diagnostic checks, then obtain independent review. ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: reuse an existing admitted repository scope without changing SQL transaction, schema, authority or runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CI on published 68646eeeaa failed trace maintenance at 17 admissions across eight callbacks (2.125 > ceiling 2). The real fresh file-backed regression fails before the repair at three admissions versus two. The first local whole census was red but its default child log was lost during shared pytest cleanup; a separate task-private census passes before the repair, so no deterministic whole-census red claim is made. After the repair, the cold/idle callback regression and 12 targeted maintenance cases pass in 12.08s; the unchanged full storage-unit ratchet passes in 34.33s. Keep all ceilings and measured work unchanged. Independent review, startup/schema and diagnostic checks pending.

Reuse the existing core repository operation for the legacy completion probe and normalization fallback. The read-only idle transaction and immediate write rechecks stay separate; provider-active refusal occurs before admission, and run_owned_db_call and physical GC are unchanged. The real file-backed regression processes a cold legacy row, enforces a nonzero upper admission budget of two, then checks completed idle work and zero registered worker handles. Before the repair it fails at three admissions; the final predicate passes in 3.11s. The unchanged Console storage-unit guard passes in 34.33s (trace 1.875 admissions per tick; all original ceilings, phases and canaries preserved). Twelve affected maintenance cases pass in 12.08s, and eight independent refusal/rollback/read-only/parked-wake/borrowed/in-flight cleanup cases pass across 8.67s and 2.29s. Independent source review approves. Three startup/exact-schema guards pass in 22.60s, imports 679/686 and UI-ready 1032/1033. Final changed-code static checks cover 81 Python files, ten new-file Ruff/format checks pass, owned formatting and whitespace pass, and the audited diagnostic inventory remains unchanged. ADR required: no new ADR; existing backlog/decisions/126-complete-local-backup-and-recovery.md. Modified production scope, one existing test module, qualification records and testing lesson. The whole census can pass before the repair; the controlled cold-worker test is the red proof. CI failed at 17/8 on published 68646eeeaa; fresh published-head remote checks remain required. No full suite, live provider, Windows, aggregate-resource or merge result.
<!-- SECTION:NOTES:END -->
