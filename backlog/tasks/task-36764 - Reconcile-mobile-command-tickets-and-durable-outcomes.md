---
id: TASK-36764
title: Reconcile mobile command tickets and durable outcomes
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36760
- TASK-36762
- TASK-36763
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 22:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent repeated or uncertain phone submissions from causing duplicate Console actions while keeping native cancellation available and outcomes honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authenticated server-issued tickets bind action, target and owner epochs to an immutable five-minute monotonic submission expiry; tampered, expired and prior-epoch tickets cannot issue effects.
- [x] #2 An exact command is admitted at most once, changed payload reuse is refused, and preparing, accepted, queued, applied, refused and unknown outcomes reflect native custody rather than transport acknowledgement.
- [x] #3 Valid-ticket evidence and pending or unknown outcomes cannot be evicted into replay; capacity refuses new non-idempotent effects while exact-target native Stop remains available under receipt pressure.
- [x] #4 Receipts needed after owner loss are durable and body-free, completed evidence prunes only after expiry, and executed schema migrations plus exact recovery catalogs preserve them.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of approved expiring-ticket, durable-receipt and native uncertainty contracts.

1. Add failing real-owned SQLite repository/ticket duplication, expiry/tampering/capacity and native handoff interruption regressions with recording dispatcher plus native custody fixtures.
2. Execute AgentRuns v24→v25 receipt migration through installed owner registry, update exact recovery schema/read catalogs and backup coverage; keep all operations parameterized and execution-owner guarded.
3. Implement authenticated immutable five-minute tickets, keyed fingerprints and atomic insertion ownership before effects, bounded retention/pruning and idempotent exact Stop under pressure.
4. Couple body-free receipt outcomes to native acceptance/queue/round evidence and honest unknown handoffs; expose paginated unresolved receipts and current child handlers without replay.
5. Run specified targeted repository/runtime/dispatch/protocol/schema checks, scoped syntax/default-Ruff/format checks, update docs, commit explicit paths and obtain independent spec+quality review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented authenticated immutable300-second HMAC tickets, body-free durable AgentRuns receipts with atomic inserted ownership/CAS, native custody/queue/final-decision outcomes and honest unknown handoffs. Executed v24→v25 SQL migration with rollback/exact recovery catalogs/read-authorizer/backup coverage. Fixed4096row bound protects valid/pending/unknown evidence; known priorowner handoffs remain queryable and pressure reclaims one oldest eligible row only after IDchecks. Exact native Stop retains authenticated idempotent no-row capacity path with requested/physical-unconfirmed/durable-unavailable reason. Independent receipt_cursor/next_receipt_cursor snapshot continuation stays inside16KiB receipt/full64KiB budgets. Native decision winner attributed under original lock, final applied/refused only after existing deadline/cancel/outcome; desktop callers preserved. Native accepted/queued handoffs may prune only after ticket invalid, without claiming consumed/exited. Existing ADR-244 governs, root/controller scope/retention/Stop/decision rulings recorded in execution ledger. Independent review found expiry-boundary redispatch and priorowner capacity defects; fix847e91f7af after implementationbd98df2be0 addressed both and scoped review is clean. Evidence311 targeted passes; fix six expiryRED +corrected real4096row pressure2RED→46coveringpasses. Zero introduced perfile defaultRuff, native formatter snapshot/verify including committedHEAD, newformat/AST/diff pass. Files: repository/service/migration/native acceptance+round seam/projection/child handler/exact recovery owners/tests/database+web docs. Task9 owns publicAPI/authlifecycle/parentcursor/createopen/cutover; no fullsuite/livephone/provider/userdata migration/push/merge claimed. Report task-6-report.md in plan-specific .superpowers workspace.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
