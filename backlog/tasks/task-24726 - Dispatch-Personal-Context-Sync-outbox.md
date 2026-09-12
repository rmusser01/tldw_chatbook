---
id: TASK-24726
title: Dispatch Personal Context Sync outbox
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 19:39'
updated_date: '2026-08-30 21:31'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-24725
references:
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/superpowers/plans/2026-08-28-personal-context-04-sync-multidevice.md
  - IMPLEMENTATION_PLAN_personal_context_sync_transport.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Atomically journal syncable canonical Personal Context mutations in Chatbook and dispatch exact encrypted whole-object envelopes idempotently without exposing device-only or plaintext profile content.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Targeted tests and verification recorded
- [x] #3 Documentation updated
- [x] #4 Static and security checks pass
- [x] #5 Independent review completed
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every syncable manifest, scope, record, and proposal mutation commits its exact canonical snapshot to the encrypted profile outbox in the same SQLite transaction; device-only records never enqueue. Schema-v1 purge barriers are encoded and validated by this transport, while the delete-everywhere mutation lifecycle remains owned by approved plan Task 6.
- [x] #2 The dispatcher validates and HMAC-tags canonical schema-v1 whole objects, copies them idempotently into the Sync-state outbox, records the destination envelope receipt, and crypto-shreds acknowledged profile-outbox bodies.
- [x] #3 Dispatcher restart/crash replay cannot duplicate canonical versions; poisoned entries are quarantined with content-free reason codes and no raw profile body appears in Sync logs, state, or diagnostics.
- [x] #4 Pulled Personal Context envelopes are validated and applied through `PersonalContextService` with profile/scope/purge/lineage fences; unsupported or conflicting content remains fail-closed.
- [x] #5 Targeted Personal Context/Sync_Interop regressions plus Ruff, compilation, Bandit, diff, and independent review gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for atomic encrypted outbox snapshots across all five domains, device-only exclusion, exact canonical bytes/HMAC, dispatcher crash/retry, poisoned quarantine, destination receipts, acknowledgement shredding, and inbound service-only application.
2. Extend the existing encrypted profile outbox with bounded pending/receipt/quarantine operations while preserving WAL snapshot concurrency and same-database mutation atomicity.
3. Implement the Personal Context envelope adapter and dispatcher, reusing the existing Sync-state outbox as the cross-database idempotent destination.
4. Integrate outbound building, inbound application, and local-first dispatch without allowing Sync code to mutate canonical repository tables directly.
5. Run targeted Personal Context and Sync_Interop regressions, Ruff, compilation, Bandit, diff hygiene, independent review, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-102 already governs the encrypted exact-wire outbox, whole-object Sync boundary, integrity, local mutation authority, and purge fencing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added deterministic encrypted profile-outbox lifecycle operations, atomic canonical snapshots for syncable objects, complete Sync CAS/head metadata, and device-only suppression.
- Added the Personal Context Sync adapter and dispatcher with HMAC validation, generic-outbox at-rest encryption, idempotent destination receipts, authenticated full-envelope crash recovery, recoverable destination-corruption quarantine, and source shredding only after verified staging.
- Routed inbound whole objects through `PersonalContextService` and made LocalFirst composition/capability checks fail closed when Personal Context transport or key custody is unavailable.
- Verification: 402 targeted Personal Context/Sync_Interop tests passed; Ruff, Python compilation, Bandit, `git diff --check`, and independent review passed with no remaining actionable blocker.
- Known skips: the full repository suite was not run per repository policy. Wrapped-key delivery/client bootstrap remains approved plan Task 3; delete-everywhere mutation lifecycle remains Task 6.
- ADR: existing `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md` applies; no new ADR was required.
<!-- SECTION:NOTES:END -->
