---
id: TASK-24400
title: Add encrypted local Personal Context repository
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29 17:03'
updated_date: '2026-08-29 17:04'
labels: []
dependencies: []
references:
  - Docs/superpowers/plans/2026-08-28-personal-context-01-core-chatbook-local.md
documentation:
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create Chatbook's dedicated local encrypted repository and key-protection boundary so canonical Personal Context records and proposals persist without plaintext leakage, support optimistic versioning and quarantine, and fail closed when key material is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Encrypted profile objects, proposals, manifests, policies, bindings, and outbox bodies persist only as versioned authenticated envelopes with separate encryption and integrity keys
- [ ] #2 Repository transactions enforce one profile per install, immutable object versions, compare-and-set heads, rollback on failure, quarantine, proposal resolution, and content destruction
- [ ] #3 Key protection prefers the OS keyring, supports the approved passphrase fallback, and enters a typed locked state without plaintext or replacement-key fallback
- [ ] #4 Targeted real-SQLite tests prove reopen behavior, key destruction, AAD/integrity failure, unique nonces, rollback, and absence of plaintext canaries from the database, WAL, and sidecar files
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the private-SQLite owner seam and existing keyring/passphrase encryption patterns, then pin the dedicated Personal Context owner/path and locked-state boundary.
2. Write focused RED tests for envelope crypto, key protection, schema/transactions, reopen/CAS/quarantine/destruction, and plaintext canaries across database, WAL, and sidecars.
3. Implement the minimal versioned AES-256-GCM envelope, separate encryption/integrity key material, keyring-first/passphrase key protector, and typed locked failures.
4. Implement schema v1 and repository transactions for manifests, immutable records/proposals, heads, encrypted local metadata/outbox, quarantine, and content destruction without cross-database atomicity claims.
5. Run targeted real-SQLite tests, private-owner inventory checks, Ruff, plaintext byte inspection, reopen verification, independent spec/code-quality review, and update Backlog notes/criteria.

ADR required: no — ADR-102 already governs storage, encryption, key custody, transaction, and destruction boundaries.

ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
<!-- SECTION:PLAN:END -->
