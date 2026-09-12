---
id: TASK-24400
title: Add encrypted local Personal Context repository
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 17:03'
updated_date: '2026-08-29 17:58'
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
- [x] #1 Encrypted profile objects, proposals, manifests, policies, bindings, and outbox bodies persist only as versioned authenticated envelopes with separate encryption and integrity keys
- [x] #2 Repository transactions enforce one profile per install, immutable object versions, compare-and-set heads, rollback on failure, quarantine, proposal resolution, and content destruction
- [x] #3 Key protection prefers the OS keyring, supports the approved passphrase fallback, and enters a typed locked state without plaintext or replacement-key fallback
- [x] #4 Targeted real-SQLite tests prove reopen behavior, key destruction, AAD/integrity failure, unique nonces, rollback, and absence of plaintext canaries from the database, WAL, and sidecar files
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the dedicated encrypted Personal Context repository behind Chatbook's private-SQLite boundary. Canonical manifests, records, proposals, runtime policies, workspace bindings, and outbox bodies use independent AES-256-GCM envelopes with random DEKs, authenticated object identity, and a separate HMAC-SHA-256 integrity key. Secure OS keyring custody is preferred; the explicit passphrase fallback uses a profile-bound scrypt/AES-GCM wrapper and never persists plaintext keys.

Repository initialization, immutable version inserts, parent/head compare-and-set, proposal resolution, local metadata updates, outbox publication, quarantine, and destruction fences use explicit SQLite write transactions. Initialization serializes key creation with schema ownership, and destruction leaves a durable generation fence so stale open repository instances cannot resurrect content; failed key deletion remains retryable. The store is registered as private owner `personal_context.repository` and remains outside centralized backup.

Verification completed with 318 targeted tests passing and one Windows-only test skipped across Personal Context, root packaging, private-SQLite inventory, and the private-SQLite seam. Ruff check, Ruff format check, and `git diff --check` passed. Independent specification and code-quality reviews approved the final implementation. The run emitted the existing `requests` dependency-version warning and unrelated stale pytest temporary-directory cleanup warnings after successful completion.

ADR required: no new ADR. The implementation follows [ADR-102](../decisions/102-personal-context-profile-authority-sync-and-encryption.md); SQLite/keyring operations intentionally do not claim cross-system atomicity.

Core files: `tldw_chatbook/Personal_Context/`, `Tests/Personal_Context/`, `pyproject.toml`, `tldw_chatbook/DB/private_sqlite.py`, `Tests/DB/test_private_sqlite_inventory.py`, `Tests/Packaging/test_profile_core_packaging.py`, and `backlog/docs/sqlite-private-owner-inventory.md`.
<!-- SECTION:NOTES:END -->
