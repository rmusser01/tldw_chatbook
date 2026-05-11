---
id: TASK-16
title: Encrypt Sync v2 payloads and recovery bundles
status: Done
assignee: []
created_date: '2026-05-10 15:07'
updated_date: '2026-05-10 15:13'
labels:
  - sync
  - client
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Chatbook-side authenticated encryption primitives for local-first Sync v2 payloads and recovery bundles so private content remains encrypted at rest and only opaque wrapped key material is sent to the server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dataset keys can be generated as opaque client-side bytes and are not serialized as plaintext in encrypted payloads or recovery bundles
- [x] #2 Sync payload encryption uses authenticated encryption with a random nonce and versioned metadata and decrypts only with the correct key
- [x] #3 Recovery bundle wrap and unwrap helpers store wrapped key material without exposing plaintext dataset keys
- [x] #4 Client and ServerSyncService expose a policy-gated call for POST /api/v1/sync/keys/recovery-bundle
- [x] #5 Focused Sync_Interop crypto and server sync service tests cover round trips nonce randomness wrong-key failure recovery wrapping plaintext absence and API routing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse existing pycryptodomex dependency for AES-256-GCM and keep all serialized metadata versioned. 2. Add failing tests for dataset key generation payload encryption/decryption random nonces wrong-key failures recovery wrapping and recovery-bundle API routing. 3. Implement crypto primitives and Sync v2 key recovery request/response schemas plus API client method. 4. Add ServerSyncService helper to store recovery bundles through the policy gate. 5. Run focused tests, security scan, diff check, update Backlog, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Chatbook Sync v2 client-side crypto and recovery bundle support. Added AES-256-GCM payload encryption/decryption using the existing pycryptodomex dependency, random nonces, versioned serialized metadata, 32-byte dataset key generation, scrypt-derived recovery wrapping and unwrapping, key readiness fields on SyncProfileState, Sync v2 key recovery request/response schemas, a TLDWAPIClient method for POST /api/v1/sync/keys/recovery-bundle, and a policy-gated ServerSyncService helper for storing opaque recovery bundle metadata. Verification: red tests first failed on the missing crypto module and recovery-bundle schemas; focused Sync_Interop plus sync API tests passed with 49 tests; git diff --check passed; Chatbook venv lacks Bandit so the server repo Bandit install was used. Full Bandit still reports existing B110 findings in legacy client.py try/except/pass paths; filtered scan excluding that existing ID had 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added authenticated client-side encryption primitives for Sync v2 local-first payloads and recovery bundles. Private payload JSON is encrypted with AES-256-GCM and random nonces, recovery bundles wrap dataset keys with a user-held secret via scrypt-derived AES-GCM material, and Chatbook can store only opaque wrapped key metadata through the Sync v2 recovery-bundle endpoint. Local-only behavior is unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit or equivalent security scan run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
