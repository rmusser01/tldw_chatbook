---
id: TASK-18
title: Add Sync v2 restore manifest service
status: Done
assignee: []
created_date: '2026-05-10 15:19'
updated_date: '2026-05-10 15:26'
labels:
  - sync
  - client
  - restore
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the Chatbook-side Sync v2 restore service so clients can fetch metadata-only restore manifests, classify locked versus recoverable encrypted datasets, pull selected domains, decrypt pulled envelopes before local apply, and keep unresolved conflicts visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restore service fetches restore manifests without requiring plaintext decrypt
- [x] #2 Restore previews distinguish locked private datasets from datasets restorable with a local key or recovery bundle
- [x] #3 Restore previews expose conflict counts and attachment availability metadata
- [x] #4 Selected dataset and domain IDs become filtered Sync v2 pull requests
- [x] #5 Pulled encrypted envelopes are decrypted and applied locally while conflicts remain listable until resolved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing restore service tests for manifest preview classification selected pull filters decrypted local apply and conflict listing. 2. Add minimal Sync v2 conflict schemas and client/service wrappers needed by restore service. 3. Implement restore_service.py using ServerSyncService and SyncEnvelopeApplier. 4. Run restore/readiness tests plus the broader Sync_Interop sync API suite. 5. Run security and diff checks, update Backlog, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Chatbook Sync v2 restore manifest service plus real server/API wrappers for restore manifests, selected envelope pull, conflict list, and conflict resolve. Restore previews remain metadata-only, classify encrypted datasets by local-key/recovery availability, expose conflict and attachment metadata, and selected restores decrypt envelopes through SyncEnvelopeApplier before local apply.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Chatbook Sync v2 restore manifest flow: restore preview classification, filtered selected restore pull, decrypted local apply, conflict visibility, tldw_api conflict schemas/client methods, and policy-gated ServerSyncService wrappers. Verification: Tests/Sync_Interop/test_restore_service.py Tests/Sync_Interop/test_server_sync_service.py Tests/tldw_api/test_sync_client.py passed 17/17; Tests/Sync_Interop Tests/tldw_api/test_sync_client.py passed 62/62; filtered Bandit on touched production files reported 0 findings; git diff --check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit or equivalent security scan run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
