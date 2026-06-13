---
id: TASK-23
title: Add Chatbook Sync v2 key recovery setup service
status: Done
assignee: []
created_date: '2026-05-10 16:05'
updated_date: '2026-05-10 16:10'
labels:
  - sync
  - client
  - local-first
  - security
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Key recovery setup requires an existing local_first Sync v2 profile with device dataset and dataset key
- [x] #2 Dataset keys are wrapped locally with the user recovery secret before calling the server
- [x] #3 Server storage receives only opaque wrapped key material and KDF metadata through the policy-gated recovery bundle wrapper
- [x] #4 Profile metadata records recovery setup without persisting raw dataset keys recovery secrets or wrapped blobs
- [x] #5 Focused Sync_Interop tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a SyncKeyRecoveryService that wraps a dataset key locally, stores the recovery bundle through ServerSyncService, and records non-secret profile metadata. 2. Implement the service using existing crypto.wrap_dataset_key_for_recovery and SyncStateRepository profile updates. 3. Export the service from Sync_Interop. 4. Run focused and broader sync tests plus Bandit and diff checks. 5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented SyncKeyRecoveryService to require an existing local_first profile with device_id and dataset_id, wrap the local dataset key with wrap_dataset_key_for_recovery, store only wrapped key material through ServerSyncService.store_v2_recovery_bundle, and persist sanitized recovery metadata on the profile. Verification: key recovery test red run failed on missing module as expected; focused tests 16 passed; broader sync tests 72 passed; Bandit JSON had empty errors/results; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Chatbook Sync v2 key recovery setup service. The service validates local-first profile state, wraps dataset keys locally with the user recovery secret, stores opaque recovery bundle material through the existing policy-gated server wrapper, and records only non-secret recovery status metadata in the local profile. Added focused tests covering sanitized storage/metadata and required profile/key preconditions.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
