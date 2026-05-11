---
id: TASK-24
title: Add Chatbook Sync v2 recovery unlock restore flow
status: Done
assignee: []
created_date: '2026-05-10 16:18'
updated_date: '2026-05-10 16:23'
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
- [x] #1 Chatbook API client can retrieve Sync v2 key recovery bundle records for a dataset with optional device and key-purpose filters
- [x] #2 Restore service can unwrap a fetched recovery bundle with a user recovery secret and use the recovered dataset key for selected restore
- [x] #3 Recovered dataset keys are kept in memory only and are not persisted to profile metadata or result payloads
- [x] #4 Recovery failures return clear errors without pulling encrypted envelopes or applying local changes
- [x] #5 Focused Sync_Interop and tldw_api tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing API client tests for GET /api/v1/sync/keys/recovery-bundle response schemas and routing. 2. Add failing restore-service tests for recovery-secret unwrap and no-secret/no-key persistence. 3. Implement client schemas/method exports and restore-service recovery flow using unwrap_recovery_bundle. 4. Run focused and broader sync tests plus Bandit and diff checks. 5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Sync v2 recovery bundle retrieval in the Chatbook API client, added a policy-gated ServerSyncService wrapper, and extended SyncRestoreService so restore_selection can fetch a recovery bundle, unwrap the dataset key with a user recovery secret, and apply selected encrypted envelopes without persisting recovered keys. Recovery failures stop before pull/apply. Also narrowed two pre-existing client JSON parse handlers from broad Exception to ValueError so Bandit passes on the touched client file. Verification: red run failed on missing recovery-list API types as expected; focused tests passed with 26 tests; broader Sync_Interop plus sync client tests passed with 76 tests; Bandit JSON had empty errors/results; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Chatbook recovery-unlock restore support. The client can now retrieve Sync v2 recovery bundle records, the server transport wrapper gates that retrieval through runtime policy, and restore can use a user recovery secret to unwrap the dataset key in memory before applying selected encrypted envelopes. Wrong recovery secrets fail before pulling or applying data.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
