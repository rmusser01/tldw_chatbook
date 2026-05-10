---
id: TASK-29
title: Add Chatbook Sync v2 stable push idempotency keys
status: Done
assignee: []
created_date: '2026-05-10 18:59'
updated_date: '2026-05-10 19:02'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make local-first Sync v2 pushes retry-safe by sending a deterministic idempotency key for each pending/outgoing envelope batch, based on non-secret batch identity rather than encrypted payload content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once sends a non-empty idempotency_key whenever it pushes envelopes
- [x] #2 The idempotency_key is stable across retry of the same dataset device cursor and pending envelope IDs
- [x] #3 The idempotency_key changes when the pushed envelope batch changes
- [x] #4 Focused LocalFirstSyncService tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing LocalFirstSyncService tests for stable retry idempotency keys and changed-batch key changes.
2. Implement deterministic idempotency-key derivation from dataset device cursor and client envelope IDs.
3. Pass the key through ServerSyncService.push_v2_envelopes without including encrypted payload bodies.
4. Run focused tests Bandit and diff checks.
5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added red LocalFirstSyncService tests proving push idempotency keys were missing: retry of the same pending batch produced None and changed batches also produced None. Implemented deterministic keys from dataset ID device ID cursor and client envelope IDs only, excluding encrypted payload bodies. Verification: focused LocalFirstSyncService tests passed 11/11; broader Tests/Sync_Interop passed 79/79; Bandit on tldw_chatbook/Sync_Interop/local_first_sync_service.py reported 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Local-first Sync v2 pushes now include stable idempotency keys derived from non-secret batch identity. The same pending batch and cursor gets the same key across retry, while changed envelope batches get a different key.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
