---
id: TASK-21
title: Add Chatbook local-first Sync v2 sync_once service
status: Done
assignee: []
created_date: '2026-05-10 15:44'
updated_date: '2026-05-10 15:52'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync service requires an existing local_first profile with device dataset and dataset key
- [x] #2 Outgoing envelopes are pushed through the policy-gated ServerSyncService wrapper
- [x] #3 Pulled remote envelopes are decrypted and applied locally
- [x] #4 Remote pull cursor and profile dataset cursors are persisted after a successful pull
- [x] #5 Focused Sync_Interop tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for ServerSyncService.push_v2_envelopes and LocalFirstSyncService.sync_once. 2. Implement the push wrapper and local-first orchestration service. 3. Export the new service from Sync_Interop. 4. Run focused and broader sync tests, Bandit, and diff checks. 5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented LocalFirstSyncService.sync_once to require local_first profile state, profile device/dataset identifiers, and a dataset key before any server call. The service pushes caller-provided outgoing envelopes, pulls from the persisted remote cursor, applies decrypted envelopes locally, and persists next_cursor into both remote cursor and profile dataset cursors. Added ServerSyncService.push_v2_envelopes with sync.v2.push.server policy enforcement.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Chatbook local-first Sync v2 sync_once orchestration plus the policy-gated server push wrapper. Verification: focused local-first/server service tests 12 passed; broader Sync_Interop plus sync client suite 68 passed; Bandit touched production scope reported 0 findings; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
