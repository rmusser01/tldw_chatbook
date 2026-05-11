---
id: TASK-22
title: Add Chatbook Sync v2 durable local outbox drain
status: Done
assignee: []
created_date: '2026-05-10 15:56'
updated_date: '2026-05-10 16:02'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SyncStateRepository persists pending Sync v2 envelopes with stable status attempt and error metadata
- [x] #2 LocalFirstSyncService can drain pending outbox envelopes before ad hoc outgoing envelopes
- [x] #3 Accepted drained outbox envelopes are marked dispatched after a successful push
- [x] #4 Rejected or conflicted outbox envelopes remain pending with recorded errors and are reported in sync_once results
- [x] #5 Focused Sync_Interop tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing repository and local-first service tests for persisted outbox enqueue drain dispatched and failed states. 2. Add SyncStateRepository outbox schema and helpers for enqueue list and mark result. 3. Teach LocalFirstSyncService.sync_once to drain pending outbox envelopes before caller-provided envelopes and update outbox statuses from push results. 4. Run focused and broader sync tests plus Bandit and diff checks. 5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a durable Sync v2 local outbox in SyncStateRepository with enqueue, pending-list, filtered-list, and push-result marking helpers. LocalFirstSyncService now drains pending outbox envelopes before ad hoc outgoing envelopes, marks accepted outbox rows dispatched, retains rejected/conflicted rows with last_error details, and returns outbox/rejection/conflict counts in sync_once results. Also replaced string-built SQL in the touched repository file with parameterized static statements so Bandit passes on the touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added durable Chatbook Sync v2 local outbox drain for local-first sync. Verification: focused repository/local-first tests 13 passed; broader Sync_Interop plus sync client suite 70 passed; Bandit touched production scope reported 0 findings; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
