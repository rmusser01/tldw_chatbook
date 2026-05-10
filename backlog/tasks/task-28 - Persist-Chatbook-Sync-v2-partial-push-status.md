---
id: TASK-28
title: Persist Chatbook Sync v2 partial push status
status: Done
assignee: []
created_date: '2026-05-10 17:00'
updated_date: '2026-05-10 17:01'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep local-first sync profile status honest when the server accepts only part of a push. Rejected or conflicted outgoing envelopes should remain visible in Sync v2 profile last_error even if pull succeeds and cursors advance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Push responses with rejected envelopes persist a push_partial_failure profile last_error
- [x] #2 Push responses with conflict envelopes persist a push_partial_failure profile last_error
- [x] #3 Successful pulls may still advance cursors while retained outbox entries and profile status preserve outgoing failure visibility
- [x] #4 Focused LocalFirstSyncService tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing LocalFirstSyncService assertion proving partial push failures survive a successful pull.
2. Implement partial-push last_error selection after apply and before final profile persistence.
3. Preserve existing cursor advancement and outbox retention semantics.
4. Run focused tests Bandit and diff checks.
5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a red assertion proving Sync v2 profile last_error was cleared after a push response with rejected and conflicted outbox envelopes. Implemented push_partial_failure status selection while preserving successful pull cursor advancement and retained outbox entries. Verification: red focused test failed with last_error None; green Tests/Sync_Interop/test_local_first_sync_service.py passed 9/9; broader Tests/Sync_Interop passed 77/77; Bandit on tldw_chatbook/Sync_Interop/local_first_sync_service.py reported 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
LocalFirstSyncService now keeps partial outgoing push failures visible in Sync v2 profile status. Rejected and conflicted push results produce push_partial_failure last_error while successful pulls can still advance cursors and retained outbox rows keep per-envelope errors.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
