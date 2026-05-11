---
id: TASK-30
title: Record Chatbook Sync v2 outbox transport failures
status: Done
assignee: []
created_date: '2026-05-10 19:06'
updated_date: '2026-05-10 19:07'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When local-first Sync v2 push transport fails before the server returns per-envelope results, retain pending outbox rows but record attempt metadata and retryable errors for the batch envelopes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Push transport exceptions increment attempt_count for pending outbox entries in the attempted batch
- [x] #2 Push transport exceptions store retryable push_failed last_error metadata on attempted pending outbox entries
- [x] #3 Outbox entries remain pending and retryable after a transport failure
- [x] #4 Focused LocalFirstSyncService tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing LocalFirstSyncService test for pending outbox rows after a push transport exception.
2. Reuse repository push-result marking to record retryable push_failed errors for attempted outbox rows.
3. Preserve existing profile last_error and retry behavior.
4. Run focused tests Bandit and diff checks.
5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a red LocalFirstSyncService test proving push transport exceptions left pending outbox rows with attempt_count 0 and no last_error. Implemented retryable push_failed marking for attempted pending outbox rows via the existing repository push-result updater while preserving profile-level push_failed behavior. Verification: focused LocalFirstSyncService tests passed 12/12; broader Tests/Sync_Interop passed 80/80; Bandit on tldw_chatbook/Sync_Interop/local_first_sync_service.py reported 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Local-first Sync v2 now records per-envelope outbox attempt metadata when the push transport fails before server per-envelope results are available. Attempted pending rows remain pending but gain attempt_count and retryable push_failed last_error metadata for retry UI and diagnostics.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
