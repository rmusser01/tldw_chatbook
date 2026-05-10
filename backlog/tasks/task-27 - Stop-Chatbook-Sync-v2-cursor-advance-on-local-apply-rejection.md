---
id: TASK-27
title: Stop Chatbook Sync v2 cursor advance on local apply rejection
status: Done
assignee: []
created_date: '2026-05-10 16:57'
updated_date: '2026-05-10 16:59'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent local-first sync_once from advancing remote cursors when a pulled envelope is rejected by a local domain adapter without raising, so malformed or unsupported local apply results remain retry-visible and profile status records the failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local adapter rejected results are treated as failed apply attempts rather than successful consumption
- [x] #2 Rejected apply results persist a stage-specific Sync v2 profile last_error
- [x] #3 Remote pull cursor and profile dataset cursor remain unchanged when pulled envelopes are rejected
- [x] #4 Focused LocalFirstSyncService tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing LocalFirstSyncService test with a pulled workspace envelope that the local adapter rejects.
2. Implement rejection detection after apply results before cursor persistence.
3. Persist apply_rejected last_error and re-raise a clear error while preserving cursor state.
4. Run focused tests Bandit and diff checks.
5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a red LocalFirstSyncService regression for a pulled workspace envelope rejected by the local adapter. Implemented apply rejection detection before cursor persistence. Verification: red run failed because sync_once did not raise on adapter rejection; green run passed Tests/Sync_Interop/test_local_first_sync_service.py 9/9; broader Tests/Sync_Interop passed 77/77; Bandit on tldw_chatbook/Sync_Interop/local_first_sync_service.py reported 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
LocalFirstSyncService now treats local adapter rejected results as failed apply attempts. It records apply_rejected in the Sync v2 profile last_error, raises a clear error, and leaves remote pull/profile dataset cursors unchanged so rejected pulled envelopes remain retry-visible.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
