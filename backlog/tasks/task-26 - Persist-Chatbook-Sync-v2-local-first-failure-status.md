---
id: TASK-26
title: Persist Chatbook Sync v2 local-first failure status
status: Done
assignee: []
created_date: '2026-05-10 16:51'
updated_date: '2026-05-10 16:54'
labels:
  - sync
  - client
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add failure-status durability to Chatbook local-first Sync v2 sync_once so push, pull, and local apply failures leave the Sync v2 profile with an actionable last_error while preserving cursor safety and clearing stale errors after a later successful sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Push pull and local apply failures persist a stage-specific Sync v2 profile last_error
- [x] #2 Failed sync_once does not advance the remote pull cursor or dataset cursor
- [x] #3 A successful later sync clears any prior last_error while preserving profile device dataset capabilities and dry-run metadata
- [x] #4 Focused LocalFirstSyncService tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing LocalFirstSyncService tests for pull/apply failure last_error cursor safety and success clearing stale errors.
2. Implement minimal failure recording around sync_once stages without swallowing the original exception.
3. Preserve existing profile metadata when recording or clearing last_error.
4. Run focused tests Bandit and diff checks.
5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added stage-specific failure recording for local-first Sync v2 push pull and apply stages. Verified red tests first failed on missing last_error persistence. Green verification: Tests/Sync_Interop/test_local_first_sync_service.py passed 8/8; Tests/Sync_Interop passed 76/76; Bandit on tldw_chatbook/Sync_Interop/local_first_sync_service.py reported 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
LocalFirstSyncService now persists actionable Sync v2 profile last_error values for push_failed pull_failed and apply_failed without swallowing the original exception or advancing cursors. Successful sync_once runs clear stale last_error values while preserving device dataset cursor capability and dry-run metadata.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
