---
id: TASK-55
title: Require Sync v2 response dataset ids
status: Done
assignee: []
created_date: '2026-05-10 21:56'
updated_date: '2026-05-10 21:58'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Sync v2 transport validation so push and pull responses must include dataset_id instead of silently accepting missing response identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ServerSyncService.push_v2_envelopes raises when a push response omits dataset_id after dispatch.
- [x] #2 ServerSyncService.pull_v2_envelopes raises when a pull response omits dataset_id after dispatch.
- [x] #3 The shared response identity validator continues to reject mismatched dataset_id values.
- [x] #4 Focused Sync_Interop tests, Bandit on touched helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red ServerSyncService push and pull tests for responses missing dataset_id. 2. Update shared Sync v2 response identity validation to require response dataset_id before comparing it. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red-green verified the missing dataset_id push and pull cases. Focused tests failed before the validator change with DID NOT RAISE, then passed after requiring response dataset_id.

Verification: test_server_sync_service.py passed 24 tests; Tests/Sync_Interop passed 115 tests; Bandit on tldw_chatbook/Sync_Interop/validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Required Sync v2 push and pull transport responses to include dataset_id before accepting them as scoped to the requested dataset. Added ServerSyncService regression coverage for missing push and pull response dataset_id values.
<!-- SECTION:FINAL_SUMMARY:END -->
