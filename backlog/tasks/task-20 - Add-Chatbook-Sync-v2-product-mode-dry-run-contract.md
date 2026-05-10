---
id: TASK-20
title: Add Chatbook Sync v2 product mode dry-run contract
status: Done
assignee: []
created_date: '2026-05-10 15:41'
updated_date: '2026-05-10 15:42'
labels:
  - sync
  - client
  - modes
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-only mode dry-run returns a local-only report without calling server sync or creating Sync v2 profile state
- [x] #2 Server-front-end mode dry-run returns a direct-server report without creating a local Sync v2 device dataset or profile state
- [x] #3 Local-first mode dry-run delegates to the existing Sync v2 server dry-run path
- [x] #4 Focused Sync scope tests pass
- [x] #5 Bandit and diff checks are clean for touched files
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing SyncScopeService tests for local_only server_frontend and local_first profile-mode dry-run behavior. 2. Implement a profile-mode dry-run method that avoids sync side effects for local_only and server_frontend and delegates local_first to ServerSyncService.run_v2_dry_run. 3. Run focused scope tests and broader Sync_Interop tests. 4. Run Bandit/diff checks, update Backlog, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added SyncScopeService.prepare_sync_v2_profile_mode as a product-mode dry-run contract. local_only and server_frontend return explicit reports without calling server sync or writing Sync v2 profile/device/dataset state. local_first requires display_name and delegates to ServerSyncService.run_v2_dry_run, preserving the existing local-first registration/enrollment path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Chatbook Sync v2 product-mode dry-run handling. Local-only and server-front-end modes are now explicitly selectable at the sync scope boundary without hidden sync side effects; local-first mode delegates to the existing Sync v2 dry-run path. Verification: red tests failed on missing prepare_sync_v2_profile_mode; Tests/Sync_Interop/test_sync_scope_service.py passed 10/10 after implementation; Tests/Sync_Interop plus Tests/tldw_api/test_sync_client.py passed 65/65; Bandit on sync_scope_service.py reported 0 findings; git diff --check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
