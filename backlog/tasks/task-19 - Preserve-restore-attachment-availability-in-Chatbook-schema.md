---
id: TASK-19
title: Preserve restore attachment availability in Chatbook schema
status: Done
assignee: []
created_date: '2026-05-10 15:35'
updated_date: '2026-05-10 15:36'
labels:
  - sync
  - client
  - restore
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 restore manifest dataset schema includes attachment availability counts from the server
- [x] #2 Sync API client tests assert attachment availability survives manifest parsing
- [x] #3 Focused sync API tests pass
- [x] #4 Bandit and diff checks are clean for touched files
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing sync API client test asserting attachment_availability is preserved. 2. Add the field to the Chatbook Sync v2 restore manifest dataset schema. 3. Run focused tests and security/diff checks. 4. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added failing sync API client coverage for restore manifest attachment_availability preservation, then added attachment_availability to the Chatbook SyncV2RestoreManifestDataset schema so restore previews keep the server's attachment availability counts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved Sync v2 restore attachment availability in Chatbook. The client schema now retains server restore-manifest attachment_availability counts, and the sync API client regression test asserts the field survives parsing. Verification: red test failed on missing field; Tests/tldw_api/test_sync_client.py plus Tests/Sync_Interop/test_restore_service.py passed 8/8 after the fix; Tests/Sync_Interop plus Tests/tldw_api/test_sync_client.py passed 62/62; Bandit on the touched schema reported 0 findings; git diff --check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
