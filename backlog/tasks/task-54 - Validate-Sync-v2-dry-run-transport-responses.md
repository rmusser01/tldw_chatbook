---
id: TASK-54
title: Validate Sync v2 dry-run transport responses
status: Done
assignee: []
created_date: '2026-05-10 21:47'
updated_date: '2026-05-10 21:49'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Sync v2 dry-run negotiation so its direct push and pull transport responses are validated before persisting profile state or cursors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dry-run raises when the dry-run push response dataset_id differs from the enrolled dataset_id.
- [x] #2 Dry-run raises when the dry-run pull response returns a non-empty page without next_cursor.
- [x] #3 Malformed dry-run transport responses do not persist Sync v2 profile state or remote pull cursors.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red dry-run tests for malformed push response dataset_id and non-empty pull response without next_cursor. 2. Reuse shared Sync v2 push, pull, and pagination validators inside run_v2_dry_run before profile/cursor persistence. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed dry-run accepted mismatched push response dataset_id and non-empty pull response without next_cursor before validation. Green/final verification: focused dry-run malformed response tests 2 passed; ServerSyncService test file 22 passed; full Tests/Sync_Interop 113 passed; Bandit on server_sync_service.py and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated Sync v2 dry-run push and pull transport responses before deriving cursors or persisting profile state. Malformed dry-run responses now fail without writing profile state or remote pull cursors.
<!-- SECTION:FINAL_SUMMARY:END -->
