---
id: TASK-58
title: Address PR 315 Sync v2 attachment review comments
status: Done
assignee: []
created_date: '2026-05-15 19:18'
updated_date: '2026-05-15 19:21'
labels:
  - sync
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/315'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR 315 for the Sync v2 attachment upload client contract without expanding the feature scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SyncV2AttachmentUploadRequest forbids unexpected extra fields and has regression coverage.
- [x] #2 The new public attachment schemas use Google-style class docstrings.
- [x] #3 upload_sync_v2_attachment serializes request payloads with exclude_none=True and focused tests cover the payload.
- [x] #4 Focused Sync client tests, Sync_Interop tests, Bandit on touched production files, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused tests for extra-field rejection and exclude_none serialization. 2. Add strict request model config, Google-style schema docstrings, and exclude_none serialization. 3. Run focused tests, Sync_Interop, Bandit, diff checks, commit, push, and resolve the review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed all three unresolved PR #315 review threads: SyncV2AttachmentUploadRequest now forbids unknown extra fields, the new exported attachment schemas have Google-style Attributes docstrings, and upload_sync_v2_attachment serializes with exclude_none=True. Added regression coverage for extra-field rejection and serialization kwargs. Verification: focused attachment tests passed; Tests/tldw_api/test_sync_client.py plus Tests/Sync_Interop passed 131 tests; git diff --check passed; Bandit on touched production files passed using the tldw_server2 venv.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #315 review feedback by hardening the Sync v2 attachment request model, documenting the exported schemas, and aligning attachment upload serialization with the client API convention.
<!-- SECTION:FINAL_SUMMARY:END -->
