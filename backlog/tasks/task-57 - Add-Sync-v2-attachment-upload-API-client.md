---
id: TASK-57
title: Add Sync v2 attachment upload API client
status: Done
assignee: []
created_date: '2026-05-15 03:14'
updated_date: '2026-05-15 03:23'
labels:
  - sync
  - chatbook
dependencies: []
references:
  - 'tldw_server2:Docs/API/sync-v2.md'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Chatbook client-side schema and API coverage for the server Sync v2 attachment feature-detection/upload endpoint so the client contract matches the merged server Sync v2 surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 attachment upload request and response schemas match the server contract for encrypted small attachment metadata.
- [x] #2 TLDWAPIClient exposes an upload_sync_v2_attachment method that calls POST /api/v1/sync/attachments with JSON payloads.
- [x] #3 Focused tldw_api sync client tests cover route shape and response parsing while existing Sync v2 client tests continue to pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing API-client test for Sync v2 attachment upload schema serialization and POST /api/v1/sync/attachments routing. 2. Add attachment upload request/response schemas aligned with the server contract. 3. Add TLDWAPIClient.upload_sync_v2_attachment and run focused Sync client tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green implementation: added a failing attachment upload API-client test that first failed because SyncV2AttachmentUploadRequest was not exported. Added SyncV2 attachment upload request/response schemas, exported them from tldw_api, and added TLDWAPIClient.upload_sync_v2_attachment for POST /api/v1/sync/attachments. Verification: Tests/tldw_api/test_sync_client.py plus Tests/Sync_Interop passed 129 tests; git diff --check passed; Bandit on touched production files passed using the tldw_server2 venv because the Chatbook venv does not include Bandit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned the Chatbook Sync v2 API client with the merged server attachment endpoint by adding encrypted attachment upload schemas, exports, a client method, and focused route/serialization coverage.
<!-- SECTION:FINAL_SUMMARY:END -->
