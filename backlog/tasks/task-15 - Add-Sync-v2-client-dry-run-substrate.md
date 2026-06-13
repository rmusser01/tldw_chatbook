---
id: TASK-15
title: Add Sync v2 client dry-run substrate
status: Done
assignee: []
created_date: '2026-05-10 07:09'
updated_date: '2026-05-10 07:17'
labels:
  - sync
  - client
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first Chatbook-side Sync v2 substrate so a configured tldw_server can be queried for capabilities, register a device, enroll a personal dataset, persist cursors, and run a no-content dry-run sync without changing local-only behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 schemas cover capabilities device registration dataset enrollment push pull cursors and per-envelope outcomes while preserving legacy media sync models
- [x] #2 Client exposes Sync v2 API methods for capabilities device registration dataset enrollment push pull and restore manifest requests
- [x] #3 Sync state and repository persist profile mode device dataset cursors and dry-run sync metadata without altering local-only defaults
- [x] #4 Server sync service can run a no-content dry-run flow that negotiates capabilities registers or reuses a device enrolls or reuses a dataset and records cursors
- [x] #5 Focused Sync_Interop and tldw_api tests cover the new dry-run path and legacy media sync tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing legacy sync schemas client methods and Sync_Interop repository patterns. 2. Add failing tests for Sync v2 schemas client calls persisted state and dry-run orchestration. 3. Implement minimal schemas client methods state repository fields and service flow. 4. Run focused tldw_api and Sync_Interop tests plus static/security checks where available. 5. Update Backlog notes and commit the client slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the first Chatbook Sync v2 dry-run substrate. Added Sync v2 protocol schemas beside the legacy media sync models, added client methods for capabilities device registration dataset enrollment push pull and restore manifest, extended sync profile state with mode device dataset and cursor metadata, persisted Sync v2 profile metadata in the sync state repository, and added a policy-gated no-content dry-run flow in ServerSyncService. Verification: initial red tests failed on missing Sync v2 symbols, focused Sync_Interop plus sync API tests passed with 42 tests, git diff --check passed, Chatbook venv lacked Bandit so the server repo Bandit install was used. Full Bandit showed pre-existing B608/B110 findings in legacy touched files; filtered scan excluding those existing IDs had 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the first Chatbook-side Sync v2 client substrate for no-content dry-run sync. Chatbook can now model Sync v2 capabilities and envelopes, call the new server endpoints, persist device dataset cursor and capability metadata, and run a dry-run negotiate/register/enroll/push-empty/pull-empty cycle without changing local-only behavior. Legacy media sync schemas and client methods are preserved.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit or equivalent security scan run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
