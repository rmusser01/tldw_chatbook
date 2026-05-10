---
id: TASK-37
title: Reject mismatched Sync v2 push envelopes client-side
status: Done
assignee: []
created_date: '2026-05-10 19:41'
updated_date: '2026-05-10 19:43'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the tldw_chatbook Sync v2 transport so a client cannot dispatch envelopes through push_v2_envelopes when the envelope dataset or device identity does not match the request identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 push_v2_envelopes raises before transport dispatch when any envelope dataset_id differs from the request dataset_id.
- [x] #2 push_v2_envelopes raises before transport dispatch when any envelope device_id differs from the request device_id.
- [x] #3 Existing valid Sync v2 server transport behavior remains covered by tests.
- [x] #4 Focused Sync_Interop tests, Bandit on the touched service file, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red transport tests for mismatched dataset_id and device_id envelopes. 2. Add client-side validation before building the push request. 3. Run focused Sync_Interop tests, full Sync_Interop suite, Bandit, and diff checks before committing.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added service-level Sync v2 push envelope identity validation for dataset_id and device_id before transport dispatch. Covered mismatched dataset and device cases with focused async tests, and verified the full Sync_Interop suite plus Bandit and git diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
