---
id: TASK-41
title: Reject mismatched Sync v2 outgoing identity before push
status: Done
assignee: []
created_date: '2026-05-10 20:11'
updated_date: '2026-05-10 20:14'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook local-first Sync v2 so ad hoc outgoing envelopes passed to sync_once cannot be pushed when their dataset_id or device_id differs from the active local-first profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once rejects outgoing_envelopes whose dataset_id differs from the profile dataset_id before contacting the server.
- [x] #2 sync_once rejects outgoing_envelopes whose device_id differs from the profile device_id before contacting the server.
- [x] #3 Failures are recorded on the Sync v2 profile without advancing the remote pull cursor.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for outgoing dataset_id and device_id mismatches before push. 2. Extend local outgoing validation to check active profile dataset/device identity. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended local-first outgoing pre-push validation so ad hoc envelopes must match the active profile dataset_id and device_id before any server contact. Added regressions for mismatched outgoing dataset and device, corrected the happy-path fixture to build outgoing content from the local profile device, and verified focused/full Sync_Interop plus Bandit and diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
