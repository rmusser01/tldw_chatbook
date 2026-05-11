---
id: TASK-51
title: Reject duplicate Sync v2 server transport envelope ids
status: Done
assignee: []
created_date: '2026-05-10 21:27'
updated_date: '2026-05-10 21:29'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the policy-gated ServerSyncService push_v2_envelopes transport so direct callers cannot dispatch a batch with repeated client_envelope_id values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ServerSyncService.push_v2_envelopes raises before client dispatch when a direct push batch repeats client_envelope_id.
- [x] #2 Existing server transport dataset and device identity rejections remain covered.
- [x] #3 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red ServerSyncService test for duplicate client_envelope_id values in a direct push_v2_envelopes batch. 2. Reuse shared outgoing envelope validation in ServerSyncService before client dispatch. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed duplicate direct push envelopes did not raise before validation and reached the fake client. Green/final verification: focused duplicate server transport test 1 passed; ServerSyncService test file 15 passed; full Tests/Sync_Interop 106 passed; Bandit on server_sync_service.py and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reused the shared outgoing Sync v2 envelope validator in ServerSyncService.push_v2_envelopes. Direct transport pushes now reject duplicate client_envelope_id values before client dispatch while preserving dataset/device identity checks.
<!-- SECTION:FINAL_SUMMARY:END -->
