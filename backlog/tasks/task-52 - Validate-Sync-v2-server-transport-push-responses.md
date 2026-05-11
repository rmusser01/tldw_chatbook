---
id: TASK-52
title: Validate Sync v2 server transport push responses
status: Done
assignee: []
created_date: '2026-05-10 21:32'
updated_date: '2026-05-10 21:34'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the policy-gated ServerSyncService push_v2_envelopes transport so direct callers reject malformed server push responses before returning them as accepted state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ServerSyncService.push_v2_envelopes raises when the server push response dataset_id differs from the request dataset_id.
- [x] #2 ServerSyncService.push_v2_envelopes raises when the server push response references an unknown client_envelope_id.
- [x] #3 The malformed response is rejected after dispatch without mutating local Sync state.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red ServerSyncService tests for mismatched push response dataset_id and unknown response client_envelope_id. 2. Reuse shared push response validation in ServerSyncService.push_v2_envelopes after client dispatch and before returning. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed direct ServerSyncService push responses with mismatched dataset_id and unknown client_envelope_id did not raise before validation. Green/final verification: focused malformed response tests 2 passed; ServerSyncService test file 17 passed; full Tests/Sync_Interop 108 passed; Bandit on server_sync_service.py and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated ServerSyncService.push_v2_envelopes responses with the shared Sync v2 push response scope helper. Direct transport callers now reject mismatched response dataset_id and unknown response client_envelope_id values before treating the push as accepted state.
<!-- SECTION:FINAL_SUMMARY:END -->
