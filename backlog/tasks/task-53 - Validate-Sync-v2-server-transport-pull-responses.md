---
id: TASK-53
title: Validate Sync v2 server transport pull responses
status: Done
assignee: []
created_date: '2026-05-10 21:38'
updated_date: '2026-05-10 21:40'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the policy-gated ServerSyncService pull_v2_envelopes transport so direct callers reject malformed server pull responses before returning them to higher-level sync flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ServerSyncService.pull_v2_envelopes raises when the server pull response dataset_id differs from the request dataset_id.
- [x] #2 ServerSyncService.pull_v2_envelopes raises when an incremental pull response includes an envelope from the requesting device.
- [x] #3 ServerSyncService.pull_v2_envelopes raises when a non-empty pull response omits next_cursor.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red ServerSyncService tests for malformed pull responses: mismatched dataset_id, own-device incremental envelope, and non-empty page without next_cursor. 2. Reuse shared pull response and pagination validators in ServerSyncService.pull_v2_envelopes after client dispatch and before returning. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed direct ServerSyncService pull responses with mismatched dataset_id, own-device incremental envelopes, and non-empty missing next_cursor did not raise before validation. Green/final verification: focused malformed pull response tests 3 passed; ServerSyncService test file 20 passed; full Tests/Sync_Interop 111 passed; Bandit on server_sync_service.py and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated ServerSyncService.pull_v2_envelopes responses with the shared Sync v2 pulled-response and pagination helpers. Direct transport callers now reject mismatched response dataset_id, own-device incremental envelopes, and uncheckpointable non-empty pages before returning pull data.
<!-- SECTION:FINAL_SUMMARY:END -->
