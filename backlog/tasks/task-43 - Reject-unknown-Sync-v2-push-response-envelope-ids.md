---
id: TASK-43
title: Reject unknown Sync v2 push response envelope ids
status: Done
assignee: []
created_date: '2026-05-10 20:21'
updated_date: '2026-05-10 20:23'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook local-first Sync v2 so push responses cannot reference client_envelope_id values that were not submitted in the current push batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once raises when accepted push response entries reference unknown client_envelope_id values.
- [x] #2 The mismatch is detected before pull, outbox dispatch, outbox attempt updates, or cursor advancement.
- [x] #3 The profile records a push failure that identifies the malformed response.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test for accepted push response entries referencing an unknown client_envelope_id. 2. Validate push response envelope ids against the submitted batch before outbox marking or pull. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added push response envelope-id validation so accepted/rejected/conflict entries must reference client_envelope_id values submitted in the current batch. Malformed responses now fail before pull, outbox dispatch, attempt updates, or cursor advancement. Updated the local-first fake server to echo submitted ids in default accepted responses and verified focused/full Sync_Interop plus Bandit and diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
