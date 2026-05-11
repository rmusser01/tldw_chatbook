---
id: TASK-42
title: Reject mismatched Sync v2 push response datasets
status: Done
assignee: []
created_date: '2026-05-10 20:16'
updated_date: '2026-05-10 20:18'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook local-first Sync v2 so push responses whose dataset_id differs from the active dataset cannot dispatch local outbox entries or advance sync state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 sync_once raises when push_v2_envelopes returns a dataset_id different from the profile dataset_id.
- [x] #2 Outbox entries remain pending and unattempted when the push response dataset_id is mismatched.
- [x] #3 The profile records the push failure without advancing the remote pull cursor.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test for mismatched push response dataset_id. 2. Validate push response dataset identity before outbox result marking or pull. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added push response dataset validation before local-first outbox result marking or pull. Mismatched push responses now raise, preserve pending outbox entries without incrementing attempts, record the profile push failure, and leave the remote pull cursor unchanged. Verified focused/local-first/full Sync_Interop tests, Bandit, and git diff checks.
<!-- SECTION:FINAL_SUMMARY:END -->
