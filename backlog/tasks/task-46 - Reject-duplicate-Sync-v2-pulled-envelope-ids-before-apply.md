---
id: TASK-46
title: Reject duplicate Sync v2 pulled envelope ids before apply
status: Done
assignee: []
created_date: '2026-05-10 21:03'
updated_date: '2026-05-10 21:05'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first and restore apply paths so a pulled response cannot apply duplicate client_envelope_id values in one batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync raises when a pulled response repeats the same client_envelope_id before local apply.
- [x] #2 Restore raises when a pulled response repeats the same client_envelope_id before local apply.
- [x] #3 The local-first failure records an apply error without advancing the remote pull cursor.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red local-first and restore tests for duplicate pulled client_envelope_id values. 2. Extend pulled response validation to reject duplicate client_envelope_id values before apply. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed the two duplicate-pull tests failed before validation change. Green/final verification: focused duplicate tests 2 passed; affected local_first+restore files 34 passed; full Tests/Sync_Interop 99 passed; Bandit on tldw_chatbook/Sync_Interop/validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added duplicate client_envelope_id validation for pulled Sync v2 responses before local apply. Covered both local-first sync and restore so duplicate pulled envelopes fail without advancing the cursor or mutating local note state.
<!-- SECTION:FINAL_SUMMARY:END -->
