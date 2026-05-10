---
id: TASK-48
title: Reject incomplete Sync v2 push responses before dispatch
status: Done
assignee: []
created_date: '2026-05-10 21:13'
updated_date: '2026-05-10 21:14'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first push response validation so every submitted client_envelope_id must be acknowledged as accepted, rejected, or conflicted before outbox dispatch or cursor movement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync raises when a push response omits one submitted client_envelope_id.
- [x] #2 The rejection records a push error without advancing the remote pull cursor.
- [x] #3 No pending outbox entries are marked dispatched or attempted for an incomplete push response.
- [x] #4 Focused Sync_Interop tests, Bandit on touched helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test where two submitted outbox envelopes receive only one push response outcome. 2. Extend push response scope validation to require an accepted/rejected/conflict outcome for every submitted client_envelope_id. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed the incomplete push response test did not raise before the validation change. Green/final verification: focused incomplete push response test 1 passed; local-first service file 26 passed; full Tests/Sync_Interop 101 passed; Bandit on tldw_chatbook/Sync_Interop/validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Required Sync v2 push responses to acknowledge every submitted client_envelope_id exactly once across accepted, rejected, and conflicts. Incomplete push responses now fail before outbox dispatch, cursor movement, or outbox attempt mutation.
<!-- SECTION:FINAL_SUMMARY:END -->
