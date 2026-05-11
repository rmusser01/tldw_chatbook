---
id: TASK-47
title: Reject duplicate Sync v2 outgoing envelope ids before push
status: Done
assignee: []
created_date: '2026-05-10 21:09'
updated_date: '2026-05-10 21:11'
labels:
  - sync
  - chatbook
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden Chatbook Sync v2 local-first push assembly so duplicate client_envelope_id values across pending outbox and ad hoc outgoing envelopes are rejected before contacting the server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first sync raises before push when the combined outgoing push batch repeats a client_envelope_id.
- [x] #2 The rejection records a push error without advancing the remote pull cursor.
- [x] #3 The pending outbox entry is not marked attempted for a local validation failure.
- [x] #4 Focused Sync_Interop tests, Bandit on touched service/helper files, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a red local-first test that queues an outbox envelope and passes the same client_envelope_id as an ad hoc outgoing envelope. 2. Validate the combined outgoing push batch before server push. 3. Run focused tests, full Sync_Interop, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red run confirmed the duplicate outgoing batch test raised only after contacting the fake server. Green/final verification: focused duplicate outgoing test 1 passed; local-first service file 25 passed; full Tests/Sync_Interop 100 passed; Bandit on tldw_chatbook/Sync_Interop/local_first_sync_service.py and validation.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated the combined local-first push batch before server push by coercing pending outbox entries and ad hoc outgoing envelopes through the same outgoing scope helper. Added duplicate client_envelope_id detection so a repeated outgoing id fails locally without server contact, cursor movement, or outbox attempt mutation.
<!-- SECTION:FINAL_SUMMARY:END -->
