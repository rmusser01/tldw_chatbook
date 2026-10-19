---
id: TASK-35576
title: An overlapping open that reused a chat's session keeps it when the first open rolls back
status: To Do
created_date: 2026-10-19 09:10
dependencies: []
labels:
- model-config-redesign
- follow-up
- console
priority: low
references:
- tldw_chatbook/UI/Console_Modules/workspace.py
- tldw_chatbook/Chat/console_conversation_hydration.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while investigating PR #3169's CI timeout (2026-10-19), in the reuse path TASK-33667 relies on. Two overlapping opens of one saved chat make one session: open A registers it (`restore_persisted_session`), and open B, reaching the reuse check after that, reuses and activates it and returns True. But A still has awaits after registering. If A then fails, or its `resume_if` turns false, `rollback_restored_session` removes the session and brings back A's prior session. B has already reported success, so the chat B opened is gone from the store. It needs a hydration failure or a navigation change inside that window, so it is rare; nothing has been seen live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When an open rolls back a session that an overlapping open has already reused and reported, the reused session stays, or the second open's result reports the failure; the user is never left on a chat that was reported open and then removed
- [ ] #2 A test drives two overlapping opens with the first failing after registration, and it fails before the fix
<!-- AC:END -->
