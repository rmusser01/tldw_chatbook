---
id: TASK-39139
title: "After a Console regenerate the conversation has no persisted active leaf, so test_regenerate_replacement_identity_resolves_after_persistence fails"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_trajectory_capture.py
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/DB/ChaChaNotes_DB.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38111's notes (round-11 branch fu/r11-recovery) list it. Line numbers are dev's.

- `Tests/Chat/test_trajectory_capture.py::test_regenerate_replacement_identity_resolves_after_persistence` (`:523`) fails at `:555-556` with `assert None is not None`: after `regenerate_message`, `db.get_conversation_active_leaf` returns None.
- The trajectory projection then has no active-leaf anchor for the replacement's identity.
- The RecoveryRequired trap hid the node until TASK-38111. It is red on its own assertion alone, at base and on dev under a marker plugin.
- Whether regenerate fails to persist the cursor or the test is stale is not yet known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes record the cause as a product gap or as test drift, with the commit
- [ ] #2 If it is a product gap: after a regenerate the persisted active leaf is the replacement, and the test passes; it is red before the fix
- [ ] #3 If it is test drift: the test is re-pinned without dropping its replacement-identity and ordering assertions
- [ ] #4 The whole file passes 3 times at `-n 4`, with the load recorded
<!-- AC:END -->
