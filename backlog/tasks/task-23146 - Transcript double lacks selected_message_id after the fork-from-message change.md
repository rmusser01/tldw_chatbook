---
id: TASK-23146
title: Transcript double lacks selected_message_id after the fork-from-message change
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - console
priority: medium
dependencies: []
---

## Description

All 3 tests in `Tests/UI/test_console_turn_activity_line.py` fail with
`AttributeError: 'types.SimpleNamespace' object has no attribute 'selected_message_id'`. Production
is correct; the transcript double predates the fork-from-message feature.

## Acceptance Criteria

- [ ] All 3 tests pass with a transcript double carrying the attributes the sync path reads
- [ ] The double's shape is derived from the production contract rather than patched attribute by
  attribute as each `AttributeError` appears

## Evidence

`tldw_chatbook/UI/Screens/chat_screen.py:11994` `selected_id = transcript.selected_message_id`, at
the top of `_sync_native_console_transcript` (starts `:11983`).

Introduced by `5d9b4bec5a` (2026-08-27) "feat(console): fork chat from a selected message (#2152)",
which updated 5 other UI test files but not this one. Also needs `set_fork_eligibilities`.
