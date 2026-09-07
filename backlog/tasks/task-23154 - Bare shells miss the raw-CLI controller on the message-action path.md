---
id: TASK-23154
title: Bare shells miss the raw-CLI controller on the message-action path
status: To Do
assignee: []
created_date: '2026-08-28'
labels:
  - tests
  - console
priority: medium
dependencies:
  - task-23144
---

## Description

15 tests in `Tests/Chat/test_console_generation_actions.py` fail because a bare `ChatScreen` shell
lacks the raw-CLI controller, which `handle_console_message_action` reads. This is the same class of
break TASK-23144 fixed, but through a **different entry path**: 23144's widened ratchet derives its
required set by exercising the chat-store setter, so it is green while this second path stays
unguarded.

That is the interesting part, and the reason this is filed rather than folded into 23144: proving
one entry path wires its controllers says nothing about another. The durable fix covers both paths,
not just this one controller.

## Acceptance Criteria

- [ ] The 15 failing tests pass with the raw-CLI controller wired through the shared stub helpers in
  `Tests/UI/console_controller_stubs.py`
- [ ] The bare-shell ratchet covers the message-action path as well as the chat-store setter path, so
  a controller added to **either** fails at the guard rather than in unrelated tests
- [ ] The extension is proven by a negative control (drop a stub, watch the guard name it), per the
  pattern 23144 established

## Evidence

Reached from `handle_console_message_action`; the controller arrived with PR #2151. Confirmed
pre-existing on dev, not caused by TASK-23144: a failure-name-set diff of
`Tests/Chat/test_console_generation_actions.py` and `Tests/UI/test_console_live_work_handoffs.py`
between pristine dev and the 23144 branch showed **40 fixed, 0 new**, with these 15 unchanged in
both arms.
