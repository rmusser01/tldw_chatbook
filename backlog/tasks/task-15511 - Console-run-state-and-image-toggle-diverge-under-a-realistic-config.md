---
id: task-15511
title: Console run state and image toggle diverge under a realistic config
status: To Do
assignee: []
labels:
  - bug
  - console
  - test-health
priority: medium
---

## Description

task-15270 made the Console test harness boot with the real (sandboxed) config
instead of a near-empty synthetic dict. Two `Tests/UI/test_console_native_chat_flow.py`
tests went red as a result. Both had only ever exercised their subject under a
config that supplied none of the relevant settings, so neither red is a
harness artifact -- each is a behaviour that had never been observed.

**1. `test_console_native_generic_provider_send_renders_completed_message`.**
A non-streaming send renders its response ("generic provider response" appears,
and `chat_api_call` is reached with the right endpoint and key), but
`console._ensure_console_chat_controller().run_state.status` is
`ConsoleRunStatus.IDLE` where the test expects `COMPLETED`. Measured: this is
NOT a race -- the status is IDLE immediately after the response renders and
stays IDLE across 40 further pumps (~0.8s). `chat_defaults.streaming` is None in
this test, so the streaming toggle is not the trigger.

Two candidate causes, neither confirmed: the completed run never transitions the
controller's state under this path, or the send is served by a different
controller instance than `_ensure_console_chat_controller()` returns, in which
case the assertion is reading the wrong object. Worth distinguishing before
fixing, because run state drives user-visible affordances (stop control,
spinner, cost ticker) and only the first cause is user-visible.

**2. `test_image_message_gets_inline_row_after_prep_and_toggle_cycles`.**
The inline image row appears after prep, then disappears after the first
pixels -> graphics toggle, where the test asserts it is still present. Likely a
config-selected image rendering mode that renders nothing rather than falling
back -- if so the user-visible symptom is an image vanishing on toggle.

## Acceptance Criteria

- [ ] The cause of the IDLE run state is identified as either a missing transition or a controller-identity mismatch, and stated with evidence
- [ ] If the run state is genuinely not settling to COMPLETED after a successful send, it is fixed and the user-visible affordances driven by it are checked
- [ ] The image row survives the pixels -> graphics toggle, or the toggle falls back to a mode that renders something and the test asserts that instead
- [ ] Both tests pass with their xfail(strict=True) markers removed
