---
id: TASK-15511
title: Console run state and image toggle diverge under a realistic config
status: Done
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

- [x] The cause of the IDLE run state is identified as either a missing transition or a controller-identity mismatch, and stated with evidence
- [x] If the run state is genuinely not settling to COMPLETED after a successful send, it is fixed and the user-visible affordances driven by it are checked
- [x] The image row survives the pixels -> graphics toggle, or the toggle falls back to a mode that renders something and the test asserts that instead
- [x] Both tests pass with their xfail(strict=True) markers removed

## Implementation Plan

1. Probe the run-state map and session keying at assertion time
2. Spy the terminal-state clear to find its caller, then diff the trigger
3. Reproduce the image default-mode resolution outside the test
4. Mutation-check the controller fix; run the module whole

## Implementation Notes

**Run state: neither of the filed candidates.** The transition was not missing
(the session's history reads IDLE -> VALIDATING -> STREAMING -> COMPLETED) and
the controller/session identity matched. The COMPLETED state was recorded and
then WIPED: the post-send UI resync (`_sync_native_console_chat_ui ->
_sync_console_chat_core_state -> update_provider_selection`) compared selection
tuples, saw `configured_model` flip `None -> "gpt-5.6-terra"`, and took the
"user changed provider settings" branch, which clears the active session's
terminal run state. `configured_model` is DERIVED state -- the config fallback
model, late-resolving once a provider key exists -- so its churn through a
routine resync is not a user action. Third instance of the task-15740/15673
class: the app's own derived-state churn read as user input.

Fix: the clear-trigger now compares EFFECTIVE selections -- the model term is
`self.model or self.configured_model`, the exact resolution the send path uses
(`model = selection.explicit_model or selection.configured_model`). A
configured-model change with an explicit model set changes nothing about what
would run and no longer wipes the state; with NO explicit model it still
clears, because the effective model genuinely changed. User-visible effect of
the bug: a completed run's terminal state (and everything driven by it)
vanishing right after completion whenever late resolution churned.
Mutation-checked: restoring the two-field comparison turns the test red.

**Image toggle: the product was right; the test's premise never arrived.** The
test already pinned `default_render_mode = "pixels"` -- through
`app_config["chat"]`, the dead seam the task-15270 triage called out by name:
`_chat_images_config` prefers `COMPREHENSIVE_CONFIG_RAW`, which the real
harness config always carries. Measured on this machine: `auto` resolves via
`terminal_overrides` (iterm2 -> regular) to **graphics**, so the first toggle
in the pixels -> graphics -> hidden cycle landed on hidden and the row
vanished one step "early". The pin now writes through the RAW section too.

Also repaired (pre-existing on untouched dev, same neighbourhood):
`test_screen_selection_builder_targets_session_without_switching_view` -- its
SimpleNamespace double predates task-15452's memo split
(`_console_derivation_memo` + `_build_console_provider_selection_uncached`);
taught the double the class-default and bound the borrowed uncached half.

Modified: `tldw_chatbook/Chat/console_chat_controller.py`,
`Tests/UI/test_console_native_chat_flow.py`,
`Tests/Chat/test_console_turn_execution_context.py`.
