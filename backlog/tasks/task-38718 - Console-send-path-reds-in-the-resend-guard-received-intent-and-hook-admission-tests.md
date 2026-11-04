---
id: TASK-38718
title: "Console send-path reds in the resend-guard, received-intent and hook-admission tests"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- console
- tests
priority: medium
references:
- Tests/UI/test_console_send_resend_guard.py
- Tests/UI/test_console_received_intent_feedback.py
- Tests/Chat/test_console_hook_admission.py
dependencies:
- TASK-36232
- TASK-38691
- TASK-33621.36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03: the same seven nodes failed in each of two `-n 4` runs through the scratch-profile wrapper (host load about 27).

These are Console send-path tests that no PR lane runs:
- `test_console_send_resend_guard.py` and `test_console_hook_admission.py` are not in the census or in any workflow target;
- the console-p0-gate runs one node of `test_console_received_intent_feedback.py` (`test_actual_send_receives_intent_before_original_configuration_read`, derived-artifacts.yml:1044), and none of the reds below.

**Resend guard (1).** `Tests/UI/test_console_send_resend_guard.py::test_enter_after_a_tab_round_trip_during_admission_sends_the_draft_once[typed_in_b]` (:279). With a key typed in tab B during admission, the test expects the sent text to stay in tab A's composer. It fails at :335 with `assert '' == 'acknowledge this send'`: the composer was cleared. The `[round_trip]` case passes.

**Received-intent feedback (5)**, in `Tests/UI/test_console_received_intent_feedback.py`:
- `test_actual_send_paints_sending_and_accepts_input_within_100ms[enter]` and `[send-button]` (:450): `NoMatches: No nodes match '#console-run-chip' on ChatScreen`, from the frame recorder's query (:82);
- `test_stopping_received_send_clears_feedback_and_keeps_draft` (:865): the same `NoMatches`;
- `test_send_feedback_survives_preparation_and_provider_wait[enter-False]` and `[send-button-True]` (:787): `AttributeError: 'ConsoleStatusChips' object has no attribute 'set_collapsed'` at :797.

**Hook admission (1).** `Tests/Chat/test_console_hook_admission.py::test_hooks_added_after_acceptance_retain_the_durable_recovery_owner` (:227). It fails at :255 with `assert (True and None)`. `resume_durable_postcommit` accepted the resume, but the stub gateway never received the messages.

**Boundary.** These reds belong to other tasks, so this one excludes them:
- the four `Tests/UI/test_console_send_draft_snapshot.py` reds (fresh_profile_first_send, submit_exception_retains, deleted_owner_refuses, enter_schedules_visible_send): TASK-38691 and TASK-33621.36;
- `test_actual_send_receives_intent_before_original_configuration_read`: TASK-36232;
- the durable received-intent harness's exchange-capture failure: TASK-33621.82.

No open task names any of the seven nodes above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the seven nodes passes on dev, or is rewritten or retired with the reason recorded in the task notes (for example, a product change it predates)
- [ ] #2 For each node, the task notes record whether the cause was test drift or a product defect, and a product defect has a test that fails before its fix
<!-- AC:END -->
