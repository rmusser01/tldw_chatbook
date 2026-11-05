---
id: TASK-38718
title: "Console send-path reds in the resend-guard, received-intent and hook-admission tests"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-05 07:14
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
- [x] #1 Each of the seven nodes passes on dev, or is rewritten or retired with the reason recorded in the task notes (for example, a product change it predates)
- [x] #2 For each node, the task notes record whether the cause was test drift or a product defect, and a product defect has a test that fails before its fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the seven nodes on the base and trace each failure to the product change behind it (git history of the asserted surface, a temporary probe where the trace is not obvious).
2. Resend guard `[typed_in_b]`: find what now clears tab A's composer and decide drift or defect against the lead ruling (TASK-33620.15).
3. Received-intent feedback: the run chip and the Status collapsible were retired by TASK-35160; move the frame recorders and display checks to the status line's `run` segment, and retire the collapsed arm.
4. Hook admission: find why the resumed turn never reaches the gateway, and fix the fake or the product.
5. Re-run the three files at `-n 4`, and record the cause per node in the notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**All seven were test drift; no product defect.** On the base (b8608e4cdb) the seven nodes failed as filed in one `-n 4` run of the three files. The fixes are test-only.

- **Resend guard `[typed_in_b]` (drift).** A temporary probe that logged every composer draft change showed what clears tab A. At acceptance, `publish_received_acceptance` commits the store draft with a CAS on A's per-chat draft revision. `_project_console_saved_input` then spends the reloaded capture and loads A's now-empty draft ("A tab reload creates a new capture of the same accepted draft", 077e5bfa09, 2026-10-31). The test's old expectation (the sent text stays, with a "was sent" notice) came from the composer's widget-wide edit serial, which a key typed in tab B moves. The store's per-chat revision does not move, and A's draft did not change, so clearing it is what the lead ruling asks for. Both params now assert that A's composer and saved draft are empty, that no "was sent" notice appears, that B keeps its draft (plus the "q"), and that one message is sent.
- **Received-intent feedback, 5 nodes (drift).** TASK-35160 (c5ceca1247, 2026-10-16) replaced the boxed status chips with one status line and removed the Status collapsible. The test file came from a branch cut before that (3e618d7119), so it still queried `#console-run-chip`, `#console-status-collapsed-copy` and `ConsoleStatusChips.set_collapsed`. The frame recorders now read the status line's cells (`#console-status-text`), and the display checks read the `run` segment through `ConsoleStatusChips.segment_visible/segment_text`.
  - `test_send_feedback_survives_preparation_and_provider_wait` is now parametrized on route only (`[enter]`, `[send-button]`). The collapsed arm and its final "Status hidden" check are retired with the collapsible.
- **Hook admission (drift).** A probe of the resume result showed the turn blocked with "Mix-in attempt could not be saved; response was not sent." Since 7970fbcada (2026-11-01), a persisted chat's turn saves its mix-in attempt before the provider, and `accept_attempt` refuses a persistence without `save_conversation_mixin_attempt` and `db` (`mixin_persistence_unavailable`). The shared `FakePersistence` predates that. The test now uses a local subclass that has both, and the resumed turn reaches the gateway. The shared fake is unchanged; its other users were not checked.

**Evidence (`-n 4`).** The three files together: 59/59, 58/59 and 59/59 at load 11-22. The received-intent file alone, 4 runs at load 21-24: 3 clean, and 1 failure of `test_actual_send_paints_sending_and_accepts_input_within_100ms[send-button]` at 117 ms against its 100 ms Sending-frame budget. The other failure in the three-file runs was the `[enter]` case, on the same budget. The budget is unchanged. It is a timing assertion that loaded hosts miss (TASK-33621.22 recorded a 161 ms miss for the earlier Preparing variant), and no lane runs it. The new frame recorder is not vacuous: each test asserts that the recorder saw the `Sending` or `Waiting for reply` cells.

**Not in scope, seen on the way.** `Tests/Performance/test_console_skill_trust_stock_owned.py::test_stock_console_skill_setup_has_original_owned_lifetime[send_enter|send_button]` share `_NaturalPreparingFrame`. They fail on the base and at the head with the same assertion, "Cold Send did not reach original configuration SQL", which is unrelated to the chip query.

**Files:** `Tests/UI/test_console_send_resend_guard.py`, `Tests/UI/test_console_received_intent_feedback.py`, `Tests/Chat/test_console_hook_admission.py`.
<!-- SECTION:NOTES:END -->

