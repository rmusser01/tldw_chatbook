---
id: TASK-38718
title: "Console send-path reds in the resend-guard, received-intent and hook-admission tests"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-05 11:05
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
- [x] #3 The send acknowledgement detects its laid-out status line, so an acknowledged send dispatches without waiting out every paint hop
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

**Files:** `Tests/UI/test_console_send_resend_guard.py`, `Tests/UI/test_console_received_intent_feedback.py`, `Tests/Chat/test_console_hook_admission.py`; review follow-ups: `UI/Console_Modules/send_acknowledgement.py`, `Tests/UI/test_console_send_acknowledgement.py`.
**Review follow-ups (AC#3, added by the coordinator from the branch review).**
- **Product defect from the same drift.** `send_acknowledgement._acknowledgement_laid_out` still looked for `#console-run-chip`, the widget TASK-35160 retired. It always hit `NoMatches` and returned False, so every acknowledged Enter waited out all `_PAINT_HOPS` (6) frame-length hops, about 100 ms, before it dispatched. It now checks the status line (`#console-status-text`) has a region and that its `run` segment is shown.
  - New test: `Tests/UI/test_console_send_acknowledgement.py::test_an_acknowledged_enter_dispatches_once_its_status_line_is_laid_out` (a census whole-file entry). It records each hop and whether the acknowledgement was laid out.
  - Red before the fix: `[(0, False), (1, False), ... (6, False)]`.
  - Green with it: `[(0, True)]` in 3 of 3 runs, so the send dispatches after the first refresh.
- **Re-measured the 100 ms node**, `test_actual_send_paints_sending_and_accepts_input_within_100ms`, at `-n 2`, head against base (base = this file before the fix).
  - Load 7-8, 12 cases each: the Sending frame median was 21.3 ms on head against 29.7 ms on base. The Preparing frame median was 60.7 ms against 174.7 ms.
  - Eight-way parallel, load up to 12, 16 cases each: Sending 32.9 ms against 33.4 ms, none over 100 ms on either side. Preparing 73.8 ms against 167.4 ms; 2 of 16 over 100 ms on head, 16 of 16 on base.
  - The fix removes the hop wait before dispatch, so the Preparing frame comes about 100 ms sooner. The Sending frame is painted before the hops, so its earlier 117 ms misses at load about 22 are not explained by this defect.
- **`retire_captured_drafts` is pinned again.** New test: `test_a_press_held_on_a_non_append_edit_of_a_sent_draft_is_not_sent` in the resend-guard file. The sequence is Enter, a tab round trip, Home, "x", then Enter again during admission.
  - The round trip gives the draft a new generation, so the second press is held rather than refused. The first send's commit cannot take its text out, so it retires the draft, and the held press is dropped.
  - With `_spent_generations.append(self._draft_generation)` removed from `retire_captured_drafts`, it was red 2 of 2: the held press sent `'xacknowledge this send'` a second time.
  - Green with the line, and the whole file passed 18/18. No PR lane runs this file.
- **Not mine.** Four `Tests/UI/test_console_send_acknowledgement_extended.py` nodes fail on base and head alike: refused_admission_rolls_back, turn_refused_before_its_echo, failed_durable_commit, and second_enter_during_admission.
<!-- SECTION:NOTES:END -->

