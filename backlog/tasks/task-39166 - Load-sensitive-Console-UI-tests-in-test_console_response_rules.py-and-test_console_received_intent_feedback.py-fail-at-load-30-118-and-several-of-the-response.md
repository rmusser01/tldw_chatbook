---
id: TASK-39166
title: "Load-sensitive Console UI tests in test_console_response_rules.py and test_console_received_intent_feedback.py fail at load 30-118, and several of the response-rules nodes are UI census entries"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
- flaky
- ci
priority: low
references:
- Tests/UI/test_console_response_rules.py
- Tests/UI/test_console_received_intent_feedback.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-36511
- TASK-38948
- TASK-38977
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-38977 and TASK-38948 on round-12 branch fu/r12-ctests). The evidence is from their notes. The census entries were checked on the round-12 stack 9091820bf2, 2026-11-08.

**`test_console_response_rules.py`.**
- Nine whole-file runs at `-n 4` and load 32-98 each failed 1-6 of these (TASK-38977's notes):
  - `test_fixture_failure_retires_held_transport[primary|helper]`;
  - `test_console_syncs_never_wait_for_a_config_rebuild[full-ui-held|full-ui]`;
  - `test_console_mount_refreshes_never_wait_for_a_config_rebuild[roleplay|mount]`.
- Base, at load 52-75, failed the same four tests.
- The coordinator's list also names `test_rule_phase_events_project_only_the_mounted_current_session[rebuild]` from the same runs.
- TASK-36511's notes saw `test_fixture_failure_retires_held_transport[primary]` flake on its base too.
- Several are UI census entries (`scripts/ui_pr_gate_census.txt`):
  - `test_console_syncs_never_wait_for_a_config_rebuild`;
  - `test_console_mount_refreshes_never_wait_for_a_config_rebuild[mount]`, `[view-start]` and `[roleplay]`;
  - `test_fixture_failure_retires_held_transport`;
  - `test_rule_phase_events_project_only_the_mounted_current_session`.

  PR runs have been green, so the risk is a loaded runner.

**`test_console_received_intent_feedback.py`.** Ten runs at load 93-118 failed 1-6 per run, each on its own 15 s bounded wait (TASK-38948's notes):
- `test_pressed_capture_survives_identical_retyping_before_promotion`;
- `test_enter_sends_pressed_body_after_clear_and_type_before_paint_dispatch`;
- `test_navigation_after_receipt_…`;
- `test_send_feedback_survives_preparation_and_provider_wait`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each listed node, the step that misses under load is named from its own failure: which wait, and which state it was waiting for
- [ ] #2 Each node waits on the state it reads, with a bound that fails naming that state; no latency budget is loosened to make it pass
- [ ] #3 Both files pass 10 of 10 whole-file runs at `-n 4`, with load recorded, at least five of them at a one-minute load of 40 or more
<!-- AC:END -->
