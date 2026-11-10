---
id: TASK-39166
title: "Load-sensitive Console UI tests in test_console_response_rules.py and test_console_received_intent_feedback.py fail at load 30-118, and several of the response-rules nodes are UI census entries"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 18:17
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
- [x] #2 Each node waits on the state it reads, with a bound that fails naming that state; no latency budget is loosened to make it pass
- [ ] #3 Both files pass 10 of 10 whole-file runs at `-n 4`, with load recorded, at least five of them at a one-minute load of 40 or more
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce each listed node under load (whole files at `-n 4`, runs in parallel to raise the load) and name, from its own traceback, the wait that missed and the state it waited for.
2. `test_console_response_rules.py`: replace the fixed-time watchdogs (2 s / 5 s from the hold's start) with a detector of the state they stand for, a UI loop that stops running callbacks while the rebuild holds config; let the holders hold until the test releases them; turn the bare `asyncio.timeout` waits on events and flags into named bounded waits; bound the held-transport test's teardown, not its mount.
3. `test_console_received_intent_feedback.py`: per the failures found in step 1, wait on each turn's own end condition with a bound that names it.
4. Keep every latency budget (the mount test's five-second after-release delivery budget, TASK-38948's 100 ms send budget, which this task does not touch).
5. Red evidence: the old tests failed under load (step 1); a temporary product edit that makes a display sync wait for the rebuild still fails the converted tests on the new detector.
6. Both files 10 of 10 whole-file runs at `-n 4`, load recorded, at least five at load 40 or more.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Base dev ae9f9e8d21, 2026-11-10. Test-only changes in both files. One listed node, `test_console_mount_refreshes_never_wait_for_a_config_rebuild[roleplay]`, now fails on a real product wait (below), so AC#1 and AC#3 are not fully met and the task stays In Progress.

**AC#1: what missed, from each node's own failure.** All runs were whole files at `-n 4` through run.sh. Runs went in parallel to raise the load; 1-minute load is given.
- `test_fixture_failure_retires_held_transport[primary|helper]`. The outer `asyncio.timeout(10)` ran out inside `host.run_test()` while it awaited `app_ready_event`. That is the mount, before the test's first step. Both cases failed in every run that listed failures by name: 2 of 2 runs at load 54-60 before the change, 4 of 4 base runs at load 67-99.
- `test_console_syncs_never_wait_for_a_config_rebuild[full-ui-held|full-ui]`. The 2 s `threading.Timer` started with the hold fired: "full-ui-held sync waited 2.943s / 3.008s / 3.083s" and "full-ui sync waited 2.465s / 2.357s", at load 25-68. Every stack it recorded ended in `selectors.select`. The loop was idle: the sync was slow, not waiting.
- `test_console_mount_refreshes_never_wait_for_a_config_rebuild[mount]`. "the UI loop did not progress during the hold" (7 < 15). The observation loop stopped at its fixed 2 s deadline after 7 `asyncio.sleep(0.05)` turns, at load 45-58.
- `test_console_mount_refreshes_never_wait_for_a_config_rebuild[roleplay]`. A real UI-thread wait, but a harness-only one. The hold began at the `_sync_console_identity_surfaces` entry, before `CompactModelBar.on_mount`. That `on_mount` took its direct-read route: `get_cli_providers_and_models()` -> `load_settings()` -> the config-participant lock loop, which is the recorded stack.
  - A temporary probe showed `_capture_setup` returning None in compose, with `_sources_current()` True and no hold yet. `ConsoleHarness` hosts `ChatScreen` on its own App, so `widget.app is not app_instance`. In production the bar reads off the loop.
  - Seen in 1 of 2 whole-file runs at load 54-60, 4 of 8 base runs of the node alone at load 32-38, and 12 of 12 probed node runs at load 45-58.
- `test_rule_phase_events_project_only_the_mounted_current_session`. `[maintenance]` failed once on base, at load 67-90, on the `asyncio.timeout(2)` wait for the attach reconcile and its control-bar sync (`:462`). `[rebuild]` runs the same step before its deferral branch. It did not fail in about 10 runs here, so its own failure is not recorded.
- `test_pressed_capture_survives_identical_retyping_before_promotion[True]`. It failed at `:744`, the 15 s wait for the turn's custody to clear after the held read was released. That was 1 of 6 runs at load 54 rising to 101-110.
- Not reproduced: `test_enter_sends_pressed_body_after_clear_and_type_before_paint_dispatch`, `test_navigation_after_receipt_…` and `test_send_feedback_survives_preparation_and_provider_wait`. None of them failed in 9 runs before the change (load 25-110) or in 11 runs of a base copy (load 59-99). TASK-38948's notes name their 15 s bounded waits. They were converted by shape: their 15 s custody waits, and in the feedback test the provider-entry wait and the provider's own `release.wait(15)`.

**AC#2: what changed.**
- `test_console_response_rules.py`:
  - `_UiLoopWatch` replaces the fixed-time watchdogs in the rule-phase, syncs and mount tests. It posts a callback to the loop every 50 ms. It fires only when one has not run in 10 s, records the loop thread's stack, and releases the hold.
  - The holders hold until the test releases them (60 s patience) and no longer end the hold on their own after 5-10 s.
  - Waits on events and flags use `_until`/`_until_set`, a 30 s bound that fails naming the state.
  - The held-transport test bounds only its teardown (a rescheduled `asyncio.timeout`), not the mount.
  - The mount test observes the hold until 15 loop turns and the expected deferral, with a 30 s bound, instead of for a fixed 2 s.
  - The harness's compact-bar reads, only from `CompactModelBar.compose` and `on_mount` frames, get providers read before the hold.
  - `test_workspace_command_opens_same_manager_in_registered_scope` was not listed but failed 2 of 10 runs: `workers.wait_for_complete()` raised `WorkerCancelled` when a newer sync replaced a worker. It now waits until the activation's workers have finished, cancelled ones included.
- `test_console_received_intent_feedback.py`:
  - `_turn_settles` waits for the released turn's own task, then for its custody to clear. Each wait has a 60 s bound and fails naming its state.
  - The provider-entry wait ends when the provider is entered or the turn ends, and names which.
  - The provider's own `release.wait` is 60 s, so it cannot fail a run that otherwise passes.
  - The remaining `_until` asserts name their state.
- Unchanged:
  - the mount test's five-second after-release delivery budget;
  - the 2 s Sending-frame and 5 s run-segment waits;
  - `test_actual_send_paints_sending_and_accepts_input_within_100ms`.
- The 2 s and 5 s timers were block detectors misreading slowness, not budgets: their recorded stacks were idle in `select`.

**Red evidence.**
- Before the change: the failures listed under AC#1.
- A temporary `load_settings()` at the top of `ChatScreen._sync_console_control_bar` failed all six config-hold nodes on the watch, at load 49-60: `syncs[full-ui|full-ui-held]`, `mount[mount|roleplay]` and `rule_phase[rebuild|maintenance]`. Each stack named `chat_screen.py _sync_console_control_bar -> load_settings`.
- A fixture that never releases the held transport (a temporary edit of `response_rules_fixtures.py`):
  - `[primary]` failed `assert release.is_set()`.
  - `[helper]` hung to the 300 s pytest timeout in `owner.dispose()`. No asyncio bound in the test escapes that wait; the old 10 s bound sat on the same await.

**AC#3.**
- `test_console_received_intent_feedback.py`: 11 of 11 whole-file runs at `-n 4`, at load 44-91; every run started and ended at load 44 or more.
- `test_console_response_rules.py`: 13 runs at load 44-80. Every listed node except `mount[roleplay]` passed every run.
  - `[roleplay]` failed 8 of 13 runs. Each failure was the watch's stack on a product wait: `ConsoleComposerBar.on_mount` runs `PromptHistory.load()` on the loop. `FileJob(self, "prompt_history")` calls `raw._async_source_selection` -> `default_prompt_history_path()` -> `get_user_data_dir()`, which takes the rebuild lock (`config_participants.py:427`) on the UI thread.
  - Removing the harness-only compact-bar wait let the hold reach this second wait. Alone it failed 5 of 15 at load 35-45, against base's 4 of 8 at load 32-38 on the compact-bar wait, so the node is no flakier than on base.
- The fix is product code (resolve the history path off the loop), so it is left for the coordinator (NEEDS_CONTEXT). This node is a UI census entry.

**Files.** `Tests/UI/test_console_response_rules.py`, `Tests/UI/test_console_received_intent_feedback.py`, `backlog/docs/lessons-testing-evidence.md` (new entry).

**Parent's combined runs (2026-11-10, for TASK-38977 AC#1).** Ten runs of this file and `test_console_prompts_modal.py` in one `-n 4` invocation, at load 36-47: 4 passed. `[roleplay]` failed 4 times on the `PromptHistory` wait above. `test_rule_phase_events_project_only_the_mounted_current_session[rebuild]` failed once, on the watch, with a second UI-thread config read: `_sync_console_native_session_tabs` -> `_maybe_show_fleet_coachmark` -> `_ensure_console_chat_controller` -> `_sync_console_chat_core_state` -> `_build_console_provider_selection_uncached` -> `_provider_readiness_app_config` -> `load_settings()` -> the participant lock loop (`config_participants.py:427`). Whether the real app routes that read the same way under `ConsoleHarness`'s own App was not checked. Two unlisted nodes also failed and were not captured: `test_actual_stop_after_generation_cancels_rule_check` (3 of the first 3 runs, 0 of the next 7; it waits a fixed 60 and 10 `pilot.pause()` turns) and, in the prompts file, `test_preservation_veto_retains_candidate_in_review_and_tamper_blocks_apply` (once).
<!-- SECTION:NOTES:END -->
