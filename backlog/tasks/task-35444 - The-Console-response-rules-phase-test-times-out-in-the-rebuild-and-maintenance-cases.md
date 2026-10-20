---
id: TASK-35444
title: The Console response-rules phase test times out in the rebuild and maintenance cases
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-18 10:10
updated_date: '2026-10-20 00:30'
labels:
- tests
- flaky
- console
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session[maintenance] runs in a required UI Fast Lane shard. TASK-34950 fixed one cause of its intermittent failure, but it failed again on 2026-10-17 with a TimeoutError in run 39050477414 (attempt 1, PR #3113, which changed no Console or UI file), after PR #3137 had changed the same test's held-config setup. A failure on unchanged code counts against a pull request in the merge queue, and two in a row evict it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the remaining timeout is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive pull-request runs in CI
- [x] #3 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the blocked UI-loop stack the test prints in each CI failure.
2. Make the display reads on that path take the live controller instead of ensuring one.
3. Pin it with a test that holds the config rebuild lock and syncs the mounted recovery placeholder.
4. After merge, count 20 consecutive passes in CI.
5. (2026-10-19, after #3163's count failed) Read the new failures' blocked stacks; pin the full UI sync with the rebuild starting at its transcript step, not only the read a stack names; move each refresh read it finds onto the existing deferral.
6. After that merges, count 20 consecutive passes again.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Data: every completed attempt of the pull_request runs of derived-artifacts.yml created from #3152's merge (2026-10-18 08:50Z) to 21:36Z: 36 runs, 35 completed attempts, every head containing the merge; 25 of them finished all ten UI shards, on 11 branches. The test failed in 5 of the 25 (shard 10): [rebuild] 39160862810/1 and 39152595713/1 (feat/tools-hub-in-settings), 39151497191/1 (docs/task-33160-ac2-remeasure, backlog only); [maintenance] 39151491658/1 (docs/task-34991-close, backlog only) and 39148610450/1 (feat/18928-approval-suggestions). All five are the TimeoutError of the 1 s wait after Stop, and all five printed the same blocked UI stack.

Cause (product): every transcript sync calls ConsoleTranscriptRegion.sync_recovery on the UI loop, which reads three display builders: the trace-call recovery state (ConsoleSessionController._console_trace_recovery_state) and the provider-continuation message and replay availability (lambdas in ChatScreen._build_console_center). All three called _ensure_console_chat_controller(), and every such call runs _sync_console_chat_core_state() -> _build_console_provider_selection() -> _provider_readiness_app_config() -> load_settings(). Right after a config save or rebuild the settings cache is cold, so load_settings() takes its guarded path and waits for config's rebuild lock (config_participants.operation(), its lock.acquire(timeout=0.05) loop) on the UI loop. The test holds that lock on purpose; when Stop's transcript sync reached this read during the hold, the loop waited past the test's 1 s. The stack's first builder is the trace-call one; fixing only it would move the wait to the next builder.

Fix: the three display reads take the live controller as it is (the session controller's existing _current_chat_controller_accessor; the screen's _console_chat_controller), and ensure one only when there is none yet, so the first open still creates it as before (review of #3163: a recoverable message restored into the store would otherwise stay hidden until something else created the controller). Only ensuring a live controller re-read config on every call; that is what waited. The recovery actions are unchanged.

Not covered: a full UI sync that starts while a rebuild holds the lock still reads config at its start (_sync_console_chat_core_state in _sync_native_console_chat_ui). None of the five failures stopped there: each blocked sync had already passed that read without waiting.

Evidence: Tests/UI/test_console_response_rules.py::test_recovery_sync_never_waits_for_a_config_rebuild invalidates the config caches under the rebuild lock and syncs the mounted recovery placeholder. Without the fix it fails with 'recovery sync waited 2.032s for the config rebuild'; with it, it passes. It is appended to scripts/ui_pr_gate_census.txt (shard 4); the floor goes 323 -> 324. Tests/UI/test_console_recovery_live_bindings.py now gives the state read a live controller and fails if it ensures a live one, and checks that it ensures one when there is none. Locally the four phase cases, the new test and the live-bindings test pass. AC2 is measured after merge.

Files: tldw_chatbook/UI/Console_Modules/session.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_response_rules.py, Tests/UI/test_console_recovery_live_bindings.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py.

2026-10-19 UTC, AC #2 not met after #3163; more causes found. Every completed pull_request attempt of derived-artifacts.yml created from #3163's merge (2026-10-19 00:48Z) to 20:27Z: 49 runs, 53 attempts; 31 attempts on 14 branches had a head containing the merge (1486e524b0) and finished UI Fast Lane (10). The phase test failed in 7 of them ([rebuild] 4, [maintenance] 3), two on #3165's docs-only heads, so it still fails on unchanged code at about the old rate. All seven printed one blocked stack, a read #3163 did not touch: _sync_native_console_chat_ui -> _current_console_rail_state -> ... -> _provider_readiness_app_config -> load_settings(), waiting on the rebuild lock after the sync's transcript await.

Fix (this PR), three refresh reads moved onto the existing deferral, each pinned:
- The full sync's rail-visibility pass goes through run_console_presentation_sync (checked display config), as wiring.py's rail-visibility sync already does; a deferral hands the rest of the sync to the existing whole-sync replay on the coalesced control-bar timer (config_sync.replay_whole_console_sync, now shared with the rail-and-controls deferral that already did this).
- The transcript sync's guidance refresh goes through run_console_presentation_sync too; its deferral already requests the control-bar sync, which refreshes the same guidance under checked config.
- The transcript's memory banner used context_control_inputs only for the effective memory, but the call also read the global context policy, which waits on a cold cache. It now passes include_global=False (a new keyword; the default keeps every other caller unchanged).
Tried and reverted: putting the full sync's core-state push (the live provider selection a send acts on) behind run_console_config_sync's non-blocking lock probe. Ordinary background config reads hold those locks for milliseconds, so the whole sync deferred constantly; test_a_probe_settling_mid_poll_still_reaches_the_console failed 5 of 5 (dev 5 of 5 passed). That push stays a live read, as before.

Pin: test_console_syncs_never_wait_for_a_config_rebuild[recovery|full-ui] (renamed from test_recovery_sync_never_waits_for_a_config_rebuild; census line updated, floor unchanged). The full-ui case starts the rebuild as the sync's transcript step begins and forces that step past its tick gate. On dev it fails at the rail-visibility read (the CI chain); here it passes; reverting any one of the three fixes makes it fail at that fix's read.

Open, filed as its own task (owner's choice, 2026-10-19: ship these three and file the rest): more refresh reads wait the same way. Locally the phase test still failed 1 of 13 runs, at the transcript's turn-file-cards get_cli_setting in ConsoleTranscript._reconcile_rows; the general approach (screen-by-screen deferral or reads that do not wait) is an owner decision. So AC #2 may still not hold after this merges.

Evidence: the response-rules file passes whole (17); the probe test 5 of 5; The UI census's 70 Console files (xdist -n 8; most failures are the local wall): 111 failed and 1341 passed here, 112 and 1339 on dev 98c7eb308d, and every failure here also fails on dev (test_console_settings_modal_save_disabled_during_active_run failed on dev only). Tests/CI 998 passed. chat_screen.py stays at dev's 24,378 lines (its ratchet was already 9 over budget on dev, from the 2026-10-18 status-strip PR). The UI-ready module census is unchanged (1030 on dev and here, identical lists).

Files: tldw_chatbook/UI/Console_Modules/config_sync.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Chat/console_chat_controller.py, Tests/UI/test_console_response_rules.py, scripts/ui_pr_gate_census.txt, backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
