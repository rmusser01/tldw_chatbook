---
id: TASK-35444
title: The Console response-rules phase test times out in the rebuild and maintenance cases
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-18 10:10
updated_date: 2026-10-18 22:09
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Data: every completed attempt of the pull_request runs of derived-artifacts.yml created from #3152's merge (2026-10-18 08:50Z) to 21:36Z: 36 runs, 35 completed attempts, every head containing the merge; 25 of them finished all ten UI shards, on 11 branches. The test failed in 5 of the 25 (shard 10): [rebuild] 39160862810/1 and 39152595713/1 (feat/tools-hub-in-settings), 39151497191/1 (docs/task-33160-ac2-remeasure, backlog only); [maintenance] 39151491658/1 (docs/task-34991-close, backlog only) and 39148610450/1 (feat/18928-approval-suggestions). All five are the TimeoutError of the 1 s wait after Stop, and all five printed the same blocked UI stack.

Cause (product): every transcript sync calls ConsoleTranscriptRegion.sync_recovery on the UI loop, which reads three display builders: the trace-call recovery state (ConsoleSessionController._console_trace_recovery_state) and the provider-continuation message and replay availability (lambdas in ChatScreen._build_console_center). All three called _ensure_console_chat_controller(), and every such call runs _sync_console_chat_core_state() -> _build_console_provider_selection() -> _provider_readiness_app_config() -> load_settings(). Right after a config save or rebuild the settings cache is cold, so load_settings() takes its guarded path and waits for config's rebuild lock (config_participants.operation(), its lock.acquire(timeout=0.05) loop) on the UI loop. The test holds that lock on purpose; when Stop's transcript sync reached this read during the hold, the loop waited past the test's 1 s. The stack's first builder is the trace-call one; fixing only it would move the wait to the next builder.

Fix: the three display reads take the live controller as it is (the session controller's existing _current_chat_controller_accessor; the screen's _console_chat_controller) and show nothing when there is none, since a paused preparation or a recoverable message only exists on a controller. The recovery actions still ensure the controller.

Evidence: Tests/UI/test_console_response_rules.py::test_recovery_sync_never_waits_for_a_config_rebuild invalidates the config caches under the rebuild lock and syncs the mounted recovery placeholder. Without the fix it fails with 'recovery sync waited 2.032s for the config rebuild'; with it, it passes. It is appended to scripts/ui_pr_gate_census.txt (shard 4); the floor goes 323 -> 324. Tests/UI/test_console_recovery_live_bindings.py now gives the state read a live controller and fails if it ensures one. Locally the four phase cases, the new test and the live-bindings test pass. AC2 is measured after merge.

Files: tldw_chatbook/UI/Console_Modules/session.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_response_rules.py, Tests/UI/test_console_recovery_live_bindings.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
