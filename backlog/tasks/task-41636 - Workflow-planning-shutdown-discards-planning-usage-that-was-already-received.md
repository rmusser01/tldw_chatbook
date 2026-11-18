---
id: TASK-41636
title: "Workflow planning shutdown discards planning usage that was already received"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- accounting
priority: low
references:
- Tests/Chat/test_workflow_planning_native.py
dependencies:
- TASK-41314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41314 retargeted the native planning shutdown node: it now waits for the worker and accepts either current or corrected accounting. The current behaviour drops the usage of a planning call whose reply had already arrived when shutdown began, so spend is under-reported. The review asked that the test stop pinning the defect, and this task fixes it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Usage from a planning reply received before shutdown is recorded
- [x] #2 The shutdown node in `Tests/Chat/test_workflow_planning_native.py` asserts the corrected accounting only
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (accounting keeps ADR-356's settlement contract; it only stops discarding usage the provider already reported).

1. Red first: the probe's `shutdown-worker-tail` branch and `test_shutdown_waits_for_same_native_planning_worker_return` assert only the corrected outcome (settled no-plan drain, tokens `settled` at 10, no recovery receipt); run it on the unfixed tree.
2. Cause: `_stream_generic_chat` receives the worker's terminal `done` (reply fully consumed, usage recorded), then waits for the held worker; shutdown's stop poll cancels that wait, the cancellation is re-raised through `_automatic_generation`, whose `completed` is False, so it settles the reservation as uncertain.
3. Fix: the call signals record that the reply arrived (`done` received); `_automatic_generation` reads usage when the call completed or its reply arrived. Interruptions before the reply keep today's conservative (uncertain) settlement.
4. Mutation check; run the planning file at `-n 4` plus the automatic-work/settlement files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** The worker in `_stream_generic_chat` consumes the provider reply and records its usage. Then it enqueues the terminal `done` item, and only after that does its guarded call return. The probe holds that return. The consumer reads `done` and then, in its `finally`, waits for the worker task. Shutdown's stop poll (TASK-33666, `watch_content_stalls`) cancels that wait, and the planning path re-raises the cancellation once the worker returns. The cancellation unwinds through `_automatic_generation` with `completed` still False, so the reservation settles with `actual=None` (uncertain). `settle_workflow_phase` then refuses with `workflow_not_settled`, and ADR-356's recovery path runs.

**Fix.** `ConsoleProviderCallSignals.reply_received` is set when the generic stream reads `done`. `done` arrives only after a reply that was consumed whole: errors are queued ahead of it and raise first, and a stop suppresses it. `_automatic_generation` now reads usage when the call completed or its reply arrived. An interruption before the reply arrived still keeps the conservative uncertain settlement, because a partial stream's usage (for example, Anthropic's `message_start` counts) would under-report spend. The probe's shutdown branch and the node now assert only the corrected outcome: driver result with no exception, tokens `("settled", 10)`, no recovery receipt, and the settled no-plan marker.

**Evidence** (scratch-profile wrapper):
- Red first: the retargeted node fails on the unfixed gateway with `AutomaticWorkRefused('workflow_not_settled')` (50 s, load ~46). With the fix it passes (73 s, load ~95).
- Mutation: dropping `or signals.reply_received` brings back the same `workflow_not_settled` failure (65 s).
- The whole planning file at `-n 4`: 27 passed in 311 s, load ~70.
- The accounting and settlement set at `-n 4` (`test_workflow_settlement_delivery`, `test_automatic_provider_budget`, `test_automatic_wake_budget`, `test_console_provider_gateway`, `test_stream_stall_watchdog`, `test_console_stop_silent_provider`, `test_console_failed_attempt_settlement`, together with the TASK-41638 files): 39 failed / 686 passed at load 65-80. The 39 failing nodes were rerun on base, where 27 fail. The 12 that failed only on head were rerun on head at `-n 4`, and 10 passed. The remaining 2 settlement nodes (`test_planning_root_waits_for_original_control_return[control-unknown]`, `test_stop_pending_initial_acceptance_drains_original_cell`) were run alone on each tree at the same time: 1 failed and 1 passed on head, and 1 failed and 1 passed on base. Their failures are fixed 10 s waits inside the children (`usage_ready.wait(10)`, `TimeoutError`).

Files: `tldw_chatbook/Chat/console_provider_gateway.py`, `Tests/Chat/workflow_planning_probe.py`, `Tests/Chat/test_workflow_planning_native.py`, `Tests/Architecture/test_module_size_ratchet.py` (8371 -> 8376).
<!-- SECTION:NOTES:END -->
