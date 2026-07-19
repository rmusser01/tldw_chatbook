---
id: TASK-232
title: Gate mid-run retry/regenerate/continue before spawning console-run workers
status: Done
assignee:
  - '@claude'
created_date: '2026-07-14 22:05'
updated_date: '2026-07-15 03:03'
labels:
  - console
  - bug
  - streaming
dependencies:
  - task-228
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-228's pre-merge review (Important #2, pre-existing — not a regression of that fix). With a run streaming into message X, the user can select an older completed/failed message Y and trigger retry/regenerate/continue: `handle_console_message_action` (`chat_screen.py` ~8240) has no run-state guard, and the pure dispatcher only blocks actions on the *selected* message's status (`Chat/console_message_actions.py:161-167`). The new worker spawns in `group="console-run"` with `exclusive=True`, cancelling the in-flight run at worker-creation time — *before* the new coroutine reaches `_active_run_rejection()` (`console_chat_controller.py:1105-1112`), which then rejects the new action against the stale STREAMING state. Because `_stop_requested` is False, both stream paths re-raise `CancelledError` without finalizing, so run_state stays STREAMING and the row stays status "streaming" — the same stuck-[streaming] face as TASK-228's V1, now user-triggered. Recoverable via Stop (`stop_active_run` falls back to scanning the store for a streaming row), which is why this rode as a fast-follow instead of blocking #635. Fix shape per the review: mirror the submit gate (`chat_screen.py` ~7484) — check `controller.run_state.is_send_allowed` before dispatching the retry/regenerate/continue workers (~8302/8310/8355) and notify ("A Console run is already running.") instead of spawning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Triggering retry/regenerate/continue while a Console run is active does not cancel the in-flight run; the user gets the already-running notification and the stream completes and finalizes normally
- [x] #2 Retry/regenerate/continue still work when no run is active (regression coverage)
- [x] #3 A test reproduces the mid-run cancellation RED against the ungated dispatch and GREEN with the gate
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test: fake in-flight run (run_state STREAMING) + retry/regenerate/continue action dispatch → assert the console-run worker is NOT spawned and the running worker is not cancelled (reproduces the cancel-before-gate)\n2. Fix: mirror the submit gate — is_send_allowed check + already-running notify before the three run_worker dispatches in handle_console_message_action\n3. Regression: actions still work when idle\n4. Sweep + guard tests\n5. Live QA on the vision rig: retry mid-stream → notify, stream finishes; PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mirrored the submit path's is_send_allowed gate at all three console-run dispatch sites in handle_console_message_action (retry/regenerate/continue) — notify 'A Console run is already running.' and return instead of spawning the exclusive console-run worker that cancelled the in-flight stream at worker-creation time (before console_chat_controller._active_run_rejection could reject the newcomer). RED-first parametrized tests (Tests/UI/test_console_run_gate.py, 6 tests): all three actions mid-run previously spawned the stream-killer (RED pre-fix), now notify-and-return; idle-path regression proves normal dispatch intact (exactly one console-run spawn). Harness: unmounted ChatScreen(_build_test_app()) with instrumented run_worker/notify — no accessor mocks on the gate itself. Live QA on the vision rig reproduced the exact defect scenario: mid-think regenerate click → verbatim toast, in-flight vision run survived and finalized (t=159s, full 150-char DB row), idle regenerate afterwards spawned a real variant. Evidence: Docs/superpowers/qa/console-retry-gate-2026-07/. Sweep 801/69/0, zero existing-test edits. Files: tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_run_gate.py.
<!-- SECTION:NOTES:END -->
