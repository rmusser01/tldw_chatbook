---
id: TASK-35660
title: >-
  Console poll pilot test drops its Send press while the provider switch's
  control-bar sync is deferred
status: Done
assignee:
  - '@claude'
created_date: '2026-10-10 17:58'
updated_date: '2026-10-19 13:40'
labels:
  - console
  - tests
  - ci
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI Fast Lane shard 3 failed test_console_poll_outlives_a_turn_the_controller_has_not_started twice on 2026-10-10 (run 38064483856 job 114249247572 on PR #3062; run 38068440232 job 114260833572 on another branch) with TimeoutError at the wait for the stalled submit_draft. That required check evicts PRs from the merge queue after two reds.

The send was not slow: neither failure logs the ChatScreen Send handler (local passing runs do), and no send stage ran. The test lifts the mount-time "Add API key" block with _select_llamacpp_console, whose control-bar sync defers instead of waiting while a worker thread holds the config lock -- the post-mount hook-permission read holds it. Send then stays disabled until the deferred 0.2 s retry, and Button.press() on a disabled button does nothing. A slow runner widens that lock window until it overlaps the switch. Locally the window closed about 100 ms before the switch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test presses Send only once Send is enabled, and fails naming Send's blocked reason if it never is
- [x] #2 A send that does not reach the stalled controller start within a generous bound fails with a message naming the stage it reached, not a bare TimeoutError
- [x] #3 With the transcript poll predicate's live-turn clauses (runtime custody and the controller's in-flight count) removed, the test fails at its first poll decision
- [x] #4 The fixed test passes 10 of 10 runs under background QoS, with 1.5 s and 5 s precapture delays, and with a hook-permission read holding the config lock across the provider switch; the pre-fix test fails under that last condition with the CI's TimeoutError
- [x] #5 A kept regression test fails whenever the shared provider-selection helper returns with Send still blocked while another thread holds the config locks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find where the CI send stopped by comparing the CI captured logs with an instrumented local run. Timestamp each stage from the press to `submit_draft`, and trace the config-lock holders.
2. Reproduce the CI interleaving with a Tests-only harness: a hook-permission read holds the config lock across `_select_llamacpp_console`'s control-bar sync.
3. Press Send only once it is enabled. Wait for the stalled start with a generous bound, and on failure name the stage the send reached.
4. Mutation-check the custody clause. Run 10 times under `taskpolicy -b`, with 1.5 s and 5 s precapture delays, and under the CI interleaving. Compare with the pre-fix test.
5. (2026-10-19, on resume) Rebase onto dev. Drop the uncommitted helper rewrite in favour of dev's TASK-34789 helper. Keep the poll test's diagnostics. Turn the lock-hold harness into a kept regression test of the helper. Re-run the evidence on the rebased head.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Renumbered, and partly superseded by TASK-34789 (2026-10-19).** This task was filed as TASK-34795 on this unpushed branch on 2026-10-10. Work then stopped at a usage limit, leaving an uncommitted rewrite of the shared `_select_llamacpp_console` helper. While it was stopped, dev's TASK-34789 (PR #3091: 6b5d118526, f0f1c591c0, 2026-10-11) diagnosed the same two runs and fixed that helper. `_sync_console_control_bar_now` takes both config locks, which are reentrant, and asserts that the sync rendered; 15 call sites use it. Dev also gave the id TASK-34795 to a notes-sync task, so this task is now TASK-35660. The rebase dropped the uncommitted helper rewrite and kept dev's helper.

**Root cause** (found 2026-10-10). The Send press was dropped. The send was not slow. Pytest prints loguru output under each failed test's "Captured stderr call" section. In both CI failures that section has the test's other `chat_screen` lines, but not `ChatScreen on_button_pressed called with button: console-send-message`, which every local run writes (re-checked from the job logs on 2026-10-19). Before the switch, the composer holds the mount-time block "Add API key in Settings > Providers & Models before sending.". The helper lifts it through `_sync_console_control_bar` -> `run_console_config_sync`. That sync defers to a 0.2 s retry instead of waiting when another thread holds the config locks. The Console's post-mount hook-permission read (`HookPermissions._current` -> `config.locked_hooks_config_snapshot`) holds them on a worker thread; a local trace showed it releasing about 100 ms before the switch. When the two overlap, the sync defers and `load_draft` replays the stale block. `Button.press()` on the disabled Send then returns without posting, so no send starts.

**What this task adds on top of dev** (test-only; no product code changed):
- `test_console_poll_outlives_a_turn_the_controller_has_not_started` asserts that Send is enabled just before it presses, and names Send's tooltip if it is not (AC #1). The shared helper now renders before it returns, so the old wait-then-press helper (`_press_send_once_enabled`) became this assertion. The wait for the stalled start is 30 s; on timeout it fails with the stage the send reached (runtime custody, the acknowledgement, Send's state, the notices), not a bare TimeoutError (AC #2). The three-decision wait has a 20 s deadline instead of an iteration count.
- `test_select_llamacpp_console_enables_send_while_another_thread_holds_the_config_locks` (new, same file) replaces the scratch harness (AC #5). A thread holds both config locks, in `run_console_config_sync`'s order, for up to 1 s across the helper call. The test then asserts that Send takes a press. It checks that the mount-time block is present. If a stalled event loop lets the hold lapse before the helper runs, it takes the hold again (at most three times) instead of failing, so a slow runner cannot turn it red in a required lane.

**Evidence on the rebased head** (dev 4de437a5c5, macOS, plain command unless noted):
- The new test against the helper as it was before TASK-34789 (`console._sync_console_control_bar()` with its result ignored, applied and restored by Edit) fails with `AssertionError: Send is disabled: 'Add API key in Settings > Providers & Models before sending.'`. On dev's helper it passes.
- Other callers, with a scratch pytest plugin that holds both config locks for 1 s across every `_select_llamacpp_console` call. With the pre-TASK-34789 helper, `test_console_resend_click_re_runs_a_failed_turn_in_place` and `test_console_r_resends_a_refused_echo_as_one_message` fail: when the helper returned, Send was disabled with the "Add API key" tooltip, the press was dropped, and the tests failed with `Text not found: 'llama.cpp stream failed'` / `'llama.cpp unavailable'`. With dev's helper both pass, and Send is blocked only by the empty draft ("Type a message to send."). `test_console_native_blocked_send_preserves_composer_text_and_shows_recovery` passes either way: it configures a ready provider before mount, so there is no block to lift. Most other callers could not be used locally. They fail at setup with the environmental `RecoveryRequired: raw_source_selection_changed` (files without the bootstrap-profile marker), or they run in a `private_profile_test` child process that the plugin does not reach.
- The poll test passes under that plugin.
- AC #3. Removing only the custody clause from `_console_transcript_poll_needed` no longer fails the test, on this branch or on clean dev. A probe showed `controller.in_flight_run_count()` already at 1 (with custody True and run status IDLE) throughout the held start, so the in-flight clause keeps the poll alive. Removing both the custody and in-flight clauses fails the test at its first decision (`[False, False, False] == [True, True, True]`). AC #3 is reworded to match. On the pre-rebase tree, removing the custody clause alone failed it (`[False] == [True, True, True]`).
- AC #4 was measured on the pre-rebase tree on 2026-10-10: 10/10 under `taskpolicy -b`, 10/10 with 1.5 s and with 5 s precapture delays, and under the CI interleaving 10/10 with a 1.5 s hold and 5/5 with a 5 s hold. The pre-fix test failed 7/7 and 1/1 with the CI's TimeoutError. On the rebased head: AC4_REBASED.

**Callers that expect Send to stay disabled.** Dev's helper waits on the config locks and the render, never on Send being enabled, so it cannot hang them. This branch does not change the helper.

**Not changed.** Dev's helper takes the locks with a blocking acquire and checks with a bare `assert rendered`. A holder that never released the locks would hang the test until the lane timeout instead of failing with a message. No such holder is known.

**The whole file.** Tests/UI/test_console_turn_resend_ui.py passed 15 of 16 on the rebased head (host load average 70-110 from other work). The red test was test_resend_poll_survives_initial_hook_admission_read, which this task does not touch. It runs in a `private_profile_test` child, and nothing this task adds runs there. Under that load it fails on clean dev too, in the same way: in 1 of 3 runs, "probe incremental reply" was in the store but not yet on screen 0.65 s after the stream began. Its other failure mode on this branch was `pilot.click` on the Resend action returning False. With this branch's test file copied into a clean dev worktree, it passed (one run passed, and one hit the 300 s pytest-timeout at load 90). Not filed: there is no CI evidence of it failing, and the load was extreme.

**Found on the way.** test_stop_renders_stopped_state_even_when_direct_sync_coalesces passes when its stream never starts. test_stop_click_acknowledges_synchronously_and_records_stop_row is red locally on clean dev. Both are filed as TASK-35661.

**Files:** `Tests/UI/test_console_turn_resend_ui.py`, `backlog/docs/lessons-testing-evidence.md` (an addendum to TASK-34789's entry, which already covers this incident's cause), this task, and TASK-35661.
<!-- SECTION:NOTES:END -->
