---
id: TASK-35660
title: >-
  Console poll pilot test drops its Send press while the provider switch's
  control-bar sync is deferred
status: Done
assignee:
  - '@claude'
created_date: '2026-10-10 17:58'
updated_date: '2026-10-20 11:05'
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
- [x] #4 The fixed test passes 10 of 10 runs with 1.5 s and 5 s precapture delays and with another thread holding both config locks across the provider switch; under background QoS it does not fail at the stalled-start wait or in the shared helper, where dev's version fails; the pre-fix test fails under the lock hold with the CI's TimeoutError
- [x] #5 A kept regression test fails whenever the shared provider-selection helper returns with Send still blocked while another thread holds the config locks
- [x] #6 The shared provider-selection helper renders before it returns even when its first warm-up or sync is not current, fails naming the last refusal and Send's state if nothing renders within a bounded time, and still fails at once for a closing or disposed Console
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find where the CI send stopped by comparing the CI captured logs with an instrumented local run. Timestamp each stage from the press to `submit_draft`, and trace the config-lock holders.
2. Reproduce the CI interleaving with a Tests-only harness: a hook-permission read holds the config lock across `_select_llamacpp_console`'s control-bar sync.
3. Press Send only once it is enabled. Wait for the stalled start with a generous bound, and on failure name the stage the send reached.
4. Mutation-check the custody clause. Run 10 times under `taskpolicy -b`, with 1.5 s and 5 s precapture delays, and under the CI interleaving. Compare with the pre-fix test.
5. (2026-10-19, on resume) Rebase onto dev. Drop the uncommitted helper rewrite in favour of dev's TASK-34789 helper. Keep the poll test's diagnostics. Turn the lock-hold harness into a kept regression test of the helper. Re-run the evidence on the rebased head.
6. (2026-10-19, second resume) Rebase onto ce4aa4a231 and re-run the evidence there. Background-QoS runs failed inside dev's helper at its one-shot asserts: log why each sync deferred, replay each order deterministically, and make the helper retry within a bound and name the reason.
7. (2026-10-19, third resume) Rebase onto 8136af6f6f, then 0fef489ece. Re-run the red/green, both changed test files on the branch and on dev, and repeats of the three tests on the branch. File anything the whole-file runs surface.
8. (2026-10-20, fourth resume) Check the interrupted session's uncommitted review pass against the committed helper and by mutation, keep what holds, and re-run the changed files and a sample of the helper's callers on the branch and on dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Renumbered, and partly superseded by TASK-34789.** This task was filed as TASK-34795 on this unpushed branch on 2026-10-10. Work then stopped at a usage limit, leaving an uncommitted rewrite of the shared `_select_llamacpp_console` helper. While it was stopped, dev's TASK-34789 (PR #3091: 6b5d118526, f0f1c591c0, 2026-10-11) diagnosed the same two runs and fixed that helper: `_sync_console_control_bar_now` takes both config locks, which are reentrant, and asserts that the sync rendered. Dev also gave the id TASK-34795 to a notes-sync task, so this task is now TASK-35660. The first resume dropped the uncommitted rewrite and kept dev's helper. The second found that helper still flaky under load and made it retry (below).

**Root cause of the CI failures** (found 2026-10-10). The Send press was dropped; the send was not slow. Pytest prints loguru output under each failed test's "Captured stderr call" section. In both CI failures that section has the test's other `chat_screen` lines (6 and 5), but not `ChatScreen on_button_pressed called with button: console-send-message`, which every local run writes. This was re-read from both jobs' own logs on 2026-10-19 with `gh api .../actions/jobs/<id>/logs`; `gh run view --job 114249247572 --log` returned the passing re-run's log instead. Before the switch, the composer holds the mount-time block "Add API key in Settings > Providers & Models before sending.". The helper lifted it through a control-bar sync that defers to a 0.2 s retry when another thread holds the config locks. The Console's post-mount hook-permission read holds them on a worker thread; a local trace showed it releasing about 100 ms before the switch. When the two overlap, `Button.press()` on the still-disabled Send returns without posting.

**What this task adds** (Tests only; no product code changed):
- The poll pilot asserts that Send is enabled just before it presses, naming Send's tooltip if not (AC #1). On timeout, its wait for the stalled start fails with the stage the send reached (runtime custody, acknowledgement, Send's state, notices), not a bare TimeoutError (AC #2). That wait and the three-decision wait are both bounded by `_LOADED_HOST_WAIT_S` = 60 s, and each ends as soon as its condition holds. The first pass used 30 s and 20 s, and both failed at load ~100 (see AC #4). The 20 s deadline had replaced dev's `range(100)` loop of `pilot.pause(0.05)`; those pauses stretch with load and a wall-clock deadline does not.
- `test_select_llamacpp_console_enables_send_while_another_thread_holds_the_config_locks` (AC #5): another thread holds both config locks when the helper is called, and Send must take a press once it returns. It pins that outcome, not the helper's lock wrap. The helper's readiness warm-up now waits the hold out (0.987 s of a 1 s hold, measured), so a helper without the wrap still passes. The helper as it was before TASK-34789 fails.
- `_sync_console_control_bar_now` retries (AC #6). Under `taskpolicy -b`, dev's helper failed with a bare `assert False` at both of its one-shot asserts:
  - `assert rendered` (line 4881): the sync deferred although the helper held both locks.
  - `assert await projection.warm()` (line 4878): the second warm-up was not current either. A probe that logged each deferred sync showed that run's projection key going stale with a refresh pending.

  The helper now warms and syncs again until the sync renders, for up to 15 s. It then fails naming the last refusal and Send's state. A closing screen or a disposed runtime still fails at once; `test_control_bar_helper_still_refuses_an_unavailable_owner` is unchanged and passes. Two deterministic tests replay the orders, and each fails on dev's helper at the matching line:
  - `test_select_llamacpp_console_renders_after_its_readiness_proof_goes_stale`: the warm-up returns current, the loop resumes the helper 1.1 s later, and a display has started the next refresh, so the one-second proof has gone stale.
  - `test_control_bar_helper_warms_again_while_each_read_goes_stale`: the settings change during the first two reads.

**Evidence** (dev ce4aa4a231, macOS, plain command unless noted, host load average 60-105 from other sessions throughout. The final test code is 2d74b40856. `e9e0089441` below has the final helper with the 30 s/20 s bounds; 78b38726d1 has the final poll test. These are this branch's commits before its rebase onto 0fef489ece and were never pushed. After the rebase, 00bc6ddab3 is 80da38d3ff, 78b38726d1 is 59433ceb4c and 2d74b40856 is 8fb07279f0. `e9e0089441` was an earlier form of 7a85f1c445 with the same helper code):
- AC #5 test: against the helper as it was before TASK-34789 (Edit-based mutation, restored) it fails with `AssertionError: Send is disabled: 'Add API key in Settings > Providers & Models before sending.'`; on the fixed helper it passes.
- AC #6 tests: each fails on dev's helper with `assert False` (lines 4881 and 4878 respectively) and passes on the fixed one. The stale-proof replay on the final code passed 8/8 and failed 3/3 at line 4881 with dev's helper (both in a scratch tree of ce4aa4a231 holding the branch's test files, with a repeat plugin). Its first version stalled the first warm-up that returned current and recorded the stall only after sleeping, so the context snapshot's concurrent warm-up (from a worker) stalled as well, and one run on the final head failed its own guard with `[True, True] == [True]`. It now stalls only a warm-up called from the test's own task, where the helper runs. With every sync forced to count as deferred, the helper fails after 15 s with "The Console control bar did not render within 15.0 s (121 attempts; last: the control-bar sync deferred). Send disabled=... tooltip=...".
- AC #1 and the CI symptom: with the pre-TASK-34789 helper and another thread holding both config locks for 1.5 s from the helper's call, this branch's poll test fails with `AssertionError: Send is disabled: 'Add API key ...'`. Dev's version fails with the CI's bare `TimeoutError` at its `wait_for(entered.wait(), 10)`.
- AC #2: with the press removed (scratch mutation), the poll test fails at its bound with "the Send never reached the stalled submit_draft: in runtime custody=False, acknowledged=False, Send='Send | $' disabled=False tooltip='Send the active Console session draft.', notices=[]".
- AC #3: removing the custody and in-flight clauses from `_console_transcript_poll_needed` fails the test at its first decision (`[False, False, False] == [True, True, True]`). Removing only the custody clause does not: `in_flight_run_count()` is already 1 throughout the held start (measured at the first resume).
- AC #4, 10 runs per condition, using a scratch pytest plugin (deleted):
  - 1.5 s precapture delay: 10/10 on e9e0089441, and 10/10 on 00bc6ddab3 (dev's helper).
  - 1.5 s lock hold from the helper's call: 10/10 on e9e0089441 and on 00bc6ddab3. In all 10 final-helper runs the hold had ended before the helper returned.
  - 5 s precapture delay: 10/10 on 78b38726d1. Before that, the 30 s/20 s bounds failed it at load ~100: 6/10 on e9e0089441 (2 at the start bound, 1 at the decisions bound, and 1 Textual `WaitForScreenTimeout` in the last wait). Paired 6-run arms gave 4/6 with dev's helper and 5/6 with the final one, each with one red at the 20 s decisions bound, and the helper rendered on its first sync in all 12. The final-head batch ran at lower load (30-50 s runs instead of 50-100 s), so it stressed the 60 s bounds less.
  - Background QoS, same hour: the final helper (e9e0089441) passed 7/10. All 3 reds were Textual `WaitForScreenTimeout` before the press (2 inside `run_test`, 1 in the composer mount wait). Dev passed 1/10: 6 reds at its 10 s stalled-start wait (the CI's TimeoutError) and 3 at the helper's `assert rendered`. The AC #5 test passed 10/10 under background QoS.
  - Pre-fix test under the lock hold: the CI's TimeoutError (the AC #1 run above).
  - History: on 2026-10-10 the pre-rebase tree passed 10/10 under `taskpolicy -b`, 10/10 with each delay, and 10/10 and 5/5 under 1.5 s and 5 s holds; the pre-fix test failed 7/7 and 1/1.
  - AC #4 was reworded on 2026-10-19. It had asked for 10/10 under background QoS. At this host load, app-booting pilots also fail before or after their bodies, on dev as well: `HelperTimeoutError` starting a private SQLite helper while the test app is built, and Textual's `WaitForScreenTimeout` inside `run_test`. A 10/10 count there measures the host, not the test.
- Other callers of the helper:
  - Under a 1.5 s lock hold with the pre-TASK-34789 helper, `test_console_resend_click_re_runs_a_failed_turn_in_place` and `test_console_r_resends_a_refused_echo_as_one_message` fail (press dropped: "Text not found: 'llama.cpp stream failed'" / "'llama.cpp unavailable'"). With dev's helper and with the fixed one they pass.
  - The 12 tests in six files that call `_sync_console_control_bar_now` directly (14 items), run on both trees at normal priority. On both, 12 items fail at setup with the environmental `RecoveryRequired: raw_source_selection_changed`, and `test_console_workbench_send_action_disables_during_active_run` passes. `test_console_active_stream_sync_skips_unchanged_chrome_and_inspector` failed on dev in the helper at line 4878 and passed on the branch.
- Whole files, one run each under load ~100:
  - Tests/UI/test_console_turn_resend_ui.py: branch (e9e0089441) 13 of 17, dev 12 of 15.
    - Red on both: `test_console_resend_click_re_runs_a_failed_turn_in_place` and `test_resend_poll_survives_initial_hook_admission_read`.
    - Dev only: `test_transcript_resend_retains_failed_rows_until_connection_is_ready` (`pilot.click` on Resend returned False).
    - Branch only: `test_console_poll_paints_again_after_work_that_ended_during_a_paint` and `test_console_r_resends_a_refused_echo_as_one_message`.
    - Every red was a UI wait timing out (Textual `WaitForScreenTimeout`, a selector wait, or a private-profile child's TimeoutError) in code this branch does not change. The poll test and both new tests in this file passed.
  - Tests/UI/test_console_readiness_config_projection.py: 27 of 28 on the branch, 26 of 27 on dev. The one red on both, `test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation`, is filed as TASK-35662.

**Rebased onto dev 8136af6f6f, then 0fef489ece (2026-10-19, third resume).** Neither rebase had conflicts. Dev's 37 new commits do not touch the three test files. Near this code they change `console_transcript.py` (TASK-34982's poll-stop fix) and, in 0fef489ece, the full UI sync's rail-visibility and guidance reads (TASK-35444). They do not change the control-bar sync the helper calls. Results (macOS, plain command, host load 43-87 from other sessions; code head 8fb07279f0 on 0fef489ece):
- Red/green, re-run on both dev commits: the branch's two test files were copied into a detached worktree of dev, so they ran against dev's helper. On each, `test_control_bar_helper_warms_again_while_each_read_goes_stale` fails at line 4878 (`assert await projection.warm()`), and `test_select_llamacpp_console_renders_after_its_readiness_proof_goes_stale` fails at line 4881 (`assert rendered`). Both pass on the branch.
- Tests/UI/test_console_readiness_config_projection.py, on both bases: 27 of 28 on the branch, 26 of 27 on dev. The one red on both is TASK-35662's.
- Tests/UI/test_console_turn_resend_ui.py, one whole-file run per tree on each base:
  - On 0fef489ece: branch 17 of 17 (load 57-63), then dev 14 of 15. Dev's red, `test_transcript_resend_retains_failed_rows_until_connection_is_ready`, timed out waiting for the Resend button. This branch does not change that test, and it was a dev-only red before the rebase as well.
  - On 8136af6f6f: branch 16 of 17 (load 43-52), then dev 15 of 15. The branch's red, `test_resend_poll_survives_initial_hook_admission_read`, failed after its helper call and press had succeeded. It waits a fixed `pilot.pause(0.65)` and then asserts the streamed reply is painted; the reply was in the store, but the screen still showed the run as Preparing. Run alone, alternating between trees, it passed 4 of 4 on each (load 47-58). It was red on both trees before the rebase. Filed as TASK-35743.
- Repeats on the branch, as separate pytest runs of the poll pilot, the AC #5 lock-hold test and the stale-proof replay: 15 of 15 on 8136af6f6f (5 each, load 62-87) and 9 of 9 on 0fef489ece (3 each, load 49-65). Each run took 22-61 s, including pytest startup.
- `./scripts/preflight.sh` passes on the rebased head.

**Callers that expect Send to stay disabled.** The helper waits for its own sync to render, never for Send to be enabled. So it cannot hang a caller whose Send legitimately stays blocked, for example by an empty draft.

**CI cost.** The two new tests in Tests/UI/test_console_turn_resend_ui.py, which is already in the UI PR gate census, each boot the app. In the whole-file run at load ~100 they took 45 s and 35 s, plus 6-14 s teardown; the file's other pilots took 45-166 s in that run. No CI timing for them exists yet.

**Found on the way.**
- `test_stop_renders_stopped_state_even_when_direct_sync_coalesces` passes when its stream never starts, and `test_stop_click_acknowledges_synchronously_and_records_stop_row` is red locally on clean dev: TASK-35661.
- `test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation` is red on dev because its stub screen has no rate reading: TASK-35662.
- `test_resend_poll_survives_initial_hook_admission_read` asserts a streamed paint after a fixed 0.65 s pause: TASK-35743.

**Files:** `Tests/UI/test_console_turn_resend_ui.py`, `Tests/UI/test_console_native_chat_flow.py` (the shared helper), `Tests/UI/test_console_readiness_config_projection.py`, `backlog/docs/lessons-testing-evidence.md` (addenda to TASK-34789's entry), this task, TASK-35661, TASK-35662, TASK-35743 and TASK-35850.

**Review pass (2026-10-20, fourth resume; on 0fef489ece, then rebased onto 250159834c).** The interrupted session had left an uncommitted review pass on the three test files. Each change was checked against the committed helper (26be55f1f9) in a scratch tree and kept; one assertion was tightened. Tests only:
- **The helper's bound now covers its warm-up.** Before, a warm-up that never settled made the helper wait until pytest's own timeout, outside its 15 s bound. The warm-up now runs under `asyncio.timeout` for the time left. `asyncio.timeout` is used rather than `wait_for` so the warm-up stays in the helper's task, which the stale-proof replay's task check relies on. The helper also resolves the projection on each attempt. The new `test_control_bar_helper_bounds_a_warm_up_that_never_settles` sets the bound to 0.3 s and blocks the read. It fails on 26be55f1f9's helper with the test's own 5 s `TimeoutError`, and passes now with "last: the readiness warm-up did not settle".
- **The fast refusals are pinned by message.** `test_control_bar_helper_still_refuses_an_unavailable_owner` now matches "the Console is closing". The new `test_control_bar_helper_refuses_a_disposed_runtime_at_once` matches "the Console runtime is disposed". With both refusals deleted from the helper (scratch mutation), 26be55f1f9's closing test still passed after 15.0 s, because the retry bound's own AssertionError satisfied its bare `pytest.raises(AssertionError)`. Both new forms fail under that mutation.
- **The stale-proof replay checks the helper's own syncs.** It records the syncs made from the test's task, snapshots them when the helper returns, and requires a deferral followed by a render. The snapshot is this pass's tightening. If a display refreshed during a stall and left nothing pending, the replay stalls again, up to three times. Scratch mutation: the helper returns after one deferred sync. Under it, 26be55f1f9's replay passed, because the product's delayed control-bar replay had enabled Send during the 1.1 s stall. The final replay fails with `[False]`.
- On dev's helper (the branch's projection test file copied into a detached tree of 0fef489ece), both refusal tests fail on the regex (input `'assert False'`), and `test_control_bar_helper_warms_again_while_each_read_goes_stale` fails at line 4878. The bound test fails at setup, because dev has no `_CONTROL_BAR_RENDER_TIMEOUT_S`. Only the 26be55f1f9 comparison is meaningful for that test.
- Whole files (macOS, plain command, one run per tree, host load 34-72 from other sessions):
  - Tests/UI/test_console_readiness_config_projection.py: branch 29 of 30, dev 26 of 27. The one red on both is TASK-35662's.
  - Tests/UI/test_console_turn_resend_ui.py: branch 17 of 17 (250 s), dev 15 of 15 (193 s).
  - Tests/UI/test_console_send_disabled_state.py (one helper caller): branch 9 of 10, dev 8 of 10. `test_enter_hotkey_still_sends_when_send_is_enabled` is red on both. It does not call the helper, and TASK-32819 had listed it as a baseline red; filed as TASK-35850. On dev only, the helper caller `test_enter_hotkey_queues_draft_behind_accepted_run` failed at its 4 s wait for the first streamed chunk. Its press had reached Send's handler, so this was not a dropped press.
  - The four tests in Tests/UI/test_console_internals_decomposition.py that call `_sync_console_control_bar_now` directly fail at setup on both trees with the environmental `RecoveryRequired: raw_source_selection_changed`.
- **Rebased onto dev 250159834c** (PR #3177, tab-click handling) with no conflicts. Dev's change does not touch the three test files or the control-bar sync. On the rebased head: Tests/UI/test_console_readiness_config_projection.py 29 of 30 (TASK-35662's red), Tests/UI/test_console_turn_resend_ui.py 17 of 17 (276 s, load 60-70), and the bound test still fails at its 5 s guard on the rebased pre-pass helper (464434106a). `./scripts/preflight.sh` passes on the rebased head.
<!-- SECTION:NOTES:END -->
