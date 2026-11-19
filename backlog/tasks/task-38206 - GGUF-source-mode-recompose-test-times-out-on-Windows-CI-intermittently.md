---
id: TASK-38206
title: GGUF source-mode recompose test times out on Windows CI intermittently
status: In Progress
created_date: 2026-10-30 10:50
updated_date: '2026-11-03 14:30'
labels:
- ci
- flaky-test
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_llm_gguf_source_modes.py::test_claim_authority_survives_screen_recompose_and_not_window_selection fails intermittently on the "GGUF source evidence - windows-latest" job with "screen recompose did not reach deferred mount": its bounded _settle_pilot_until wait expires before the recomposed LLMManagementWindow reaches _finish_deferred_mount. It failed on PR #3286 in runs 40798969907 (2026-10-29, together with test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp], "llamacpp managed source controls did not set") and 41091204880 (2026-10-30). Neither head changed the LLM screen, and the same tests passed on the macOS job and on neighbouring heads. The wait looks too tight for a loaded Windows runner, or the recompose has a Windows-only slow path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is identified as a wait budget too tight for loaded Windows runners or a Windows-only slow recompose path, with timing evidence from a Windows run.
- [ ] #2 The GGUF source evidence job passes 10 consecutive Windows runs without dropping the deferred-mount assertion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the two failing Windows job logs (runs 40798969907, 41091204880) for timing evidence: per-node durations, startup-diagnostic suspension samples, and the pytest-timeout thread dumps.
2. Trace what the recompose wait depends on (screen recompose -> LabScreen body remount -> LLMManagementWindow.on_mount -> call_after_refresh(_finish_deferred_mount)) and how `_settle_pilot_until` bounds it (already wall clock, 10 s).
3. Reproduce the Windows-only path locally with a scratch, uncommitted proxy of Textual's `_win_sleep` (every timer sleep and every timer cancel holds a default-executor thread) at the runner's 8-thread pool, with a 64-thread control.
4. Give `_settle_pilot_until` a Windows budget that covers the measured starvation while keeping 10 s elsewhere; keep every assertion. The sibling keyboard test shares the helper, so it gets the same budget.
5. Verify under the proxy and under CPU load; run the file, ruff and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Cause: a Windows-only slow path, not CPU speed.** `_settle_pilot_until` already had a wall-clock budget (10 s). On Windows, Textual runs every timer sleep as a blocking job on the loop's default executor (`textual/_win_sleep.py`). Cancelling a timer needs a second job. The app's `FreshContextExecutor` has min(32, 4 vCPU + 4) = 8 threads on windows-latest. The screen recompose removes every widget and stops their timers, so it waits for one of those threads. So does each `thread=True` worker, such as the inventory read that the keyboard test's "managed ... did not settle" waits depend on.

**Windows evidence from the two failing jobs:**
- Calls took 41.9 s and 56.1 s, against 8.4 s on macOS.
- The pytest-timeout thread dumps from the startup-diagnostic step of the same jobs show all 8 `asyncio_0..7` threads parked in `_win_sleep.wait_inner`, with the main thread idle in `GetQueuedCompletionStatus`. The loop was waiting on the executor, not computing.
- The diagnostic's suspension samples show tasks waiting in `_win_sleep.cancel` (line 95, an executor job) under `Timer._stop_all` for about 0.9 s.
- They also show 20+ `_win_sleep.wait` (line 113) calls queued at once, five of them for 47 s.

**Local proxy (scratch plugin, not committed).** It routes `textual.timer.sleep` and `textual._time.sleep` through `run_in_executor`, both the sleep and its cancel, with the app's executor capped at 8.
- The recompose wait took 12.8, 23.6 and 13.3 s, against 0.33 s and one pause natively. With the 10 s budget the test failed 3 of 3 with the CI message.
- The same proxy with 64 threads took 0.96-0.99 s.
- Host CPU load alone (load average 37-41 on 18 cores, other sessions) took 0.6-1.1 s.
- A census of the proxy's in-flight sleeps reached 27-48 against 8 threads. Long sleepers: three 86400 s, one 120 s, two 65 s, two 5 s, one 3 s.
- The keyboard tests' "managed inventory did not settle" waits took 10.2 s (llamacpp) and 15.1 s (llamafile) under the proxy, which matches the 2026-10-29 sibling failure.

**Change.** `Tests/UI/test_llm_gguf_source_modes.py`: `_SETTLE_SECONDS = 60.0` on win32, 10.0 elsewhere. It is used by `_settle_pilot_until`, the helper every wait in the file shares, so the sibling keyboard test is covered too. No assertion was removed or loosened. The 60 s budget stays inside the Windows job's 180 s per-node timeout. macOS and Linux keep their quick 10 s failure inside their 60 s timeout. I also added a lessons entry to `backlog/docs/lessons-testing-evidence.md`.

**Verification (2026-10-31, claude/burn-38206 on dev 94a8a85f03).**
- Under the 8-thread proxy, with the module's own win32 branch selected (reloaded with `sys.platform` set to win32, budget 60.0 confirmed): the recompose test passed 5 of 5, keyboard llamacpp 2 of 2 and keyboard llamafile 2 of 2.
- Plain local run of the whole file: 31 passed. This used `-n 4` and the local-only bootstrap-profile marker aid, because without it 10 plain tests error with `RecoveryRequired: raw_source_selection_changed`, the known local profile issue (TASK-33621.51 lesson).
- ruff: no new findings. The 9 that remain are pre-existing.

**AC#2 is open.** It needs 10 consecutive Windows runs of the GGUF source evidence job, which cannot be triggered from here.

**Follow-up worth filing (not in scope).** The same starvation applies to the shipped app on Windows machines with few cores. Three 24 h timers pin 3 of the 8 threads permanently, and thread workers queue behind sleeping timers. Sizing or splitting the default executor on Windows would address it at the source.

**Review round (2026-10-31, claude/burn-fixB).** The 60 s budget applied per wait, so several waits in one test could sum past the 180 s node timeout and turn an assertion into a pytest-timeout kill. Windows settles now share one deadline per test, 120 s from its setup (autouse fixture), leaving 60 s of the node timeout for teardown; other platforms keep 10 s per wait. No assertion changed. `test_windows_settles_share_one_deadline_per_test` pins it (RED: missing; GREEN); the whole file passes locally (32, bootstrap-profile aid).

**Review round 3.** The budget test compared two real `time.monotonic()` reads 10 ms apart, which is flaky where the clock is coarser than that (Windows 3.12). The deadline now reads an injectable `_settle_clock`, and the test drives it with a fake clock (budgets exactly 120 then 90; spent at the deadline). RED under a frozen clock: `assert 120.0 > 120.0`; GREEN.

**Follow-up finding (2026-11-03, TASK-38703, claude/bd-38703 on dev 335f984bd2).**

From 2026-11-02 the job failed on all three OSes. On Windows it was this test's "screen recompose did not reach deferred mount" again. That was a different cause, not this starvation: a Console refresh loop that dev gained in PR #3166 (e017592b44, merged 2026-10-31 21:32Z). The cost chip and the context estimates fed the single-slot tokenizer gate "OpenAI" and "openai". Each sync then recaptured the request and re-ran the readiness config read, and the read's publication ran another sync: about 15 a second, even under the pushed LLM screen. With the process never idle, every `pilot.pause()` ran to its 1 s cap. Locally the keyboard nodes took 114 s and 193 s instead of 20-33 s. TASK-38703 fixes it by keying the gate on the provider config key; after the fix the recompose node takes 10.7-17.0 s on macOS in 5 of 5 runs.

Three things to know when reading the red Windows runs:
- Every red run before 335f984bd2 ran with the old 10 s budget. This task's 120 s shared deadline reached dev only in that merge (#3352).
- The loop's worker reads (`console-readiness-config`, thread reads through storage admission) also took default-executor threads, the same pool the Windows timer sleeps hold. So on Windows the loop made the starvation worse.
- With the loop gone, the starvation itself is unchanged. TASK-38462 still owns the source fix.

AC#2 still needs 10 consecutive Windows passes in CI, counted from a head that has both this task's deadline and the TASK-38703 fix.

**The 120 s Windows deadline stays for now.** TASK-38462 (claude/bd-38462, e08f7653a2) moves Windows Textual timer sleeps off the default executor. Waits of 0.25 s or more use `asyncio.sleep`, and shorter ones a dedicated pool. If Windows CI then shows the recompose wait back near macOS, under 1 s, the Windows branch can return to the shared 10 s budget. Pinning that needs Windows job timings. The time to decide is AC#2's 10-run streak: lower the budget only if those runs carry TASK-38462 and their recompose waits fit inside 10 s with margin.
**Windows run 1 of 10 (AC#2):** passed in run 41784255481 at d0d26a6fdc (2026-11-04). The 7 full-app nodes took 13.4-23.8 s and the 31 backend tests passed, with no teardown errors. The head carries the TASK-38703 loop fix, the TASK-38462 timer fix and the 120 s Windows deadline. AC#2 needs 9 more consecutive passing Windows runs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
