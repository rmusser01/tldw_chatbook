---
id: TASK-38462
title: Keep Windows timer sleeps from starving the default thread pool
status: Done
assignee:
- '@claude'
updated_date: '2026-11-03 15:10'
created_date: 2026-10-31 11:30
labels:
- windows
- performance
priority: medium
references:
- backlog/tasks/task-38206 - GGUF-source-mode-recompose-test-times-out-on-Windows-CI-intermittently.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On Windows, Textual runs every timer sleep, and every timer cancellation, as a blocking job on the event loop's default executor (textual/_win_sleep.py). The app's default executor has min(32, cores + 4) threads, so 8 on a 4-vCPU machine. TASK-38206's thread dumps from two windows-latest CI jobs show all 8 threads parked in `_win_sleep.wait_inner`, with 27–48 timer sleeps queued, including three 24-hour timers that hold 3 threads permanently. Thread workers (`run_worker(..., thread=True)`) and timer cancellations queue behind the sleeping timers, so a screen recompose, which stops many timers, or an inventory read can stall for tens of seconds. This is what made the GGUF source-mode tests time out on Windows CI. It also affects the shipped app on low-core Windows machines, not only tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On Windows, long-lived timers and thread workers no longer compete for the same bounded executor, so a recompose or thread worker does not wait behind sleeping timers (shown with the TASK-38206 8-thread proxy or a Windows run).
- [x] #2 The three 24-hour timers no longer each hold a default-executor thread for their whole duration on Windows.
- [x] #3 macOS and Linux timer and worker behavior is unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the seam: list every Textual 8.2.8 importer of `textual._time.sleep` and `textual._win_sleep` (only `textual/timer.py` binds `sleep`; only `_time.py` imports `_win_sleep`).
2. Recreate the TASK-38206 8-thread proxy as a scratch, uncommitted plugin under `.worktrees/burn-scratch/`. Take a census of timer wait lengths and executor queueing, plus baseline timings for the recompose test and the two keyboard tests.
3. Write failing unit tests first (simulated win32, runnable on macOS): long waits never touch the default executor; short waits run on a dedicated pool; cancellation returns at once and frees the pool; non-win32 is untouched. Also add a guard test that pins the Textual internals the fix replaces.
4. Implement in `Utils/text_selection_crash_guard.py`, where `ThreadWorkerContextGuard.on_load` already owns the default executor. On win32 only, replace `textual.timer.sleep`: `asyncio.sleep` for coarse waits, and a dedicated pool for short high-resolution waits.
5. Rerun the proxy timings with the fix. Run the touched and startup-heavy suites, ruff and preflight, then write the notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**What changed.** On win32 only, `ThreadWorkerContextGuard.on_load` calls `install_windows_timer_sleep()`, which replaces `textual.timer.sleep`, the global that `Timer._run` awaits. `Load` is the app's first event on its loop, before compose and mount start any timer.
- Timer waits of 0.25 s or more use `asyncio.sleep` and hold no thread. This covers the 24 h, 120 s, 65 s, 5 s and 3 s timers and every 1 s or 0.5 s interval.
- Shorter waits keep a high-resolution wait: `time.sleep` runs on a dedicated `textual-timer` pool (cap 32, threads start lazily). CPython 3.11+ implements `time.sleep` on Windows with the same high-resolution waitable timer that `_win_sleep.wait_inner` uses (checked in 3.12 `timemodule.c`).
- A cancel returns at once and needs no executor job. The thread finishes its sub-0.25 s sleep and goes back to the pool.
- Waits under 0.5 ms still yield once, as Textual's Windows sleep does.
- Other platforms return before touching anything.

**Seam survey.** In Textual 8.2.8, only `textual/timer.py` binds `textual._time.sleep`, and only `_time.py` imports `_win_sleep`. Textual's other sleeps (pilot, app, `_wait`) are already `asyncio.sleep`. So `textual.timer.sleep` is the one seam to replace. A pin test fails when that changes, or when `_win_sleep.py` stops using `run_in_executor(None, ...)`, the upstream fix that makes this code removable. The `# ponytail:` comment names both ceilings.

**Design choices.**
- The suggested direction was to run Textual's own win sleep on a dedicated pool. I dropped it because that function hard-codes `run_in_executor(None, ...)`, so redirecting it would mean patching `asyncio` inside `_win_sleep`.
- The threshold is 0.25 s, not 1 s. A 1 s interval timer's wait is `next_timer - now`, just under 1 s, so a 1 s cut would have kept every 1 s timer on threads: 56 of 163 waits in the recompose census fell in [0.5, 1). At 0.25 s the Windows tick (~15.6 ms) is at most about 6% of a coarse wait.

**Evidence (macOS, worktree claude/bd-38462 on dev 335f984bd2, 2026-11-03).**
- Unit tests, `Tests/App/test_windows_timer_sleep.py` (win32 simulated by patching `sys.platform` around the install only). RED: 7 of 8 failed (no API); only the Textual pin passed. GREEN: 8 passed.
- Mutations: short waits to the default executor (3 tests fail), `>=` to `>` at the threshold (1 fails), the `on_load` call removed (1 fails), the platform gate removed (2 fail).
- Proxy: TASK-38206's 8-thread Windows-sleep proxy, recreated as an uncommitted plugin in `.worktrees/burn-scratch/s1-proxy*`. It is handed to `@private_profile_test` children through `PYTEST_PLUGINS`. Three reps of the recompose test and both keyboard tests (`-n 8`), load average 21-41:

| settle | 8-thread proxy, no fix | proxy + fix | native, no proxy |
|---|---|---|---|
| screen recompose reaches deferred mount | 14.64 / 22.88 / 13.72 s | 3.52 / 3.52 / 1.70 s | 1.47 / 1.27 s |
| llamacpp managed inventory | 11.79 / 10.88 / 10.93 s | 1.22 / 1.12 / 1.04 s | 1.05 / 0.04 s |
| llamafile managed inventory | 11.92 / 11.57 / 16.93 s | 1.12 / 1.32 / 1.11 s | 1.19 / 1.04 s |

- Default-executor census per test. Without the fix: 150-424 timer waits plus 22-27 cancels per test, 28-43 in flight, and thread-worker queue delay up to 8.4-12.8 s. With the fix: zero timer jobs, worker queue delay at most 0.04-0.56 s.
- Dedicated pool in the keyboard test: at most 15 waits ran at once; 3 short waits were cancelled; queue p50 0.4 ms, max 0.2 s (the host-load noise the default executor also shows).
- The keyboard tests' total call time is not the right comparison. It is about 200-250 s natively on this host, and the run without the fix is faster (105-134 s) because starvation throttles the app's own thread-worker polling: about 150 default-executor jobs per test against 4,700-7,800 natively.
- Live smoke: the real app ran in a sandbox (writes to the real profile denied, outbound IP denied, no provider key) with the fix forced on. It booted to the setup wizard in about 30 s and ran for 55 s: 394 short waits on 6 pool threads, 108 waits on `asyncio.sleep`. Ctrl+Q exited with rc=0.
- Regression run: `Tests/App`, `Tests/Performance/test_screen_leaks.py`, `test_boot_worker_census.py`, `test_gc_idle_refreeze.py`, `test_app_import_weight.py`, `Tests/UI/test_app_startup_delegation.py` and `Tests/UI/test_llm_gguf_source_modes.py` gave 389 passed, 55 failed, 1 skipped.
  - 54 of the failures fail identically on a detached origin/dev worktree. 50 are the known local `RecoveryRequired: raw_source_selection_changed`; the rest are a missing `tldw_profile_core.v2_contract`, the construct-time import allowlist, a screen-leak child timeout and a GGUF fence assertion.
  - The 55th, GGUF `test_inventory_completion_cannot_write_through_replaced_window`, failed once in that combined run. In isolation it passed 4 of 4 on this branch and 4 of 4 on dev. It waits a fixed 8 pauses after a recompose.
- `Tests/Architecture` (module and screen size ratchets, diagnostic inventory, serial): 285 passed, 1 skipped. Ruff shows no new findings; the 5 remaining in the guard module are identical on origin/dev.

**AC#3.** On macOS and Linux, `install_windows_timer_sleep` returns before creating the pool or touching Textual (`test_other_platforms_keep_textuals_sleep`), and nothing else in the module changed behaviour.

**Still Windows-only.** Neither the GGUF source evidence job nor any Windows UI job has run on this head yet. They should show (a) no `asyncio_N` thread in `_win_sleep.wait_inner` in the startup-diagnostic dumps, (b) recompose and inventory settles near the macOS numbers, and (c) smooth short-interval animation with the pool's `time.sleep`.

**Files.** `tldw_chatbook/Utils/text_selection_crash_guard.py`, `Tests/App/test_windows_timer_sleep.py`, `Docs/Development/console-optimization-review-list.md` (OPT-78 status) and `backlog/docs/lessons-testing-evidence.md` (an update to the TASK-38206 entry).

**Review round (PR review, 2026-11-03).** Test-only fixes (commit 8b05d8bb22).
- `test_other_platforms_keep_textuals_sleep` saved `textual.timer.sleep` as its starting value. On real Windows a TldwCli booted earlier in the same process leaves `_windows_timer_sleep` installed process-wide, so the darwin and linux cases failed there. The test now starts from `textual._time.sleep`. A new `earlier_boot` parameter installs the Windows sleep first, as that earlier boot would.
- RED on macOS: with the old setup line the `earlier_boot` cases failed 2 of 4. The unmodified old module also failed both cases when `install_windows_timer_sleep()` ran under a patched `win32` platform before it, which is the reviewer's reproduction. With the fix, both runs pass 4 of 4.
- Intervals use `time.perf_counter`. On Windows Python 3.12, `time.monotonic` ticks every 15.625 ms, so `>= 0.025` after a 30 ms sleep could flake. This one cannot be shown as a failing test on macOS.
- `Tests/App/test_windows_timer_sleep.py`: 10 passed.

**Re-review round (2026-11-03).** Commit 55fe92f2f2. `test_other_platforms_keep_textuals_sleep[earlier_boot=True]` installed the Windows sleep, which the test's own reset line then overwrote before the body ran, so it guarded only that reset, and it leaked the pool it created. Off win32 nothing installs that sleep, so the parameter is dropped and the reset keeps a comment saying why it exists; the test creates no executor. `Tests/App/test_windows_timer_sleep.py`: 8 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
