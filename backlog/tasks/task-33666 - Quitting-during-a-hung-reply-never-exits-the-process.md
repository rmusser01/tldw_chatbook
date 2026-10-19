---
id: TASK-33666
title: Quitting during a hung reply never exits the process
status: Done
assignee:
  - '@claude'
created_date: '2026-10-02 19:30'
updated_date: '2026-10-11 17:27'
labels:
  - console
  - lifecycle
  - bug
dependencies: []
references:
  - qa/task-33662-recovery-relaunch/README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: found during TASK-33662's live check on 2026-10-02; the bug predates that change.

With a reply hung, Ctrl+Q → Quit settles the reply as `stopped`, but the process stays alive after the app unmounts:
- on the TASK-33662 build, for more than 2 minutes, even after the stub provider was stopped;
- on the merge-base build, still alive after 40 s.

Users must kill the terminal to get out. Capture q1 in qa/task-33662-recovery-relaunch/ shows the quit dialog during a live run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a reply hung (a provider that accepts the request and never streams), Ctrl+Q → Quit exits the process within a bounded time (state the bound) and the reply is settled as stopped.
- [x] #2 The cause, meaning which thread or task keeps the process alive, is named in the notes with evidence.
- [x] #3 A test reproduces the hang and passes after the fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on a scratch profile against a stub provider that accepts the request and never answers; dump every thread after Quit and name what the exit is joining.
2. Make the model adapter's wait for the next chunk read the run's cancel flag while no chunk arrives.
3. Tests: the bridge releases its run thread on a silent provider; the watchdog's stop path keeps the stall and cancel contracts.
4. Measure the quit again, on the base commit and on the fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A Stop or a quit now ends a reply's wait on a silent provider within half a second, so the thread the exit was joining comes back.

**What kept the process alive (AC#2).** One thread: the app loop's default-executor worker (`asyncio_N`) that runs the reply, `ConsoleChatController._run_owned_chat_db_operation` → `ConsoleAgentBridge.run_reply` → `AgentService.run_turn` → `run_agent_loop` → `_StreamingModelAdapter._chat_call_impl`, blocked in `future.result()` for the provider turn. That turn's coroutine sat in `watch_content_stalls` waiting for the first chunk, and the adapter read the cancel flag only after a chunk arrived. Quit settled the reply and raised the flag; nothing read it. The exit then waited on that worker: `asyncio.run` → `Runner.close()` → `loop.shutdown_default_executor()` → a helper thread in `ThreadPoolExecutor.shutdown(wait=True)` → `join`.

Evidence: `faulthandler` dumps of every thread, 8 s and 30 s after confirming Quit, on 10fead87cb with a scratch profile and a llama.cpp-shaped stub that accepts the POST and sends nothing. Both dumps show the same three frames: the main thread in `asyncio/runners.py:72 close` → `run_until_complete`; `Thread-2 (_do_shutdown)` in `concurrent/futures/thread.py:239 shutdown` → `join`; and the worker in `console_agent_bridge.py:3918 _chat_call_impl` under the call chain above. The model loop (`console-agent-loop`) was idle in `select`.

The process did end, at 120.3 s with status 1. The exit watchdog (`Utils/app_shutdown.py`, `[general] shutdown_grace_seconds`, 120 s) printed "Shutdown did not finish within the grace period (app shutdown); exiting hard. Threads still alive: ['Thread-2 (_do_shutdown)', 'asyncio_5', 'chatbook-storage-admission(daemon)', 'console-agent-loop(daemon)', 'theme-launch-default_0', 'tldw-exit-watchdog(daemon)']". So on this branch "never exits" means "holds the terminal for two minutes, then a hard exit".

TASK-33628.5's exit wait is not it. That wait (`console_durable_writes.end_store_after_writes`, bounded at 5 s) runs inside the app's unmount; in both dumps the main thread is already in `Runner.close()`, past the unmount.

**The fix.** `watch_content_stalls` takes the consumer's cancel predicate (`should_stop`). While no item arrives it reads the predicate every `STOP_POLL_SECONDS` (0.5 s) and, once true, ends the read through the same `asyncio.timeout` cancellation the stall window uses. The stream then ends as if the source had, and the source is closed. `_StreamingModelAdapter` passes the run's cancel predicate; the run loop's own check then settles the run as cancelled.

**The bound (AC#1).** The run thread is released within one poll (0.5 s) of the cancel flag, plus the model loop's bounded join (5 s, `_LOOP_THREAD_JOIN_SECONDS`). Measured from confirming Quit to process exit, at 211x44 on the scratch profile, with the machine at a load average above 30: 1.8 s (no response headers) and 2.2 s (headers, then silence), both status 0. The reply's row reads `assistant_generation_state = 'stopped'` and `console_dispatch_checkpoints` is empty. The watchdog's 120 s stays the hard limit.

**Limit: providers on the threaded transport.** Only `llama_cpp` streams over the async transport. Every other provider goes through `ConsoleProviderGateway._stream_generic_chat`, which runs the synchronous `chat_api_call` on a worker of the model loop's own executor. Their run thread is released the same way, but that worker stays inside `requests`' socket read, and `concurrent.futures.thread._python_exit` joins it at interpreter exit. Measured on this fix with vLLM against the same stub: exit after 120.4 s, status 1, "Threads still alive: ['asyncio_0(daemon)', 'chatbook-storage-admission(daemon)', 'tldw-exit-watchdog(daemon)']", both with no response headers and with headers followed by silence; the reply was still settled as stopped. For those providers the bound is the provider's HTTP read timeout or the watchdog's 120 s, whichever comes first, as before. Aborting that read is TASK-30015, which names a stop as well as a stall; the dump and timing are in its notes.

**Rulings.**
- Ruling: the adapter passes its plain cancel predicate, not `stream_cut` — a redirect (TASK-28227) is not a cancel, and this rider is about Stop and quit — cost if wrong: a redirect sent during a silent first-token wait still takes effect at the first chunk, as it did before.
- Ruling: the threaded transport's blocked read is left to TASK-30015 — ending it needs the bridge to hold the response's socket and close it (that task's AC#2), which changes how the gateway owns its provider worker — cost if wrong: quitting during a silent non-llama.cpp provider still holds the terminal for up to 120 s and ends with status 1.
- Ruling: the User Guide states the limit instead of staying silent — console.md, "Quirks & troubleshooting", gains "Quitting while a reply waits can hold the terminal": a few seconds with llama.cpp; with any other provider, the connection's timeout or 120 s, then the watchdog's message and status 1 — a two-minute hold that ends in an error line reads as a crash unless the guide names it — cost if wrong: one bullet to delete. TASK-30015's notes say to update it when that task lands.

**Controller ruling, 2026-10-11 (Phase 9 ledger, R12): this rider closes with both bounds.** AC#1 asks for a bounded time and for the bound to be stated. Both are stated, here and in the User Guide: about 2 s with llama.cpp, the path this rider reproduced, and the provider's read timeout or the watchdog's 120 s (exit status 1) on the threaded transport. The tick stands. The gateway is not changed in this task; aborting that read stays with TASK-30015 (To Do, outside this program), whose AC#1 and AC#2 are that abort. The 120 s hold is Phase 9's known residual, and the controller reports it to the owner.

**Checked again at the commit (2026-10-11, at 2748183de7).**
- Failing first: this rider's two test files and TASK-33667's, run against 10fead87cb's source, give 3 failed, 23 passed, 7 errors. `test_stop_during_a_silent_provider_releases_the_run_thread` fails after 15.5 s with "run_reply stayed blocked on a silent provider after the cancel flag was set", and the seven new watchdog nodes error at setup because `STOP_POLL_SECONDS` does not exist. The other two failures are TASK-33667's new test and `test_non_finite_config_falls_back_to_default`, named below.
- Live, same scratch profile and stub, 211x44: `llama_cpp` with no response headers exited 1.3 s after Quit was confirmed, status 0; vLLM with headers followed by silence exited after 120.4 s, status 1, with the watchdog's line naming `asyncio_0(daemon)`. After both runs the reply's row reads `assistant_generation_state = 'stopped'` and `console_dispatch_checkpoints` has no row.

**Tests.** `Tests/Chat/test_console_stop_silent_provider.py::test_stop_during_a_silent_provider_releases_the_run_thread` reproduces the hang at the bridge (AC#3): a gateway that accepts the request and never streams, a 120 s stall window and a 300 s first-token window, and the run thread must be back within 15 s of the cancel flag, with the run and the reply settled as cancelled and stopped and the provider request closed. Before the fix it fails with "run_reply stayed blocked on a silent provider after the cancel flag was set". `Tests/Chat/test_stream_stall_watchdog.py` gains seven nodes for the stop path: a stop ends a silent stream with the watchdog on, off (`None`) and disabled (`0`); items several polls apart are neither dropped nor reordered; the stall still trips between polls; a task cancel still propagates and closes the source; and a stop that lands on the tick the window expires ends as a stop and raises nothing into the loop.

`Tests/Chat/test_stream_stall_watchdog.py::test_non_finite_config_falls_back_to_default` fails on 10fead87cb and on the fix alike (RecoveryRequired, raw_source_selection_changed): it reads config with no profile marker.

**Fix round 1 (review of c6af8a5792).** Three points in the stop path, none of which changes the bound.
- The stop poll left a reference cycle for every item. `poll` reschedules itself by name, so the function sat in its own closure cell, and each awaited item left that cycle to the cyclic collector, on the per-token path of every Console reply. `poll` now receives itself as an argument (`loop.call_later(STOP_POLL_SECONDS, poll, poll)`) and never names itself, so there is no cell to empty, and cancelling the handle drops its callback and arguments. Fix round 1 emptied the cell with `del poll` instead; fix round 2 replaced that, because pyflakes then reported the nested `poll` as an undefined name (F821), and declared `poll_handle` ahead of the function, which clears the module's one mypy error. Measured with a throwaway script, Python 3.12.11, collector off: 0 unreachable objects after 5,000 items with no poll firing, and 0 after 300 items with 843 polls firing between them; fix round 1 measured 0 on the `asyncio.wait_for` path. `test_the_stop_poll_leaves_nothing_for_the_cyclic_collector` keeps it: 200 items, fewer than 200 unreachable objects. Before the change it found 1,810.
- A Stop on a silent provider no longer clears the session's stall streak. A stream ended by the stop poll returns normally, so it reached `reset_session_stalls` under the comment "A productive turn", although nothing had arrived; before this rider the same Stop ran out the window and was recorded as a stall. `_chat_call_impl` now resets only when the turn received at least one item. The streak feeds one thing, the `repeated_this_session` field of the stall warning. `test_stop_during_a_silent_provider_releases_the_run_thread` records a stall before the run and asserts it is still counted after; before the change that assert failed with `0 == 1`. `test_a_reply_that_streamed_clears_the_stall_streak` holds the other arm.
- No test stopped a stream that had produced content and then gone silent, where the poll runs on the shorter between-items window. `test_stop_ends_a_stream_that_went_silent_after_content`: the source yields two items and then waits, and the stop predicate turns true after three polls. The result is exactly those two items with no `StreamStallError`, and the source is cancelled once and closed. It passes before and after this round; it closes a gap in the tests, not a defect.

Files: `tldw_chatbook/Chat/stream_stall_watchdog.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `Tests/Chat/test_stream_stall_watchdog.py`, `Tests/Chat/test_console_stop_silent_provider.py`, `backlog/docs/lessons-live-verification.md`, `Docs/User_Guide/console.md`, and TASK-30015's notes.
Lanes (whole-PR review of Phase 9, round 1, 2026-10-18): `Tests/Chat/test_console_stop_silent_provider.py` (both nodes, bootstrap_profile) joins the PR Fast Lane's admission-sensitive step whole (`pr-fast-lane-admission`, `ADMISSION_TARGETS`); the step's 36 added targets passed 3 of 3 serial in the lane's install set (358 passed each), and the whole step in order passed (929 passed, 2 skipped, 1 xfailed). The seven stop-path nodes of `Tests/Chat/test_stream_stall_watchdog.py` are sandboxed and pure, so they run in the PR Fast Lane's sandboxed step (`pr-fast-lane`, `FAST_LANE_TARGETS`). In that job's install set (serial, `--timeout=180`) the 11 targets this round adds passed 3 of 3 (44 passed each, 24-53 s). They go in as node ids, not the file: in the whole sandboxed step in order the file's older `test_non_finite_config_falls_back_to_default` refuses with `RecoveryRequired: raw_source_selection_changed` (it reads guarded config), and the other 25 cases pass.
<!-- SECTION:NOTES:END -->
