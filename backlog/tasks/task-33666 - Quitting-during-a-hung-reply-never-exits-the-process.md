---
id: TASK-33666
title: Quitting during a hung reply never exits the process
status: Done
assignee:
  - '@claude'
created_date: '2026-10-02 19:30'
updated_date: '2026-10-11 02:10'
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
- Ruling: no User Guide change — console.md already says Stop "ends the wait", and no page states how long a quit takes.

**Tests.** `Tests/Chat/test_console_stop_silent_provider.py::test_stop_during_a_silent_provider_releases_the_run_thread` reproduces the hang at the bridge (AC#3): a gateway that accepts the request and never streams, a 120 s stall window and a 300 s first-token window, and the run thread must be back within 15 s of the cancel flag, with the run and the reply settled as cancelled and stopped and the provider request closed. Before the fix it fails with "run_reply stayed blocked on a silent provider after the cancel flag was set". `Tests/Chat/test_stream_stall_watchdog.py` gains seven nodes for the stop path: a stop ends a silent stream with the watchdog on, off (`None`) and disabled (`0`); items several polls apart are neither dropped nor reordered; the stall still trips between polls; a task cancel still propagates and closes the source; and a stop that lands on the tick the window expires ends as a stop and raises nothing into the loop.

`Tests/Chat/test_stream_stall_watchdog.py::test_non_finite_config_falls_back_to_default` fails on 10fead87cb and on the fix alike (RecoveryRequired, raw_source_selection_changed): it reads config with no profile marker.

Files: `tldw_chatbook/Chat/stream_stall_watchdog.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `Tests/Chat/test_stream_stall_watchdog.py`, `Tests/Chat/test_console_stop_silent_provider.py`, `backlog/docs/lessons-live-verification.md`, and TASK-30015's notes.
<!-- SECTION:NOTES:END -->
