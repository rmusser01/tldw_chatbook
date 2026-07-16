---
id: TASK-227
title: >-
  Stop task-cancel can race the agent worker finally-reset and finish done
  instead of cancelled
status: Done
assignee:
  - '@claude'
created_date: '2026-07-14 03:33'
updated_date: '2026-07-16 15:39'
labels:
  - console
  - agents
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while testing the stop-before-first-token fix: stop_active_run()'s task cancellation can race _run_agent_reply's finally block resetting _stop_requested, letting a still-running background bridge thread complete its run as 'done' although the user stopped it. Documented in .superpowers/sdd/task-8-report.md (gate fix wave). Narrow window; the store-level no-op fix covers the common case.

Broadened by the Plan-B final whole-branch review (.superpowers/sdd/plan-b-final-review.md) to also cover two related narrow races in the same stop/cancel family:
- LOW-1: ConsoleChatStore.reset_stream_content unconditionally sets a message's status to "streaming", so a disobedient model's post-stop tool-call turn can resurrect an already-stopped message back to "streaming" and leave it stuck there (append_stream_chunk was hardened to no-op on "stopped"; reset_stream_content was not).
- LOW-2: ConsoleChatController._finalize_agent_reply's mark_message_complete/finalize_variant_stream/mark_message_failed calls all raise ValueError via _validate_can_mark_terminal when a Stop lands in the narrow window after asyncio.to_thread returns RUN_DONE but before finalize (the message is already "stopped"); benign in effect (run_state still correctly settles STOPPED) but surfaces a logged exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A stop that lands during an in-flight agent bridge thread always persists the run as cancelled,A regression test covers the cancel-vs-finally race deterministically
- [x] #2 reset_stream_content no-ops (mirroring append_stream_chunk) when the target message is already stopped, with a regression test (LOW-1)
- [x] #3 The post-completion Stop race in _finalize_agent_reply no longer raises or logs an unhandled ValueError, with a regression test (LOW-2)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause each of the 3 races by reading console_chat_controller.py (_run_agent_reply, stop_active_run, close_session, shutdown), console_chat_store.py (reset_stream_content, _validate_can_mark_terminal), and console_agent_bridge.py/agent_runtime.py's should_cancel polling chain.
2. AC1: replace the should_cancel closure's sole dependency on the shared, resettable _stop_requested with a per-run threading.Event captured by value at run start; have stop_active_run/close_session/shutdown set both via a new _signal_stop() helper; the finally block detaches (never clears) the event.
3. AC2: add an early-return no-op in reset_stream_content when message.status == "stopped", mirroring append_stream_chunk's existing hardening.
4. AC3: add an early no-op guard at the top of _finalize_agent_reply that reads the message back and returns a benign STOPPED result if it is already "stopped", before touching outcome.status.
5. Write RED regression tests per AC (real-thread parking + polling for AC1's actual race; direct store/controller calls for AC2/AC3), confirm RED against unfixed code, then implement and confirm GREEN.
6. Run the full requested suites (console controller/store/bridge/swap + Tests/Agents) and check for flakiness on the real-thread AC1 test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC1 (main race): _run_agent_reply's should_cancel closure read only the shared, mutable _stop_requested, which the run's own finally resets the instant the coroutine handles the CancelledError raised by stop_active_run's task.cancel() -- but asyncio.to_thread survives Task cancellation, so the bridge's background OS thread keeps running past that reset and could observe should_cancel()==False, finishing (and persisting to AgentRunsDB) as RUN_DONE. Fix: a fresh threading.Event is created per run, captured by closure value (not read off self), and set by a new _signal_stop() helper (used by stop_active_run/close_session/shutdown) alongside _stop_requested; the finally block only detaches self._active_cancel_event (never .clear()s the Event itself), so a still-running thread's closure permanently observes the cancellation. Regression test (test_console_agent_swap.py::test_stop_during_parked_bridge_thread_persists_cancelled_not_done) uses a REAL ConsoleAgentBridge + AgentRunsDB with a gateway that blocks the actual worker thread on a threading.Event until after the coroutine side has fully unwound (finally already ran) -- confirmed RED pre-fix (persisted "done"), GREEN post-fix (persisted "cancelled"); stable across 5 repeated runs.

AC2 (LOW-1): ConsoleChatStore.reset_stream_content now no-ops (returns the message unchanged) when status is already "stopped", mirroring append_stream_chunk's existing hardening -- prevents a disobedient model's post-stop tool-call turn from resurrecting a stopped message back to "streaming".

AC3 (LOW-2): ConsoleChatController._finalize_agent_reply now reads the message back first and, if already "stopped" (the ultra-narrow window after asyncio.to_thread returns an outcome but before finalize runs), returns a benign STOPPED result without touching mark_message_complete/finalize_variant_stream/mark_message_failed -- avoiding both the ValueError those raise via _validate_can_mark_terminal and finalize_variant_stream's silent resurrection-to-complete (it has no such guard at all).

Files: tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/Chat/console_chat_store.py, Tests/Chat/test_console_agent_swap.py, Tests/Chat/test_console_chat_store.py.

Verification: Tests/Chat/test_console_chat_controller.py, test_console_chat_store.py, test_console_agent_bridge.py, test_console_agent_swap.py, Tests/Agents -- 281 passed, 0 failed.

### Follow-up: AC3 regenerate window (post-review)

**Honesty note**: the AC3 fix as originally shipped above only covered a
Stop landing in the post-outcome window for a *plain send/retry*. A
follow-up review found it did not actually cover a stopped **regenerate**
in that same window, because `mark_message_stopped`
(console_chat_store.py:678) RESTORES a mid-regenerate message to its
*prior* status (e.g. "complete"), not "stopped" -- so
`_finalize_agent_reply`'s `current.status == "stopped"` guard never fired
for that case, and the two failure modes the guard was meant to prevent
were both still reachable via regenerate:
- RUN_DONE fell through to `finalize_variant_stream`, which has no
  stopped-guard of its own and fabricated a phantom variant from the
  already-popped variant base, silently resurrecting the message to
  "complete" with a bogus extra variant entry.
- RUN_ERROR/RUN_STUCK fell through to `mark_message_failed`, which raised
  an uncaught `ValueError` via `_validate_can_mark_terminal` (status
  "complete" is not "pending"/"streaming") -- uncaught because the call
  happens in `_run_agent_reply` *after* its own try/except/finally block
  returns -- wedging `run_state` at STREAMING forever (every subsequent
  send rejected as "a Console run is already running").

**Fix** (this follow-up commit): `_finalize_agent_reply` now also accepts
the run's own per-run `cancel_event` (the same `threading.Event`
introduced for AC1, captured by value in `_run_agent_reply` and passed
through at the call site) and treats the run as stopped when
`cancel_event.is_set()` is true, independent of what status
`mark_message_stopped` left the message at. The original
`current.status == "stopped"` check is kept as a belt for any caller that
reaches this method without a `cancel_event` in scope. A genuinely
invalid finalize on a NON-stopped run still raises (unchanged).

**Tests** (RED-confirmed against the pre-follow-up code, then GREEN):
`Tests/Chat/test_console_agent_swap.py::test_finalize_after_already_stopped_regenerate_no_phantom_variant`
(RUN_DONE: no phantom variant, run_state STOPPED, content unchanged) and
`::test_finalize_after_already_stopped_regenerate_error_no_wedge`
(RUN_ERROR: no exception, run_state STOPPED not stuck STREAMING, next
send accepted). Full suite re-run:
`test_console_chat_controller.py` + `test_console_chat_store.py` +
`test_console_agent_swap.py` + `test_console_variant_stream.py` +
`Tests/Agents` -- 251 passed, 0 failed (plus
`test_console_agent_bridge.py` separately -- 37 passed).
<!-- SECTION:NOTES:END -->
