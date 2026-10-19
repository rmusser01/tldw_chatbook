---
id: TASK-35311
title: >-
  Finalize a DONE reply as complete when shutdown or session close lands after run end
status: Done
assignee:
  - Robert
created_date: '2026-10-18 00:00'
labels:
  - console
  - durability
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console UAT 2026-10-17 (`qa/console-uat-2026-10-17/`, origin/dev `6c0bde2437`): assistant
replies ended with `assistant_generation_state='stopped'` instead of `'complete'` when the app
instance shut down soon after the reply finished — content fully persisted and rendered, only
the terminal state wrong. A reply whose app lived longer got `'complete'`.

Cause: the bridge worker returns `RUN_DONE` but the controller task is still suspended at
`await run_provider_worker` or has not reached `mark_message_complete`. `controller.shutdown()`
calls `_signal_stop` (permanently sets the cancel event) then cancels the task; the
CancelledError branch and `_finalize_agent_reply`'s `stopped_now` decide purely on
`cancel_event.is_set()`, discarding the DONE outcome. User-Stop and shutdown are currently
indistinguishable at this seam (citation repair already threads a `cancel_reason`).

Governance: ADR-230 / ADR-094 turn-lifetime context; user-Stop semantics are pinned by
`test_shutdown_stops_and_awaits_active_stream_task` and the durable-turn acceptance
parametrization and must not change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:AC:BEGIN -->
- [x] A cancellation landing after the run outcome is RUN_DONE with reason shutdown/session-close finalizes the message `complete` with full content
- [x] User Stop after a DONE outcome still finalizes `stopped` (existing spec preserved)
- [x] Shutdown of a still-streaming run still finalizes `stopped` (existing spec preserved)
- [x] The finalization step is shielded so shutdown cannot skip the terminal write
- [x] Adjacent stop/shutdown/finalize tests stay green
<!-- SECTION:AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test: cancel injected at the finalize window with a DONE outcome and shutdown reason →
   assert `complete` (today `stopped`).
2. Thread a stop reason (`user`/`shutdown`/`session_close`) through `_signal_stop`, recorded
   per session; feed citation repair's `cancel_reason`.
3. CancelledError branch + `_finalize_agent_reply.stopped_now` become reason- and
   outcome-aware; shield the finalize call site (bounded drain).
3. Variant tests for user-stop and mid-stream shutdown.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: stop-reason threading. `_signal_stop` now records why each session's cancel event was
  set (`user` / `shutdown` / `session_close`); callers updated (shutdown and `_finish_begin_shutdown`
  pass "shutdown", both `begin_session_close` sites pass "session_close", Stop button and draft
  supersession keep the "user" default). `_finalize_agent_reply.stopped_now` treats a set cancel
  event as "must be stopped" only for user stops or non-DONE outcomes.
- Outcome recovery: `_run_agent_reply` now runs the provider worker as a retained shielded task
  (`_active_provider_workers` registry), so a shutdown cancel landing on the awaiting submit task
  can recover the already-finished `(run_id, outcome)`; with RUN_DONE and a non-user reason the
  CancelledError handler runs the success finalization via a bounded drain
  (`_finalize_despite_cancellation`) instead of marking stopped.
- Deliberately NOT shielded: the normal finalization call site stays a direct await in the submit
  task — moving it into a task broke citation repair's post-generation step (regression caught by
  `test_citation_repair_agent_real_genuine_fallback_copy_does_not_bypass`, reverted).
- Semantics preserved (tests green): user Stop after DONE → `stopped`; shutdown of a still-streaming
  run → `stopped` with partial content; accepted-but-not-dispatched turns settle via the untouched
  `console_native_commit` path. Known adjacent case (documented, not changed): teardown landing
  after the text renders but before the run outcome returns leaves `stopped` with full content —
  the stream is not provably finished there.
- Tests: `Tests/Chat/test_console_shutdown_after_done_finalization.py` (4, RED first), cloning the
  `test_console_agent_swap.py` post-outcome-window harness; agent-swap and durable-turn-acceptance
  suites re-run (2 pre-existing baseline-red stop tests documented, not caused by this change).
- UAT validation: full `qa/console-uat-2026-10-17` re-run — 17/17 checkpoints, DB terminal states
  now `complete` for J1/J2/J3 replies and `failed` for the injected 500.
- Modified files: `Chat/console_chat_controller.py` (reason threading, recovery, drain helper),
  plus the new test file.
- ADR check: no new ADR — aligns terminal-state writes with the durability spec pinned by
  TASK-337/task-227 tests inside the existing ADR-230/094 turn-lifetime boundaries.
