---
id: TASK-543
title: >-
  Console branching: record the persisted reply id for stopped-via-cancel agent
  runs
status: Done
assignee:
  - '@claude'
created_date: '2026-07-24'
labels:
  - console
  - agents
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase C (agent-marker anchoring) writes the produced assistant reply's persisted id onto the agent run on every terminal path that returns through the finalizer (success, failure, cancelled-outcome, and the post-outcome `stopped_now` race). The dominant user-Stop path is different: `stop_active_run` calls `task.cancel()`, so the controller's `run_id, outcome = await asyncio.to_thread(...)` raises `CancelledError` before `run_id` is ever bound, and the `except asyncio.CancelledError` branch returns without recording. The run's `assistant_message_id` stays NULL even though the stopped reply WAS persisted, so it falls to the ordinal fallback on resume instead of being id-anchored + off-path-hidden. This is not a regression (it is exactly pre-Phase-C behavior) and is benign on a linear resume; it only leaks stale markers via leftover-append when a stopped reply later becomes an off-path sibling (stop → edit-&-resend the parent turn → resume onto the new branch). Documented in the Phase C plan addendum.

Fix direction (final-review recommendation): expose the most-recent primary run id via the bridge (per conversation/session) so the CancelledError branch can call `_record_run_assistant_message` too, and add a test that exercises the REAL `stop_active_run` → `task.cancel()` path (every current stop/cancel test simulates the stop via a normally-returning `run_reply`, so the gap is invisible to the suite).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user Stop delivered via task cancellation records the stopped reply's persisted id on the run (id-anchored + off-path-hidden on resume, same as other terminal paths)
- [x] #2 A test exercises the real `stop_active_run` → `task.cancel()` path and asserts the run's `assistant_message_id`
- [x] #3 A never-persisted stopped reply still leaves the run NULL (ordinal fallback), never a stale id
<!-- AC:END -->

## Implementation Plan

1. Bridge seam `latest_unanchored_primary_run_id(conversation_id)`: newest
   non-superseded PRIMARY run's id, returned ONLY while its
   `assistant_message_id` is still NULL (the mis-write guard for a Stop that
   beats `create_run` -- a finished run is always anchored by a finalizer
   path, so an anchored newest row means record nothing).
2. Controller CancelledError branch: after `_mark_stream_stopped`, call
   `_record_run_assistant_message` with the recovered id (defensive wrapper;
   the existing helper already no-ops on a never-persisted stop -> AC#3).
3. Tests exercising the REAL `stop_active_run` -> `task.cancel()` path via
   the parked-bridge-thread scaffolding (yield-then-park gateway so the gate
   flushes a chunk and the stopped reply persists), plus the NULL guard.

## Implementation Notes

- The run ROW exists long before a user can Stop (`create_run` at loop
  start), so the newest non-superseded primary IS the active run; the
  NULL-anchor guard turns the one racy exception into a safe no-op instead
  of overwriting the previous run's good anchor.
- Test detail discovered en route: the agent stream's fence-gate is
  line-buffered -- a chunk without a newline never flushes to the store, so
  a zero-chunk (or newline-less) stop legitimately never persists and stays
  NULL -> ordinal fallback (pinned by the no-persistence test).
- Files: `tldw_chatbook/Chat/console_agent_bridge.py`,
  `tldw_chatbook/Chat/console_chat_controller.py`,
  `Tests/Chat/test_console_agent_swap.py`.
