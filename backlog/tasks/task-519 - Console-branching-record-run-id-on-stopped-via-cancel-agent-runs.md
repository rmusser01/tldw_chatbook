---
id: TASK-519
title: >-
  Console branching: record the persisted reply id for stopped-via-cancel agent
  runs
status: To Do
assignee: []
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
- [ ] #1 A user Stop delivered via task cancellation records the stopped reply's persisted id on the run (id-anchored + off-path-hidden on resume, same as other terminal paths)
- [ ] #2 A test exercises the real `stop_active_run` → `task.cancel()` path and asserts the run's `assistant_message_id`
- [ ] #3 A never-persisted stopped reply still leaves the run NULL (ordinal fallback), never a stale id
<!-- AC:END -->
