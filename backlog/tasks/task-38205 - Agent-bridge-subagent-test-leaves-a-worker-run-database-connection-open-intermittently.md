---
id: TASK-38205
title: Agent bridge sub-agent test leaves a worker run-database connection open intermittently
status: Done
assignee:
- '@claude'
created_date: 2026-10-30 09:30
labels:
- agents
- flaky-test
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Chat/test_console_agent_bridge.py::test_tool_marker_with_brackets_renders_literally_not_escaped errors intermittently at teardown: the autouse _retire_original_run_databases fixture finds an AgentRunsDB maintenance participant still holding an admitted connection after the main-thread close ("Main-thread original close must not hide a live worker owner"). The test spawns a sub-agent, so a worker thread appears to still own a run-database connection after run_reply returns. It reproduced in 1 of 6 runs of the file on dev 03eb75814b (xdist -n 4) and on PR #3286's heads, so it is not caused by that PR. The rate is load-sensitive: 1 of 6 on dev in one session, but up to 10 of 11 whole-file runs on a loaded machine. Likely mechanism, from code reading: worker_guard (Agents/activation.py) closes a fleet child's thread-local AgentRunsDB handle in its finally, which runs only after run_child returns, i.e. after fleet.finish and the settle signal (agent_service.py, the child run path), so the primary can return and the fixture can close and assert while the child thread has not yet reached that finally. Either the fleet should signal settlement only after the child releases its connection (a real ownership gap), or the fixture should wait for the late release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The open connection's owning thread and call site are identified, and the leak is classified as a product ownership bug or a test-timing assumption, with evidence.
- [x] #2 The test file passes 20 consecutive runs under xdist on dev without weakening the fixture's no-live-worker-owner assertion unless the ownership contract changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Instrument (temporarily) the fixture's failing assertion to print each still-registered connection's owner thread, whether it is alive, the stack that opened it, and the owner's current stack; reproduce under xdist load.
2. Classify: if the owner is a child that already settled but has not reached `worker_guard`'s close, fix the product ordering with a regression test; if it is a child still running under the survivor contract (`subagents_outlive_turn`), the test is missing the synchronisation its siblings use.
3. Remove the instrumentation, measure the unmodified repro rate, apply the fix, then run the whole file 20 consecutive times under xdist with a second copy running concurrently.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Test-timing assumption, not a product ownership bug. Fixed by joining the fleet threads in the test before teardown, as its sibling `test_spawn_renders_marker_and_persists_linked_subagent` already does. The fixture's assertion is unchanged.

**Mechanism (AC #1).** Temporary instrumentation recorded each registered lease's opening stack in `participants._register_core_connection` and printed it at the fixture's assertion, with `lease.resource_thread` and that thread's live stack from `sys._current_frames()`. It reproduced on the first run:
- Owner: thread `fleet-4cd1093b`, alive, the spawned child's worker.
- Opened at `run_child -> _run_owned -> _run_one -> _latest_durable_event_id -> AgentRunsDB.get_run` (agent_service.py ~4675/3726).
- Owner's stack at assert time: still in its first model call (`run_agent_loop -> call_model -> observe_context_assembled -> observe_trace_step -> AgentRunsDB.insert_steps_at_indices`).
- Log: `_settle_fleet: 1 sub-agents are outliving their turn` came before `run_reply end`.

The child had not finished, so the suspected window (settle signal before `worker_guard`'s close) was not involved. `DEFAULT_SUBAGENTS_OUTLIVE_TURN = True` and `_settle_fleet` deliberately does not join survivors (PR3a-1 Task 2), so `run_reply` returns while the child still runs. That is the product contract. `_join_fleet_threads()` waits for the thread to end, which is after `worker_guard`'s `finally` closes the child's handle, so the wait is bounded and deterministic.

**Evidence (AC #2).** All runs used the whole file, `run.sh <W> ab1|ab2 4 Tests/Chat/test_console_agent_bridge.py` (xdist -n 4), with two copies running at once to add load.
- Before the fix: 9 of 12 runs errored on this test only (6 per copy).
- After the fix: 40 of 40 passed (20 consecutive per copy, 343 passed each), at load average about 36 on 18 cores.

No product code changed, so no ordering regression test was added and the size ratchets are unaffected. Ruff showed the same 21 errors and 28 format hunks in this test file before and after the change; none are on the changed lines, and the file was not reformatted.

Files: `Tests/Chat/test_console_agent_bridge.py`, `backlog/docs/lessons-testing-evidence.md` (new entry), this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
