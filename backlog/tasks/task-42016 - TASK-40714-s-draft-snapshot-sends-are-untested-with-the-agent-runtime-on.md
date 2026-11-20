---
id: TASK-42016
title: "TASK-40714's draft-snapshot sends are untested with the agent runtime on"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:38
labels:
- follow-up
- console
- tests
priority: low
references:
- Tests/UI/test_console_send_without_mcp_store.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41900 (round 17). TASK-41900 runs TASK-40714's ordinary send with the agent runtime on; the draft-snapshot sends (queued, armed tokenless, spoken) go through the same `_AgentSources` root but are still tested with the runtime off.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each draft-snapshot send is also tested with the agent runtime on and an unbound store
<!-- AC:END -->

## Implementation Plan

1. Extend `test_draft_snapshot_sends_without_a_bound_mcp_store` with the TASK-41900 `runtime` parameter (`gateway`, `agent`): in the agent case the wrapped builder turns the runtime on after the build and reloads `app.app_config` (`_turn_on_agent_runtime`).
2. Assert the runtime really reached each send (TASK-41900's lesson), not only that the scenario passed.
3. Run the new nodes; if a scenario fails with the runtime on, trace and report it before changing product code.
4. Show the assertion of the runtime path red under a temporary edit, then green; record lane times.

## Implementation Notes

- **Approach (TASK-41900's pattern).** `test_draft_snapshot_sends_without_a_bound_mcp_store` gains the `runtime` parameter (`gateway`, `agent`): 3 scenarios x 2 unbound profiles x 2 runtimes, 12 nodes (6 new). One wrapped builder now serves both profiles; in the agent case it calls `_turn_on_agent_runtime` (saves `[console] agent_runtime = true`, reloads `app.app_config`, records each `_AgentSources`).
- **Correction to the Description.** It says the draft-snapshot sends "go through the same `_AgentSources` root". Only the spoken send does in these scenarios: it is the only one that runs its turn. The queued send stops at the queue entry and the armed tokenless send's `accept_turn` is replaced by the scenario, so neither builds `_AgentSources`. With the runtime on they share only the owned capture, which freezes `agent_runtime_enabled` (asserted below).
- **The path ran (TASK-41900's lesson).** Each node asserts that the send's owned capture froze the runtime flag as set (`tool_configuration["agent_runtime_enabled"]`). Only the spoken send runs a turn in its scenario (the queued one waits in the queue, the armed one replaces `accept_turn`), so with the runtime on it must also have built exactly one `_AgentSources`; the other two must have built none. `_record_mcp_maximum` became `_record_captures` (it keeps the snapshots), so the ordinary-send test and this one share one recorder.
- **Outcome.** All 12 pass at base product code: no product change was needed. Probe runs showed each scenario makes exactly one owned capture.
- **Red evidence (all temporary, reverted).** (1) TASK-41900's fix off (`unbound_console_sources` branch in `_AgentSources.__init__` disabled): both spoken-agent nodes failed (the spoken send never reported; `_speak_armed_send` timed out); the other 10 passed, as they run no turn. (2) TASK-41900's trap (runtime saved after the build, no `app_config` reload): both spoken-agent nodes failed with `assert {False} == {True}`; the queued and armed scenarios reload `app_config` themselves (`_persist_console_provider_config`), so they stayed green. (3) Runtime left off in the agent nodes: all 6 failed with `assert {False} == {True}`.
- **Runs.** Whole file at `-n 4` (load 19.7): 30 passed. Lane: the file is already a whole UI census entry (no census edit). Serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`), load 23.1 -> 15.3: 30 passed in 173 s; the 6 new nodes' calls took 32.7 s (spoken 8.8 + 7.8, queued 4.5 + 4.1, armed 4.0 + 3.5).
- Files: `Tests/UI/test_console_send_without_mcp_store.py`.
