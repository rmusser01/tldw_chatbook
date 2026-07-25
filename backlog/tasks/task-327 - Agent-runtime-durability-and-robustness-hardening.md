---
id: TASK-327
title: Agent runtime durability and robustness hardening
status: Done
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-07-25 07:18'
labels:
  - agents
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several low-severity robustness gaps in the agent runtime, each bounded today (none Critical), grouped as one hardening pass. Bundled per finding for a single PR; can be split if preferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loop detection catches cyclic (non-consecutive) tool calls, e.g. A→B→A→B, not only consecutive repeats (`agent_runtime.py:352-360`)
- [x] #2 On app start, pre-existing `running` agent-run rows are reconciled to an error/interrupted state (they are orphaned on crash; `agent_service.py:455` persists steps only at run end)
- [x] #3 `AgentRunsDB` sets `PRAGMA busy_timeout` and enables WAL for concurrent-run writes (`base_db.py:97`)
- [x] #4 A runtime-level per-tool timeout wraps `deps.invoke_tool` so a custom/blocking provider cannot wedge a cooperative-cancel run (`tool_catalog.py:96-111`)
- [x] #5 The bridge guards one active run per conversation (shared `_live`/`_historical_cache` in `console_agent_bridge.py`), or documents the controller as the sole serialization point
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Five durability fixes to the agent runtime. AC#1: replaced the one-slot consecutive-repeat loop guard with a deque of the last LOOP_DETECTION_N*MAX_LOOP_PERIOD call-keys + a pure _detect_cycle() that trips RUN_STUCK on any period-1..4 cycle (period-1 keeps the 3-repeat threshold; periods>=2 trip at 2 full repeats), so A->B->A->B is now caught. AC#2: AgentRunsDB.reconcile_orphaned_runs() flips crash-orphaned 'running' rows to 'error' (COALESCE-preserving any partial result) once per file per process on open. AC#3: WAL + busy_timeout=5000 on AgentRunsDB connections (WAL guarded for :memory:). AC#4: RunBudget.max_tool_call_seconds (default 300.0s, 0=unlimited) is deliberately set ABOVE MCP's own ~186s worst case (121s approval wait + 65s execution), propagated through clamp_child_budget, enforced by a module-level _call_with_timeout daemon-thread helper (catches BaseException, with a defensive fallback so it can never raise into the pure loop) wrapping only the builtin/custom registry.invoke_by_name. Skill calls route around it into their own budget-clamped nested loop, but MCP tools are NOT exempt: MCPToolProvider shares the same per-run ToolCatalogRegistry as builtins, so MCP calls DO flow through the timeout wrapper -- the generous default exists so the wrapper never preempts MCP's own internal bounds. AC#5: doc-only -- documented ConsoleChatController (_active_run_rejection/run_state.is_send_allowed, Tests/UI/test_console_run_gate.py) as the sole per-conversation run-serialization point; the bridge's _live/_historical_cache are display-only, not a mutual-exclusion guard. The pure runtime stays pure (timeout lives entirely in agent_service). Files: agent_models.py, agent_runtime.py, AgentRuns_DB.py, agent_service.py, console_agent_bridge.py + tests.
<!-- SECTION:NOTES:END -->
