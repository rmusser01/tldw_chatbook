---
id: TASK-327
title: Agent runtime durability and robustness hardening
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: [agents, tech-debt]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several low-severity robustness gaps in the agent runtime, each bounded today (none Critical), grouped as one hardening pass. Bundled per finding for a single PR; can be split if preferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Loop detection catches cyclic (non-consecutive) tool calls, e.g. A→B→A→B, not only consecutive repeats (`agent_runtime.py:352-360`)
- [ ] #2 On app start, pre-existing `running` agent-run rows are reconciled to an error/interrupted state (they are orphaned on crash; `agent_service.py:455` persists steps only at run end)
- [ ] #3 `AgentRunsDB` sets `PRAGMA busy_timeout` and enables WAL for concurrent-run writes (`base_db.py:97`)
- [ ] #4 A runtime-level per-tool timeout wraps `deps.invoke_tool` so a custom/blocking provider cannot wedge a cooperative-cancel run (`tool_catalog.py:96-111`)
- [ ] #5 The bridge guards one active run per conversation (shared `_live`/`_historical_cache` in `console_agent_bridge.py`), or documents the controller as the sole serialization point
<!-- AC:END -->
