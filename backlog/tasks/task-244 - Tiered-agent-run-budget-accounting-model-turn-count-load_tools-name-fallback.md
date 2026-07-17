---
id: TASK-244
title: >-
  Tiered agent-run budget accounting (model-turn count) + load_tools
  name-fallback
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-16 16:00'
updated_date: '2026-07-17 01:03'
labels:
  - agents
  - console
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CONSOLE_RUN_BUDGET.max_steps counts STEP_MODEL/STEP_TOOL_CALL/STEP_TOOL_RESULT identically even though only STEP_MODEL costs a provider round-trip, so a measured 10-step discovery-plus-one-tool-call run (4 real model turns) already consumes 62% of the 16-step budget before any retry or second tool round — the exact shape that hit step exhaustion on the Skills Phase-2 gate's successful discovery run. Separately, agent_service.py's load_schemas only resolves catalog ids (e.g. builtin:calculator), silently dropping a bare tool name a model echoes back from a find_tools result line, burning a full extra round on a generic 'No valid tools found to load' error even though ToolCatalogRegistry.resolve_name() already exists and is unused there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The run budget's primary limiter is (or is accompanied by) a model-turn/provider-call count rather than only raw AgentStep entries
- [ ] #2 The documented 10-step/4-model-turn discovery-plus-execution floor (console_agent_bridge.py's own comment) leaves headroom for at least 2 additional real tool rounds under the recommended budget
- [ ] #3 load_schemas resolves a bare tool name via registry.resolve_name() as a fallback when the direct catalog-id lookup fails, before giving up on that id
- [ ] #4 A new test reproduces a model calling load_tools with a bare name (not a catalog id) and confirms the tool loads instead of erroring
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-tiered-budgets-load-name-fallback.md — 2 SDD tasks: (1) RunBudget.max_model_turns (default 8 == max_steps, provably unreachable at engine defaults; clamp_child_budget passthrough) + loop model-turn check w/ distinct 'model-turn budget exhausted' copy + CONSOLE_RUN_BUDGET -> max_model_turns=8/max_steps=32 backstop + comment rewrite; (2) load_schemas resolve_name fallback for bare tool names, gates unchanged.
<!-- SECTION:PLAN:END -->
