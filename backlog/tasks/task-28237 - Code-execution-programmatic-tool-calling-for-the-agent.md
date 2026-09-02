---
id: TASK-28237
title: Code execution / programmatic tool calling for the agent
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - agents
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hermes-parity deferred row C6, promoted by TASK-26041's review: still the highest-leverage unfiled row. Let the model write a short script that composes existing tools programmatically instead of one JSON round-trip per call. Both preconditions matured on dev: registry.invoke_by_name is the single tool-dispatch choke point (Agents/agent_service.py:3013, run_tool_policy.py:5), and sandboxed skill-script execution already exists with discard-writes semantics (Agents/tool_catalog.py:458). This composes them: a script runtime whose tool bindings route through the SAME permission gate as individual calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A script can call multiple registered tools in one execution, each call passing through the existing permission gate and execution log
- [ ] #2 Script execution reuses the existing sandboxed skill-script runtime; no new execution surface
- [ ] #3 A gated tool inside a script raises the same approval card as a direct call; denial fails the script honestly
- [ ] #4 Resource bounds (wall time, output size) match or tighten the existing per-call bounds
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-28227, colliding with the
"Agent-loop-active-turn-redirect-keeping-completed-tool-results" task that arrived on origin/dev first (dev minted 28226-28228
within the hour after this batch's sweep; re-verified at the wave-2 dev merge,
2026-09-02). Per the TASK-19601 owner rule the younger task renumbers with
provenance; it is now TASK-28237.
