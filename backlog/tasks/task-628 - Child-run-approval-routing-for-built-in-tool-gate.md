---
id: TASK-628
title: Child-run approval routing for the built-in tool gate (nested-subagent stamp clobber)
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tools, security, agents]
dependencies: [TASK-545]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`clamp_child_budget` zeroes `max_subagents` for a spawned child run, but the child still **inherits the parent's allow-list** (minus spawn/skill names — `Agents/agent_service.py:560-576`), so a child agent run can call built-in tools. TASK-545/P1's review hook and `BuiltinToolGate` are scoped **per run**: `BuiltinToolGate.begin_turn()` clears this turn's stamps at the start of every run's review pass, and a child run gets its own `begin_turn()` call. A child's `begin_turn()` therefore clears whatever stamps its *parent* run had just set for this turn, and — because the child runs nested inside the parent's already-blocked worker thread (`asyncio.to_thread`) — any gated call the child makes has no working approval route back to the UI.

MCP already has a documented answer to this class of problem: `MCPToolProvider.stamp_scope`, threaded through as `AgentService`'s generic `review_state_scope` seam and wired at `console_agent_bridge.py:1196-1200` (`getattr(mcp_provider, "stamp_scope", None)`). Built-ins have no equivalent — `BuiltinToolGate` exposes no `stamp_scope`, and nothing threads a shared or child-aware stamp scope for it.

This is **dormant today**: nothing resolves to `"ask"` in P1 because no built-in tool declares `risk_tags` yet, so no child run ever actually needs an approval route. It becomes live and security-relevant the moment TASK-545/P2 tags a mutating tool — a child run hitting `ask` with no route fails closed per the P1 design's §5 fallback, which is safe but would make every gated built-in tool call unusable from inside any sub-agent. This task is a **hard prerequisite for P2 shipping any gated mutating built-in tool**: either thread the parent's approval route (or a shared `BuiltinToolGate` instance/stamp scope) into children, or explicitly exclude `"mutates"`-tagged tools from child allow-lists until routing is solved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `BuiltinToolGate` gains a child-run-aware stamp scope (mirroring `MCPToolProvider.stamp_scope`/`review_state_scope`), so a child run's `begin_turn()` does not clobber verdicts the parent run already stamped this turn
- [ ] A test drives a parent run that spawns a child, has the child call a tool tagged `"mutates"`, and confirms the child's call resolves via the shared/inherited approval route rather than failing closed solely because it is nested
- [ ] Until the above ships, TASK-545/P2 does not tag any built-in tool `risk_tags=("mutates",)` that is reachable from a child run's inherited allow-list — documented explicitly as a blocking dependency on this task
- [ ] Behavior for a child run with genuinely no available approval route (e.g. headless) still fails closed — this task does not weaken that guarantee, only makes the normal in-app path work
<!-- AC:END -->
