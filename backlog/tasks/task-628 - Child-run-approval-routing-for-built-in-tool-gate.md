---
id: TASK-628
title: Child-run approval routing for the built-in tool gate (nested-subagent stamp clobber)
status: Done
assignee: []
created_date: '2026-07-25'
labels: [tools, security, agents]
dependencies: [TASK-545]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`clamp_child_budget` zeroes `max_subagents` for a spawned child run, but the child still **inherits the parent's allow-list** (minus spawn/skill names — `Agents/agent_service.py:560-576`), so a child agent run can call built-in tools. TASK-545/P1's review hook and `BuiltinToolGate` are scoped **per run**: `BuiltinToolGate.begin_turn()` clears this turn's stamps at the start of every run's review pass, and a child run gets its own `begin_turn()` call. A child's `begin_turn()` therefore clears whatever stamps its *parent* run had just set for this turn, so a child's `begin_turn()` clears whatever stamps its *parent* run had just set for this turn.

MCP already has a documented answer to this class of problem: `MCPToolProvider.stamp_scope`, threaded through as `AgentService`'s generic `review_state_scope` seam and wired at `console_agent_bridge.py:1196-1200` (`getattr(mcp_provider, "stamp_scope", None)`). Built-ins have no equivalent — `BuiltinToolGate` exposes no `stamp_scope`, and nothing threads a shared or child-aware stamp scope for it.

This is **dormant today**: nothing resolves to `"ask"` in P1 because no built-in tool declares `risk_tags` yet, so no child run ever actually needs an approval route. It becomes live and security-relevant the moment TASK-545/P2 tags a mutating tool — a child run hitting `ask` with no route fails closed per the P1 design's §5 fallback, which is safe but would make every gated built-in tool call unusable from inside any sub-agent. This task is a **hard prerequisite for P2 shipping any gated mutating built-in tool**: either thread the parent's approval route (or a shared `BuiltinToolGate` instance/stamp scope) into children, or explicitly exclude `"mutates"`-tagged tools from child allow-lists until routing is solved.
**CORRECTION (implementation, 2026-07-25):** an earlier draft of this description attributed the failure to the child running "nested inside the parent's already-blocked worker thread", implying no approval round trip could reach the UI. **That is wrong.** `spawn()` calls `_run_one` directly — same call stack, same worker thread — and `call_from_thread` only needs the *UI* thread free, which it is. Nested approval round trips demonstrably work: `test_parent_deny_is_not_overridden_by_a_same_turn_spawned_childs_approval` drives two in a single turn for MCP. The real and only defect was the shared-mutable-state clobber described above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `BuiltinToolGate` gains a child-run-aware stamp scope (mirroring `MCPToolProvider.stamp_scope`/`review_state_scope`), so a child run's `begin_turn()` does not clobber verdicts the parent run already stamped this turn
- [x] A test drives a parent run that spawns a child, has the child call a tool tagged `"mutates"`, and confirms the child's call resolves via the shared/inherited approval route rather than failing closed solely because it is nested
- [x] Until the above ships, TASK-545/P2 does not tag any built-in tool `risk_tags=("mutates",)` that is reachable from a child run's inherited allow-list — documented explicitly as a blocking dependency on this task
- [x] Behavior for a child run with genuinely no available approval route (e.g. headless) still fails closed — this task does not weaken that guarantee, only makes the normal in-app path work
<!-- AC:END -->

## Implementation Notes

`BuiltinToolGate.stamp_scope()` mirrors `MCPToolProvider.stamp_scope`: snapshots `_stamps` and `_payload` on enter, **restores (never merges)** on exit. `_payload` is included because the child's `begin_turn()` drops it too, and restoring it preserves the parent's one-store-load-per-turn property.

`AgentService.review_state_scope` holds only ONE context manager, and two components now own per-turn state a nested run would clobber. Added module-level `_combine_state_scopes(scopes)` in `console_agent_bridge.py`, entering all present scopes on an `ExitStack`. It returns `None` for zero scopes and the single callable unchanged for one, so the pre-task-628 MCP-only wiring stays byte-identical.

Investigation corrected the task's own premise (see the CORRECTION above): there is no threading hazard, and the child already shares the parent's exact gate instance and review closure — this was purely a state-lifecycle bug.

Tests: `stamp_scope` unit tests (restore, restore-on-raise, reentrant nesting); `_combine_state_scopes` tests (passthrough for 0/1, both entered, reverse-order unwind on raise); and three integration tests through a **real spawn** — AC#2 (child's mutating-tool call resolves AND the parent's pre-child approval survives), a **negative control** running the identical interleave with the scope unwired to prove AC#2 isn't passing trivially, and AC#4 (a run with no approval route at all still fails closed).

AC#2's test was verified discriminating by sabotage: removing `stamp_scope`'s restore makes it fail (`assert not True`); restoring makes all 6 pass.

Suites: Agents+MCP+Tools 637 passed; Chat 2203 passed / 69 skipped.
