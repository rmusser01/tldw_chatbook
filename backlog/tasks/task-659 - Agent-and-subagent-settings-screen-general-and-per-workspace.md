---
id: TASK-659
title: Agent and sub-agent settings screen (general and per-workspace)
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [agents, settings, ui]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent runtime's capabilities and behavioral limits are effectively invisible and unconfigurable. A user cannot see what the agent is allowed to do, how much work it will attempt per message, what it will spend, or how its sub-agents behave — and cannot change any of it without editing source.

Today the only user-facing agent knob is `[console] agent_runtime` (a bool, default on, read at `UI/Screens/chat_screen.py` `_console_agent_runtime_enabled`). Everything that actually governs a run is a hardcoded module constant in `Chat/console_agent_bridge.py` and `Agents/agent_models.py`:

- `CONSOLE_MAX_MODEL_TURNS` (20) — tool-calling rounds per user message
- `CONSOLE_MAX_STEPS` (64) and `CONSOLE_MAX_WALL_SECONDS` (1200.0) — the two backstops that must stay sized together with it
- `RunBudget.max_subagents` (2), `max_active_tools` (8), `max_subagent_result_chars` (4000)
- `RunBudget.max_total_tokens` (0 = unlimited) — a spend ceiling nobody can set
- `RunBudget.max_tool_call_seconds` (300.0) — the per-tool-call timeout

These are not academic. Sub-agents inherit the parent's turn and step budget (`clamp_child_budget` clamps only wall-clock), so raising the round cap to 20 moved the worst case to `20 * (1 + max_subagents)` = **60 provider turns for a single message**, with no way for an operator to see or lower that. Tool permissions have a comparable gap: TASK-545/P1 ships built-in tool gating under the `agent:builtin` namespace with no UI at all, and TASK-656 covers surfacing that.

Users also work in distinct workspaces (`DB/Workspace_DB.py`, "local workspace operating contexts"), where appropriate agent behavior differs — an exploratory research workspace wants long runs and many sub-agents; a workspace touching real files wants tight caps and stricter tool permissions. A single global setting cannot express that.

This task adds a settings surface for agent and sub-agent behavior, with per-workspace overrides layered over a general default, following the existing settings-screen decomposition (`UI/Screens/settings_screen.py` plus per-domain modules such as `settings_library_rag_defaults.py`, `settings_privacy_security.py`, `settings_storage_defaults.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A settings surface exists for agent behavior, following the established per-domain settings-module pattern rather than adding a monolithic screen
- [ ] It displays the agent's current capabilities read-only: whether the agent runtime is enabled, which tools are available to it, and which runtime tools exist (spawn, find/load tools, skill file, install skill, run skill script)
- [ ] The run-budget knobs are editable: tool-calling rounds per message, step backstop, wall-clock backstop, max sub-agents, max active tools, token spend ceiling, and per-tool-call timeout
- [ ] Coordinated caps cannot be set into an unreachable state: raising rounds without a sufficient step backstop is either prevented or auto-sized, since a fence round costs 3 steps (N rounds need `3*(N-1)+1`); the invariant currently pinned by `test_console_budget_step_cap_admits_a_full_model_turn_run` holds for any user-set combination
- [ ] Sub-agent behavior is separately visible and configurable, and the screen states the worst-case provider-turn arithmetic (`rounds * (1 + max_subagents)`) so the cost consequence of a change is not hidden
- [ ] Settings resolve per-workspace with a general default: a workspace override wins where set, the general value applies otherwise, and the effective value in force is shown
- [ ] Changing a setting takes effect on the next run without an app restart, and is validated at the boundary (no value can produce a budget the runtime rejects or that makes a cap unreachable)
- [ ] The per-tool-call timeout cannot be set below MCP's own worst case without an explicit warning — lowering it re-opens the abandoned-thread double-execution window documented on `RunBudget.max_tool_call_seconds` (TASK-327)
- [ ] Unit tests cover: resolution precedence (workspace override vs general default), the coordinated-cap validation, and that a saved value actually reaches the next run's `RunBudget`
<!-- AC:END -->

## Notes

Complements TASK-656 (surfacing `agent:builtin` tool permissions in a UI). These two should almost certainly share one screen — permissions and behavior are the same operator question — so whichever ships second should extend the first rather than adding a parallel surface.
