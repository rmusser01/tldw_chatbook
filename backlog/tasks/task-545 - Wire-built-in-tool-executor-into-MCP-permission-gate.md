---
id: TASK-545
title: Wire built-in tool executor into MCP permission gate
status: To Do
assignee: []
created_date: '2026-07-24 12:00'
updated_date: '2026-07-25'
labels: [tools, security, agents]
dependencies: [TASK-331]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Rescoped 2026-07-25 — see `Docs/superpowers/specs/2026-07-25-builtin-tool-permission-gate-design.md` for the full investigation.** The original text described wiring `Tools/tool_executor.py`'s `ToolExecutor` and gating `write_file`/`create_note`/`update_note`. That premise does not hold: `ToolExecutor` (System A — reached only from the deprecated legacy/enhanced chat path, `Event_Handlers/worker_events.py` and `chat_streaming_events.py`, on the main loop) and `Agents/tool_catalog.py`'s `BuiltinToolProvider` (System B — the path `run_agent_loop` actually calls on the worker thread, today only `calculator`/`datetime`) are two disjoint systems, not two call sites into one path. The Console never reaches System A at all — `console_chat_controller.py`/`console_agent_bridge.py` contain zero references to `execute_tool_call`/`get_tool_executor`/`tool_executor` — while System A's fs/note tools (`write_file`, `create_note`, `update_note`, etc.) live there and nowhere else, and its own `[tools]` config is separately dead (TASK-547). Gating System A would harden code slated for removal while leaving the live path (System B) unprotected; gating System B needs a permission gate that doesn't exist yet.

**Decision (user-approved): port the fs/note tools to the agent runtime (System B) and gate them there.** Work is split into three phases, all tracked under this one task:

- **P1 — the gate seam (done, this pass).** Built on branch `feat/builtin-tool-permission-gate`: a concrete `Tool.risk_tags` property (default `()`, vocabulary = existing `HIGH_RISK_TAGS`); `resolve_builtin_state` in `MCP/permission_store.py`, a sibling of `resolve_effective_state` (not a modification of it) under a namespace deliberately distinct from the built-in MCP *server*'s own live key — `server_key = "agent:builtin"`, never `builtin:tldw_chatbook` (`MCP/readiness.py` `BUILTIN_SERVER_KEY`) — with precedence tool override → server default → built-in `allow` floor, no definition-hash comparison, and the existing risk-flooring pass unchanged; `Agents/builtin_tool_gate.py`'s `BuiltinToolGate` (one permission-store load per turn, fail-closed on a missing service, never raises); enforcement inside `BuiltinToolProvider.invoke` as defense-in-depth; a run-level `build_tool_review_hook` wired **unconditionally** (previously the review hook existed only when MCP was configured, so a user with zero MCP servers had no gate at all); and per-row `options` on `ChatApprovalCard` so built-in rows can be narrowed (excluding only `always_allow`, the sole persistent write) without lying about what a bulk action will do. Session-scoped decisions only — nothing persists under `agent:builtin` in P1. No tool is ported or tagged yet, so this phase changes no behavior for `calculator`/`datetime` (both untagged, both still executing silently) **except one deliberate change**: the kill switch is now global (spec §7), so a user with the MCP kill switch on now has built-in tool calls refused where they previously executed — this is intentional and is what the "kill switch blocks built-in tools" acceptance criterion pins.
- **P2 — port the tools (not started).** Move `read_file`, `list_directory`, `write_file`, `rag_search`, `create_note`, `search_notes`, `update_note`, `web_search` into `BuiltinToolProvider`, tag the mutating ones with `risk_tags`, and let this same machinery gate them end-to-end. **Blocked on the child-run approval routing follow-up** (nested sub-agents inherit the parent's allow-list and can call built-ins, but the review hook is per-run and has no equivalent of MCP's `review_state_scope`) — P2 must not ship a gated mutating tool before that is resolved.
- **P3 — config and the legacy path (not started).** Fix the dead `[tools]` config (TASK-547) and decide System A's fate (port its remaining behavior, gate it in place, or remove `Tools/tool_executor.py` entirely) so no tool anywhere ever executes ungated.

TASK-331's sandbox fix made the fs tools functional-within-a-sandbox; this task's P1+P2 are the intended protection layer on top of that. This task stays open (not Done) until P2 and P3 both land — see the unchecked criteria below.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
P1 — the gate seam (complete):
- [x] `Tool` ABC exposes a concrete `risk_tags` property defaulting to `()`; no existing subclass breaks
- [x] A built-in tool resolves through `resolve_builtin_state` via a `GatedToolRef` using `server_key="agent:builtin"`, with precedence tool override → server default → built-in `allow` floor, and no definition-hash comparison
- [x] No built-in permission entry is ever written or read under `builtin:tldw_chatbook`; a test asserts the two namespaces stay disjoint
- [x] With no `unified_mcp_service` available, untagged tools still execute and `"mutates"` tools fail closed — a missing service is never treated as allow-everything
- [x] Turn-level resolution: a run executing N built-in calls in one turn performs at most one permission-store load
- [x] An untagged built-in (calculator, datetime) resolves to `allow` and executes with no prompt — verified as a no-behavior-change test
- [x] A tool tagged `"mutates"` has its inherited allow floored to `ask` (`risk_floored=True`)
- [x] `BuiltinToolProvider.invoke` blocks execution when the kill switch is on, when the resolved state is `deny`, and when the state is `ask` with no approval route — each returning `ToolResult(ok=False, ...)`, never raising
- [x] A run-level `build_tool_review_hook` is wired whether or not MCP is configured, and folds MCP's gate in when a provider exists; built-in and MCP approvals appear in one card per turn
- [x] `resolve_builtin_state` exists as a sibling of `resolve_effective_state`; the MCP resolver's behavior is unchanged (its existing tests still pass untouched)
- [x] `BuiltinToolProvider(gate=None)` is gated by default — a test asserts a bare instance refuses a `deny`-resolved tool
- [x] The review hook routes calls by owner: MCP-claimed names to the MCP gate, registry-resolved built-ins to the built-in gate, and everything else (skills, native spawn) unreviewed
- [x] Built-in approvals are session-scoped only; no persistent allow/deny is written for `agent:builtin` in P1
- [x] The approval card offers built-in rows `approve_once`/`approve_session`/`deny` and NOT `always_allow`; MCP rows keep all four, verified byte-identical
- [x] The kill switch blocks built-in tools, and its MCP workbench label/echo no longer describe it as MCP-only
- [x] Tests inject a temporary permission-store path; no test touches the real user store

P2 — port the tools (not started):
- [ ] `read_file`, `list_directory`, `write_file`, `rag_search`, `create_note`, `search_notes`, `update_note`, `web_search` are implemented as `BuiltinToolProvider` tools, reachable from the Console's agent-runtime path
- [ ] Each mutating tool carries `risk_tags` and is floored to `ask` by default; a live end-to-end test drives at least one tool through allow/ask/deny
- [ ] The child-run approval routing follow-up is resolved before any mutating tool ships gated (no unreviewable nested-subagent execution path)

P3 — config and legacy path (not started):
- [ ] TASK-547's dead `[tools]` config is fixed and reachable
- [ ] System A's (`Tools/tool_executor.py`) fate is decided and implemented — ported, gated in place, or removed — with no tool left executing ungated in either system
<!-- AC:END -->
