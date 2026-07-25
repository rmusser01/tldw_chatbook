# Built-in tool permission gate (P1 of TASK-545)

**Date:** 2026-07-25
**Backlog:** TASK-545 (wire built-in tool executor into the permission gate), with TASK-547 (dead `[tools]` config) deferred to P3.
**Branch:** `feat/builtin-tool-permission-gate` (worktree off `origin/dev` @ `d8364963b`).

Phase 1 of three. Ships the **gate seam** on the live agent-runtime path — no tool porting, no config changes. P2 ports the fs/note tools behind this gate; P3 fixes `[tools]` and decides the legacy path's fate.

## Why the filed task's plan is not what we are building

TASK-545's description says to wire `Tools/tool_executor.py`'s `ToolExecutor` into the gate, naming `write_file`/`create_note`/`update_note`. Investigation (2026-07-25, against `origin/dev`) found that premise does not hold:

**`ToolExecutor` and `BuiltinToolProvider` are two disjoint systems with non-overlapping tool sets, not two call sites into one path.**

| | System A — `Tools/tool_executor.py` | System B — `Agents/tool_catalog.py` |
|---|---|---|
| Class | `ToolExecutor` | `BuiltinToolProvider` |
| Tools | datetime, calculator, code_audit **+ read_file, list_directory, write_file, rag_search, create_note, search_notes, update_note, web_search** | datetime, calculator **only** (`tool_catalog.py:132-133`) |
| Executed from | `Event_Handlers/worker_events.py:590-594` and `Event_Handlers/Chat_Events/chat_streaming_events.py:268-270` — both Textual message handlers on the **main loop** | `ToolCatalogRegistry.invoke_by_name` ← `AgentService._make_invoke_tool` ← `run_agent_loop`, on the **worker thread** (`console_chat_controller.py:3306` `asyncio.to_thread`) |
| Reached by the Console | **No.** `console_chat_controller.py` / `console_agent_bridge.py` contain zero references to `execute_tool_call`, `get_tool_executor`, or `tool_executor` | Yes — this is the shipping path |
| Gate today | none (and its config is dead — TASK-547) | **none at all** |

System A is reachable only from the legacy/enhanced chat path, which the project has declared deprecated in favour of the Console (the same scope decision that excluded all enhanced-path findings from tasks 320-334). Gating it would harden code slated for removal while leaving the live path unprotected.

**Decision (user-approved):** port the tools to the agent runtime and gate them there. P1 builds the gate; System A is left untouched, its fs tools remaining unreachable because its config is dead. Its fate is decided in P3, so no ungated window is ever opened.

**Task hygiene:** TASK-545's ACs describe System A and must be rewritten to match this scope before P1 is marked Done (CLAUDE.md requires updating the AC when deviating, not just the notes).

## Design

The gate reuses MCP's permission machinery rather than duplicating it. `resolve_effective_state` (`MCP/permission_store.py:516`) touches only `tool.server_key`, `tool.name`, `tool.description`, `tool.input_schema`, and `tool.tags` — and `HubTool`'s own docstring already names `builtin:tldw_chatbook` as a legitimate `server_key`, with `source` documented as `local|builtin|server` (`MCP/hub_tool_catalog.py:20-48`). The data model anticipated this.

### 1. Risk tags on the `Tool` ABC

`Tools/tool_executor.py`'s `Tool` gains:

```python
@property
def risk_tags(self) -> tuple[str, ...]:
    return ()
```

**Concrete, not abstract** — an abstract property would break every existing `Tool` subclass. The vocabulary is the *existing* `HIGH_RISK_TAGS = frozenset({"mutates", "process"})` (`permission_store.py:69`), not a new taxonomy. Keep tags ≤ 5 (`HubTool._MAX_TAGS`).

P1 tags nothing: calculator and datetime are read-only and stay untagged. The mutating tools that get `("mutates",)` arrive in P2.

### 2. Built-in → HubTool adapter

A pure function mapping a built-in `Tool` onto the frozen `HubTool` dataclass:

```
server_key="builtin:tldw_chatbook", server_label="Built-in", source="builtin",
name=tool.name, description=tool.description,
input_schema=tool.parameters, tags=tool.risk_tags,
stale=False, executable=True
```

### 3. Resolution: allow-floor + risk flooring, no hash

**A new resolver, not a parameter on the existing one.** `resolve_effective_state` walks tool override → server default → **MCP global default** and compares a stored `definition_hash`. Built-ins need a different floor and no hash, so P1 adds a sibling to `MCP/permission_store.py`:

```python
def resolve_builtin_state(payload: dict[str, Any], tool: HubTool) -> EffectiveToolState
```

It mirrors `resolve_effective_state`'s precedence walk and reuses its risk-flooring logic verbatim, substituting the built-in floor and omitting the hash step. A separate function rather than an optional argument keeps the MCP path byte-identical and avoids adding a conditional branch inside a security-critical resolver.

Precedence: **tool override → server default (if the user set one) → built-in floor `"allow"`**, then the existing risk-flooring pass (inherited allow + `HIGH_RISK_TAGS` → `ask`, `risk_floored=True`).

Deliberately **not** the MCP global default: changing MCP's global posture must not silently start prompting for calculator. Net effect today — untagged tools run silently (zero behavior change), and P2's `"mutates"` tools land on `ask` automatically.

**No definition-hash comparison for built-ins.** `definition_hash` guards against a *remote* server mutating a tool after you trusted it. Built-in tools are in-process code: anyone who can change them already has code execution, so the check provides no security — while any release editing a tool's description or schema would flip `config_changed=True` and downgrade stored decisions back to `ask`, re-prompting users at upgrade time for nothing.

Note the sequencing: in P1 the hash path is **dormant anyway**, because the hash is only compared against a stored tool-level entry and §6 writes none. Omitting it is what makes persistence *safe to add later* — the decision is load-bearing the moment the follow-up UI introduces stored built-in entries, which is precisely why it is settled here rather than there.

The new origin value (`builtin_default`) degrades gracefully in existing consumers — `mcp_inspector.py:1206` uses `_ORIGIN_SENTENCES.get(effective.origin, _UNKNOWN_ORIGIN_SENTENCE)` and `mcp_permissions_mode.py:175` only equality-checks `"tool_override"` — but it gets its own sentence.

### 4. Run-level review hook (restructure)

**The current hook is MCP-conditional and cannot simply be generalized.** `_compose_mcp_provider()` returns `(None, None)` when MCP is not eligible (`console_chat_controller.py:1065`, `:976`) and the controller passes `review_tool_calls=mcp_review_hook` (`:3318`). For a user with no MCP servers — the common case — there is **no review hook at all**, and MCP's per-call `_approval_callback` is provider-bound too, so there would be no approval route whatsoever.

P1 therefore composes the hook at **run level, independent of MCP**:

```
build_tool_review_hook(builtin_gate, mcp_provider_or_none, request_approvals)
```

Always built (built-ins always exist); MCP's per-call gate is folded in when a provider is present, preserving today's single-approval-card-per-turn UX across both sources. `ConsoleChatController.request_mcp_approvals` is already provider-agnostic (it marshals a batch to the UI and polls an `Event`) and is reused as-is.

**Classifying a call.** The hook routes each `ToolCall` by owner: a name the MCP provider claims (its existing `pending_gate_for` path) goes to the MCP gate; otherwise, if the run's `ToolCatalogRegistry` resolves the name to the built-in provider, it goes to `builtin_gate`. Names owned by neither (skills, native spawn) are returned unreviewed, exactly as today. Skill tools already route around `invoke_tool` entirely into their own budget-clamped nested loop and are not gated here.

**Card rows.** Built-in rows reuse the existing dict shape (`llm_name`, `server_key`, `tool_name`, `server_label`, `arguments`, `reason`) with `server_key="builtin:tldw_chatbook"` and `server_label="Built-in"`. `ChatApprovalCard` never interprets these — it groups by `llm_name` and renders the label as row-header text — so no card changes are needed for grouping or display. Its `_REASON_SUFFIXES` already knows `"risk_floored"`, which is the only reason built-ins can produce in P1.

### 5. Two enforcement points

- **Pre-dispatch** (the run-level hook): batches everything needing approval into one card per turn.
- **`BuiltinToolProvider.invoke`**: kill switch → resolved state → stamped verdict. Defense-in-depth, so a caller that skips the hook still cannot execute ungated.

Denials return `ToolResult(ok=False, error=...)` — **never a raise**. `run_agent_loop` is a pure loop that must not see exceptions from tool invocation.

When state is `ask` and no approval route is available (no stamped verdict and no callback), `invoke` **fails closed**: `ToolResult(ok=False, error="tool requires approval")`. In P1 this is unreachable (nothing resolves to `ask`); it is specified now because P2 makes it live.

**Layering:** `Agents/tool_catalog.py` stays dependency-light — the gate is a callable injected into `BuiltinToolProvider`:

```python
BuiltinGate = Callable[[Tool], EffectiveToolState]   # kill switch surfaces as state "deny"
BuiltinToolProvider(gate: BuiltinGate | None = None)
```

`gate=None` means "build the real gate lazily on first use" (importing `MCP/` inside the function, not at module scope, to avoid a startup-cost and circular-import hazard) — **not** "ungated". A bare `BuiltinToolProvider()`, which is how it is constructed today at `console_agent_bridge.py:756-757` and `:865`, must therefore be gated by default; tests inject an explicit permissive or scripted gate. Taking the whole `Tool` (not just its name) lets the adapter read `description`/`parameters`/`risk_tags` without a second lookup.

### 6. Session-scoped decisions only

P1 offers **approve-once** and **approve-for-session** (`UnifiedMCPControlPlaneService.approve_for_session` / `is_session_approved`). It does **not** write persistent allow/deny for built-ins.

Rationale: the approval card's persistent options call `set_tool_state`, but the MCP workbench renders servers from the live MCP catalog, so a synthetic `builtin:` key will not appear there. A persistent "Deny" would brick a built-in tool with no in-app way to undo it short of hand-editing JSON. Restricting P1 to session scope also means nothing is persisted, which independently keeps the omitted hash check (§3) inert.

**The card must not offer choices it will not honor.** `ChatApprovalCard`'s four options are module-level (`_DECISION_OPTIONS`) and applied to every row. P1 adds an optional per-row `options` key to `set_batch`'s call dicts: rows that omit it keep the full four (MCP behavior byte-identical), while built-in rows pass only `approve_once` and `approve_session`. Silently remapping a user's "Always allow" to a session approval would be a dishonest UI and is explicitly rejected.

Persistent decisions ship with the built-in permissions UI — filed as a follow-up, and a prerequisite for P2 offering them.

### 7. Kill switch becomes global

`BuiltinToolProvider.invoke` honors the existing kill switch, and the MCP workbench's label and echo text (`UI/MCP_Modules/mcp_workbench.py:1273-1289`, `:1678-1689`) change so the toggle no longer reads as MCP-only. One switch stops all tool execution — the behavior a user in trouble expects.

## Interaction with the task-327 tool-call timeout

`_call_with_timeout` (`Agents/agent_service.py`) wraps `registry.invoke_by_name` with `RunBudget.max_tool_call_seconds` (default 300.0), polling cancellation every 0.5s. Any approval wait inside `BuiltinToolProvider.invoke` therefore happens **inside** that wrapper and inherits its documented hazard: if the timeout fires first, the agent is told the call failed while the abandoned worker thread keeps waiting — and a late approval would execute the tool for real, with a retry executing it twice.

**Requirement:** the built-in approval timeout must stay comfortably below `max_tool_call_seconds`. MCP's approval default is 120s + 1s poll slack against the 300s ceiling; built-ins reuse the same bound. Any future change to either value must preserve that ordering, and the constant carries a comment saying so.

## Sub-agents (specified now, load-bearing for P2)

`clamp_child_budget` zeroes `max_subagents` but a child **inherits the parent's allow-list** minus spawn/skill names (`agent_service.py:560-576`), so a child can call built-in tools. The review hook is per-run, so a child's gated call would fall through to the per-call approval path from inside a nested run on an already-blocked worker thread.

In P1 this is unreachable (nothing resolves to `ask`). P1's required behavior: a child run with no approval route **fails closed** per §5. P2 must not ship gated mutating tools until child-run approval is resolved — either by threading the parent's approval route into children, or by excluding `"mutates"` tools from child allow-lists.

## Acceptance criteria

- [ ] `Tool` ABC exposes a concrete `risk_tags` property defaulting to `()`; no existing subclass breaks.
- [ ] A built-in tool resolves through `resolve_effective_state` via a `HubTool` adapter using `server_key="builtin:tldw_chatbook"`, with precedence tool override → server default → built-in `allow` floor, and **no** definition-hash comparison.
- [ ] An untagged built-in (calculator, datetime) resolves to `allow` and executes with no prompt — verified as a no-behavior-change test.
- [ ] A tool tagged `"mutates"` has its inherited allow floored to `ask` (`risk_floored=True`).
- [ ] `BuiltinToolProvider.invoke` blocks execution when the kill switch is on, when the resolved state is `deny`, and when the state is `ask` with no approval route — each returning `ToolResult(ok=False, ...)`, never raising.
- [ ] A run-level `build_tool_review_hook` is wired **whether or not** MCP is configured, and folds MCP's gate in when a provider exists; built-in and MCP approvals appear in one card per turn.
- [ ] `resolve_builtin_state` exists as a sibling of `resolve_effective_state`; the MCP resolver's behavior is unchanged (its existing tests still pass untouched).
- [ ] `BuiltinToolProvider(gate=None)` is gated by default — a test asserts a bare instance refuses a `deny`-resolved tool, proving `None` does not mean "ungated".
- [ ] The review hook routes calls by owner: MCP-claimed names to the MCP gate, registry-resolved built-ins to the built-in gate, and everything else (skills, native spawn) unreviewed.
- [ ] Built-in approvals are session-scoped only; no persistent allow/deny is written for `builtin:` keys in P1.
- [ ] The approval card offers built-in rows only the session-scoped options; MCP rows keep all four, verified byte-identical.
- [ ] The kill switch blocks built-in tools, and its MCP workbench label/echo no longer describe it as MCP-only.
- [ ] Tests inject a temporary permission-store path; no test touches the real user store.

## Out of scope

- Porting fs/note tools into `BuiltinToolProvider` (P2).
- `[tools]` config / TASK-547 / the legacy System A decision (P3).
- A settings UI for built-in tool permissions (follow-up; gates persistent decisions).
- Changing `ToolExecutor` or the legacy chat path in any way.

## Follow-ups to file

1. Built-in tool permissions UI — surface `builtin:tldw_chatbook` in the workbench (or a Tools settings pane) so persistent allow/deny becomes safe to offer.
2. Child-run approval routing — prerequisite for P2 shipping gated mutating tools.
3. `local_file_ingestion.py:1148` `get_cli_setting("database", {})` — the second instance of TASK-547's bug, found during investigation.
