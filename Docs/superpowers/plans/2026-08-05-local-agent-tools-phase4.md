# Local Agent Tools — Phase 4 (MCP Exposure) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the local agent tool set (`fs_*`, `fs_patch`, `web_*`, `git_*`) to external MCP clients through `MCP/server.py`, routed through `LocalToolProvider`'s permission gate — never wrapping cores directly (fail-open, per the re-plan spec).

**Architecture:** A server-side composition module (`MCP/local_server_tools.py`) builds a `LocalToolProvider` for the non-Console process: permission state resolved fresh per call from `MCPPermissionStore`, kill switch honored, NO approval callback (ask fails closed), and an external-appropriate no-callback refusal string. `TldwMCPServer` registers the provider's catalog on FastMCP behind a new `[mcp] expose_local_tools` config flag (default false). `todo_write` is NOT exposed (Console-session-scoped state; nothing external to render into — disclosed deviation).

**Spec:** `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md` §3.1 (including the two "implementation facts" — binding) · **ADRs:** 032, 033

## Verified facts (do not re-derive)

- `TldwMCPServer` (`MCP/server.py:102`) registers tools via `@self.mcp.tool()` decorators on async inner functions in `_register_tools` (:168+); `FastMCP` import is conditional (`MCP_AVAILABLE`).
- `MCPPermissionStore(path)` (`MCP/permission_store.py:149`) has `.load() -> dict` and `.get_kill_switch() -> bool`; the canonical path is derived like `Path(store.path).with_name("mcp_permissions.json")` (`unified_control_plane_service.py:2429` — read how `store` is obtained there and mirror it, or find the simpler canonical location the store uses).
- `resolve_effective_state(payload, hub_tool)` (`permission_store.py:516`) gives the tool→server→global resolution + risk floor; `LocalToolProvider.hub_tool_for(name)` produces the HubTool.
- `LocalToolProvider` seams: `resolve_state`, `kill_switch`, `approval_callback` (None = fail closed with `LOCAL_TIMEOUT_REFUSAL` — misleading copy for external clients that can never approve; Task 1 adds an override), `record_decision`.
- Operator grant path (re-plan §3.1 fact 2): Console "Always allow" persists allow+definition_hash under `local:__local__`; explicit tool-level allow is never risk-floored — that's how an operator enables external use.
- FastMCP tool names must match `[a-zA-Z0-9_-]{1,64}` — `fs_read` etc. are fine.
- `LocalToolProvider` `invoke()` is sync; FastMCP tools are async — call the sync invoke directly (it's a worker-thread-safe pure function; the server has no approval round-trip to block on).
- Run tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` from the worktree. MCP server tests live in `Tests/MCP/` — follow their fixtures/marking (the `mcp` package may be optional; check how existing server tests handle MCP_AVAILABLE).

---

## Task 0: Backlog task

- [ ] Create "Local agent tools phase 4: MCP exposure". ACs:
  1. External MCP clients can call allowed local tools through the server (invocation routed through LocalToolProvider's gate)
  2. ask-state tools fail closed externally with an external-appropriate refusal (no approval card exists outside the Console)
  3. Kill switch and deny states honored; operator grants (always-allow from Console) enable external use
  4. Exposure gated behind [mcp] expose_local_tools (default false); todo_write not exposed (documented)
  5. All new tests pass
  Commit: `docs: create phase-4 backlog task`

---

## Task 1: Provider no-callback refusal override + server composition

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (optional `no_callback_refusal` param)
- Create: `tldw_chatbook/MCP/local_server_tools.py`
- Test: `Tests/Agents/test_local_tool_provider.py`, `Tests/MCP/test_local_server_tools.py`

- [ ] **Step 1: Failing tests**
  - Provider: `LocalToolProvider(..., no_callback_refusal="custom")` returns "custom" on ask+no-callback; default unchanged (`LOCAL_TIMEOUT_REFUSAL` — pinned, existing tests keep passing).
  - Composition: `build_server_local_provider(workspace_root, permission_store)` returns a provider whose resolve_state resolves through a REAL temp `MCPPermissionStore` (set an allow → executes; default → fails closed); kill switch on → kill-switch refusal; `None`/missing store file → fail closed (ask default).
- [ ] **Step 2: Implement**
  - Provider: new optional constructor param `no_callback_refusal: str | None = None`; the `no_callback` verdict maps to it when set (keep `LOCAL_TIMEOUT_REFUSAL` as the default; the timeout verdict keeps the pinned string).
  - `MCP/local_server_tools.py`:

```python
EXTERNAL_NO_CALLBACK_REFUSAL = (
    "tool requires operator approval (permission state is 'ask' and external "
    "MCP clients cannot approve); an operator must grant 'allow' for this "
    "tool in the Console or permission store"
)

def build_server_local_provider(
    workspace_root: Path, permission_store: Any
) -> LocalToolProvider:
    """Compose a LocalToolProvider for non-Console (external) MCP serving.

    resolve_state loads the store payload FRESH per call (operator changes
    take effect immediately); approval_callback is None (fail closed);
    no_callback_refusal is EXTERNAL_NO_CALLBACK_REFUSAL; kill_switch from
    the store. Follows _compose_local_provider's discipline minus the
    Console-only seams (session approvals, persist, todo store).
    """
```

- [ ] **Step 3:** tests pass
- [ ] **Step 4:** `git commit -m "feat: server-side local provider composition + external refusal override"`

---

## Task 2: `[mcp] expose_local_tools` config

**Files:**
- Modify: `tldw_chatbook/config.py` (mcp section coercion + template — find the `[mcp]` section's existing handling; follow the console pattern from Task 4 of phase 1)
- Test: the mcp config test file (find with `grep -rln "\[mcp\]\|get_cli_setting(\"mcp\"" Tests/ | head`)

- [ ] **Step 1: Failing test** — default False; `"yes"` → True.
- [ ] **Step 2: Implement** (bool coercion + commented template line: `# expose_local_tools = false   # expose workspace-local agent tools (fs_*/git_*/web_*) to external MCP clients; permission-gated, writes effectively denied until granted`).
- [ ] **Step 3:** tests pass
- [ ] **Step 4:** `git commit -m "feat: [mcp] expose_local_tools config flag"`

---

## Task 3: Server registration

**Files:**
- Modify: `tldw_chatbook/MCP/server.py`
- Test: `Tests/MCP/test_local_server_tools.py` (extend)

- [ ] **Step 1: Failing tests**
  - With the flag on and a temp permission store granting `fs_read`: the server has a registered `fs_read` tool whose invocation returns file contents (call the registered function directly — follow how existing Tests/MCP server tests invoke tools).
  - `fs_write` (default ask) → the EXTERNAL_NO_CALLBACK_REFUSAL text.
  - Deny state on `fs_glob` → `LOCAL_DENY_REFUSAL`; kill switch → `LOCAL_KILL_SWITCH_REFUSAL`.
  - Flag off (default) → no `fs_*`/`git_*`/`web_*` tools registered (existing tools unaffected).
  - `todo_write` is NOT in the registered set even when the provider would offer it (the composition must not pass a todo_store; assert its absence).
- [ ] **Step 2: Implement** — in `TldwMCPServer.__init__` (or a `_register_local_agent_tools()` called from it): when `[mcp] expose_local_tools` is true, build the permission store (canonical path per the verified facts), `build_server_local_provider`, then for each `list_catalog()` entry register an async FastMCP tool named after the entry, with the provider's `load_schema` description/parameters, whose body calls `provider.invoke(tool_id, args)` and returns `result.content` or raises/returns the error string per FastMCP conventions (check how existing tools in server.py return errors — follow that convention). Workspace root: `[console] workspace_root` or cwd, same rule as the Console.
- [ ] **Step 3:** tests pass; `pytest Tests/MCP/ -q` for regressions
- [ ] **Step 4:** `git commit -m "feat: expose local agent tools through MCP server (permission-gated)"`

---

## Task 4: Docs + close-out (controller-led)

- [ ] Docs: `tldw_chatbook/MCP/server.py` module docstring (or README if present) gains a section on the exposed local tools: the flag, the permission model (external ask fails closed; grant via Console always-allow), the todo_write omission.
- [ ] Backlog task: ACs, Implementation Notes, Done.
- [ ] Final review subagent; superpowers:finishing-a-development-branch.
