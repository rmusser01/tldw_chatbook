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
- **FastMCP derives input schemas from Python type annotations, NOT arbitrary JSON** — `provider.load_schema().parameters` cannot be fed to `@mcp.tool()` for dynamic tools. Decision (plan review): register each local tool with a generic `arguments: dict` signature; handler-side validation is fail-safe because `invoke()` never raises and handler exceptions become error `ToolResult`s. Error returns follow the server.py convention: `{"error": str}` dicts (`server.py:187`, `:209`).
- **The `mcp` package is NOT installed in the repo venv** (`MCP_AVAILABLE` is False there; no test in `Tests/MCP/` ever instantiates `TldwMCPServer`). Decision (plan review): registration logic lives in a pure, FastMCP-free builder — `_local_agent_tool_registrations(provider) -> list of (name, description, parameters, async handler)` — fully testable without the `mcp` package; the thin FastMCP binding layer is covered by `pytest.mark.skipif(not MCP_AVAILABLE)` tests only. Do NOT install the `mcp` extra into the venv for this phase.
- Permission store canonical path (pinned by plan review): `get_user_data_dir() / "mcp_permissions.json"` — this is where Console "Always allow" grants actually land (`app.py:4544` builds the local store at `get_user_data_dir() / "local_mcp_store.json"`; `unified_control_plane_service.py:2429` derives the permissions file next to it). Any other path silently breaks the operator-grant path.
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
- Modify: `tldw_chatbook/config.py` — NOTE (plan review): the `[mcp]` template section is at config.py:3307 followed by a `[mcp.tools]` sub-table at :3327, so the commented template line MUST go above :3327 or it lands in the wrong TOML table. There is NO typed coercion block for `[mcp]` yet — add one (a new small block, following the `[console]` pattern at config.py:786), not an extension of an existing one.
- Test: the mcp config test file (find with `grep -rln "\[mcp\]\|get_cli_setting(\"mcp\"" Tests/ | head`)

- [ ] **Step 1: Failing test** — default False; `"yes"` → True.
- [ ] **Step 2: Implement** (bool coercion in a new `[mcp]` coercion block + commented template line above the `[mcp.tools]` sub-table: `# expose_local_tools = false   # expose workspace-local agent tools (fs_*/git_*/web_*) to external MCP clients; permission-gated, writes effectively denied until granted`).
- [ ] **Step 3:** tests pass
- [ ] **Step 4:** `git commit -m "feat: [mcp] expose_local_tools config flag"`

---

## Task 3: Server registration

**Files:**
- Modify: `tldw_chatbook/MCP/server.py`
- Test: `Tests/MCP/test_local_server_tools.py` (extend)

- [ ] **Step 1: Failing tests** (against the PURE builder — no `mcp` package needed):
  - With a temp permission store granting `fs_read`: `_local_agent_tool_registrations(provider)` includes an `fs_read` entry whose handler returns file contents.
  - `fs_write` (default ask) → handler returns `{"error": EXTERNAL_NO_CALLBACK_REFUSAL}` per the server.py error-dict convention.
  - Deny state on `fs_glob` → `{"error": LOCAL_DENY_REFUSAL}`; kill switch → `{"error": LOCAL_KILL_SWITCH_REFUSAL}`.
  - `todo_write` is NOT among the registrations (the composition passes no todo_store; assert absence).
  - Skip-if-unavailable binding test (`pytest.mark.skipif(not MCP_AVAILABLE)`): with the flag on, `TldwMCPServer._register_local_agent_tools()` registers the same names on FastMCP; with the flag off (default), no `fs_*`/`git_*`/`web_*` tools are registered and existing tools are unaffected.
- [ ] **Step 2: Implement** — two layers:
  1. In `MCP/local_server_tools.py`: `_local_agent_tool_registrations(provider) -> list[LocalToolRegistration]` where each registration is `(name, description, parameters, handler)`; `handler(arguments: dict)` (async or sync — pick per the binding layer's needs) calls `provider.invoke(tool_id, arguments)` and returns `result.content` on ok or `{"error": result.error}` on failure. The provider's `load_schema` supplies description/parameters (kept for future SDK versions / introspection, even though the FastMCP layer can't consume the JSON schema directly today).
  2. In `MCP/server.py`: `_register_local_agent_tools()` called from `__init__` (NOT inside `_register_tools` — keeps the AST-walking `_extract_registered_entries` catalog unaffected). When `[mcp] expose_local_tools` is true: build `MCPPermissionStore(get_user_data_dir() / "mcp_permissions.json")`, `build_server_local_provider(workspace_root)` (root = `[console] workspace_root` or cwd, same rule as Console), then for each registration bind a FastMCP tool with a generic `arguments: dict` signature that delegates to the handler. When the flag is false (default), the method is a no-op.
- [ ] **Step 3:** tests pass; `pytest Tests/MCP/ -q` for regressions
- [ ] **Step 4:** `git commit -m "feat: expose local agent tools through MCP server (permission-gated)"`

---

## Task 4: Docs + close-out (controller-led)

- [ ] Docs: `tldw_chatbook/MCP/server.py` module docstring (or README if present) gains a section on the exposed local tools: the flag, the permission model (external ask fails closed; grant via Console always-allow), the todo_write omission.
- [ ] Backlog task: ACs, Implementation Notes, Done.
- [ ] Final review subagent; superpowers:finishing-a-development-branch.
