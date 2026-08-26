# MCP Fewer Permission Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Chatbook-native `/fewer-permission-prompts` command that reports safe, local MCP permission recommendations from recent approval history.

**Architecture:** Add a pure MCP recommendation module that consumes recent execution-log rows, live `HubTool` definitions, and resolved permission states. Wire it into `UnifiedMCPControlPlaneService` for catalog-aware report and apply APIs, then expose the report through the existing Console slash-command path.

**Tech Stack:** Python 3.11+, Textual, pytest, existing MCP `ExecutionRecord`, `HubTool`, `EffectiveToolState`, and `MCPPermissionStore` APIs.

## Global Constraints

- V1 is MCP-only; no bash or shell command allowlist recommendations.
- No telemetry, analytics, remote upload, or new tracking store.
- Recommendations use the existing local redacted MCP execution log and existing permission store.
- Persistent allow changes must go through `UnifiedMCPControlPlaneService.set_tool_state(..., "allow", tool=tool)`.
- Auto mode and model-based auto-approval are deferred.
- ADR required: yes
- ADR path: `backlog/decisions/081-mcp-prompt-reduction-recommendations.md`
- Reason: This touches local permission recommendations, persisted security state, privacy boundaries, and Console/MCP service contracts.

---

### Task 1: Pure Recommendation Engine

**Files:**
- Create: `tldw_chatbook/MCP/permission_prompt_reducer.py`
- Test: `Tests/MCP/test_permission_prompt_reducer.py`

**Interfaces:**
- Consumes: `HubTool`, `EffectiveToolState`, raw execution-log row mappings.
- Produces:
  - `PermissionPromptRecommendation`
  - `PermissionPromptReport`
  - `build_permission_prompt_report(records, tools, states, min_approved_count=2) -> PermissionPromptReport`

- [ ] **Step 1: Write failing tests**

```python
def test_recommends_repeated_agent_approved_ask_gated_tool():
    tool = _tool("local:docs", "search")
    records = [
        _record("local:docs", "search", "approved"),
        _record("local:docs", "search", "approved"),
    ]
    states = {("local:docs", "search"): EffectiveToolState(state="ask", origin="global_default")}

    report = build_permission_prompt_report(records, [tool], states)

    assert [(r.server_key, r.tool_name, r.approved_count) for r in report.recommendations] == [
        ("local:docs", "search", 2)
    ]
```

Also cover these breaks: below threshold is excluded, already allowed is excluded, denied is excluded, config-changed/risk-floored ask states are excluded, missing live tool is excluded, and recommendations sort by approved count then last seen.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/tmp/tldw_test_stubs:$PWD /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_permission_prompt_reducer.py -q
```

Expected: import or attribute failure because `permission_prompt_reducer.py` does not exist.

- [ ] **Step 3: Implement minimal engine**

Create dataclasses and grouping logic. Only records with `decision == "approved"` and `initiator == "agent"` count. Match records to live tools by `(server_key, tool_name)`. Recommend only when `EffectiveToolState.state == "ask"` and both `config_changed` and `risk_floored` are false.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command. Expected: all reducer tests pass.

### Task 2: Control Plane Report And Apply APIs

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Test: `Tests/MCP/test_control_plane_prompt_reducer.py`

**Interfaces:**
- Consumes: `build_permission_prompt_report(...)`.
- Produces:
  - `async def permission_prompt_recommendations(self, *, min_approved_count: int = 2, limit: int = 200) -> PermissionPromptReport`
  - `async def apply_permission_prompt_recommendation(self, server_key: str, tool_name: str, *, min_approved_count: int = 2) -> PermissionPromptRecommendation`

- [ ] **Step 1: Write failing tests**

Test a fake local service with a real `LocalMCPStore`, one local profile record with discovered tools, and appended `approved` execution-log records. Assert the service returns one recommendation and `apply_permission_prompt_recommendation` persists a tool-level allow with a definition hash.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/tmp/tldw_test_stubs:$PWD /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_control_plane_prompt_reducer.py -q
```

Expected: missing service methods.

- [ ] **Step 3: Implement service methods**

Add a private async helper to collect local MCP tools from `local_external_catalog()` plus built-in inventory. Read `self.execution_log.read_recent(limit)` defensively. Reuse `effective_tool_states(tools)` and `set_tool_state(server_key, tool_name, "allow", tool=tool)`.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command. Expected: all control-plane prompt-reducer tests pass.

### Task 3: Console Slash Command

**Files:**
- Modify: `tldw_chatbook/Chat/console_command_grammar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Chat/test_console_command_grammar.py`
- Test: focused Console command handler tests if existing fakes can exercise the method without mounting the full screen.

**Interfaces:**
- Consumes: `UnifiedMCPControlPlaneService.permission_prompt_recommendations`.
- Produces: `/fewer-permission-prompts` command with handler id `fewer-permission-prompts`.

- [ ] **Step 1: Write failing grammar test**

```python
def test_fewer_permission_prompts_command_is_builtin():
    registry = default_console_registry()

    assert registry.parse("/fewer-permission-prompts") == CommandParse(
        "command", "fewer-permission-prompts", ""
    )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/tmp/tldw_test_stubs:$PWD /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_command_grammar.py -q
```

Expected: parser returns unknown for the new command.

- [ ] **Step 3: Implement grammar and handler**

Register command constants in `console_command_grammar.py`. In `chat_screen.py`, dispatch to `_console_command_fewer_permission_prompts`, call the service, append a local system message report, and clear the composer draft only when the command is handled.

- [ ] **Step 4: Verify GREEN**

Run grammar tests and any focused handler test. Expected: pass.

### Task 4: Closeout

**Files:**
- Modify: `backlog/tasks/task-21162 - Add-MCP-fewer-permission-prompts-recommendations.md`
- Verify: focused pytest and `git diff --check`

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=/tmp/tldw_test_stubs:$PWD /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_permission_prompt_reducer.py Tests/MCP/test_control_plane_prompt_reducer.py Tests/Chat/test_console_command_grammar.py -q
```

- [ ] **Step 2: Run diff whitespace check**

```bash
git diff --check
```

- [ ] **Step 3: Update Backlog task**

Check off acceptance criteria and add concise implementation notes with ADR path, test commands, and modified files.
