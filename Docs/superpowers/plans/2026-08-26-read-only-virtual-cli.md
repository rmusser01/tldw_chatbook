# Read-Only Virtual CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose one model-only `virtual_cli` structured tool whose ten read-only commands reuse Chatbook's existing workspace/Git safety cores while each command retains an independent Allow/Ask/Off permission.

**Architecture:** A small `VirtualCliRegistry` validates the outer `{command, argv}` payload, parses only documented argv forms, and dispatches directly to existing `local_tool_impls`/`git_tool_impls` functions—never to a shell. `VirtualCliProvider` presents one model schema but projects ten synthetic `HubTool` permission records under `local:__virtual_cli__`. Review and invocation preserve native tool-call identity so two `virtual_cli` calls in one batch can receive different decisions. The existing local tool gate, kill switch, workspace binding, path resolver, sensitive-path denylist, Git exclusions, caps, and result handling remain authoritative.

**Tech Stack:** Python 3.11, standard-library `argparse` with a non-exiting error adapter, existing Agent `ToolProvider`/catalog APIs, MCP permission store and approval cards, Textual Tools workbench, pytest.

**Backlog task:** `TASK-22509`

**ADR required:** yes

**ADR path:** `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md`

**Reason:** ADR-094 establishes the model-only no-shell boundary, reserved synthetic principal, independent command permissions, and default-Ask behavior.

**Prerequisite:** None. This plan is independent of the raw executor and may be implemented before or after TASK-18926.

---

## Task 1: Define the fixed virtual command grammar and direct dispatch registry

**Files:**

- Create: `tldw_chatbook/Tools/virtual_cli_impls.py`
- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Create: `Tests/Tools/test_virtual_cli_impls.py`
- Modify: `Tests/Tools/test_local_tool_impls.py`
- Modify: `Tests/Tools/test_git_tool_impls.py`

- [ ] **Step 1: Write failing schema, argv, and dispatch tests**

Pin the v1 command set exactly:

```python
VIRTUAL_CLI_COMMANDS = (
    "ls", "cat", "grep", "find", "stat",
    "git_status", "git_diff", "git_log", "git_blame", "git_branches",
)
```

Pin these accepted forms:

```text
ls [PATH]
cat PATH [--offset N] [--limit N]
grep PATTERN [--mode content|files|count]
find GLOB
stat PATH
git_status [PATH]
git_diff [--staged] [--range REF] [--path PATH] [--stat]
git_log [--count N] [--path PATH]
git_blame PATH [--start N] [--end N]
git_branches
```

Test unknown commands, unknown/abbreviated flags, missing/extra positionals, non-string argv items, more than 64 items, per-item text over 4 KiB, aggregate UTF-8 size over 16 KiB, NUL, invalid integers, and shell-looking values. Shell metacharacters inside a positional string remain literal input to the selected core; they are never interpreted as pipes, redirects, substitutions, globs beyond `find`'s explicit pattern, or expansion.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Tools/test_virtual_cli_impls.py Tests/Tools/test_local_tool_impls.py Tests/Tools/test_git_tool_impls.py -k virtual_cli
```

Expected: FAIL because the registry and `stat` core do not exist.

- [ ] **Step 3: Add the read-only `stat_path` core through the existing choke point**

In `local_tool_impls.py`, add:

```python
def stat_path(path: str, *, workspace_root: Path) -> str:
    resolved = resolve_workspace_path(path, workspace_root, intent="read")
    info = resolved.stat()
    kind = "directory" if resolved.is_dir() else "file" if resolved.is_file() else "other"
    return "\n".join((
        f"path: {resolved.relative_to(workspace_root.resolve())}",
        f"type: {kind}",
        f"size: {info.st_size}",
        f"modified_ns: {info.st_mtime_ns}",
        f"mode: {info.st_mode & 0o7777:04o}",
    ))
```

Return only allowlisted fields; do not expose uid/gid owner names, extended attributes, symlink targets outside the root, or platform-specific credential metadata. Resolve and denylist-check before `stat()`.

- [ ] **Step 4: Implement a non-exiting command parser**

Use one tiny `ArgumentParser` subclass whose `error()` raises `VirtualCliArgumentError` and configure `add_help=False`, `allow_abbrev=False`, `exit_on_error=False`. Build one parser per command at module load; no third-party parser dependency and no shell-string parser.

Expose:

```python
@dataclass(frozen=True, slots=True)
class VirtualCliRequest:
    command: VirtualCliCommand
    argv: tuple[str, ...]

class VirtualCliRegistry:
    def __init__(self, workspace_root: Path): ...
    def execute(self, command: str, argv: Sequence[str]) -> str: ...
```

`execute()` validates the outer shape first, parses the selected argv second, and then directly calls exactly one of `list_directory`, `read_file`, `grep_files`, `glob_files`, `stat_path`, `git_status`, `git_diff`, `git_log`, `git_blame`, or `git_branches`.

- [ ] **Step 5: Prove existing security cores remain authoritative**

Add parameterized tests that invoke the virtual aliases against:

- `..`/absolute outside-root paths;
- symlinks escaping the workspace;
- sensitive credential/config/permission-store paths;
- a repository containing denylisted paths in status/diff output;
- oversized directory/grep/glob/Git outputs.

Assert failures and caps match the direct core behavior. Do not duplicate path validation or Git subprocess code in `virtual_cli_impls.py`.

- [ ] **Step 6: Run the focused implementation tests**

```bash
pytest -q Tests/Tools/test_virtual_cli_impls.py Tests/Tools/test_local_tool_impls.py Tests/Tools/test_git_tool_impls.py -k "virtual_cli or stat_path"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Tools/virtual_cli_impls.py tldw_chatbook/Tools/local_tool_impls.py Tests/Tools/test_virtual_cli_impls.py Tests/Tools/test_local_tool_impls.py Tests/Tools/test_git_tool_impls.py
git commit -m "feat: add read-only virtual CLI registry"
```

## Task 2: Reserve `__virtual_cli__` at every external-profile seam

**Files:**

- Modify: `tldw_chatbook/MCP/local_store.py`
- Modify: `tldw_chatbook/MCP/hub_tool_catalog.py`
- Modify: `Tests/MCP/test_local_store.py`
- Modify: `Tests/MCP/test_hub_tool_catalog.py`

- [ ] **Step 1: Write failing reservation tests**

Mirror every existing `__local__` test with `__virtual_cli__`, including whitespace-normalized create/save rejection and filtering of hand-edited records during load/catalog projection. Include a control profile such as `__virtual_cli__x` to prove only the exact reserved id is blocked.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/MCP/test_local_store.py Tests/MCP/test_hub_tool_catalog.py -k virtual_cli
```

Expected: FAIL.

- [ ] **Step 3: Replace the singular reserved constant with a shared set**

Keep the change small:

```python
_RESERVED_EXTERNAL_PROFILE_IDS = frozenset({"__local__", "__virtual_cli__"})
```

Apply exact normalized membership at create/update validation, persisted profile load, runtime-state load, tool-record load, and external catalog projection. Internal synthetic HubTools may still use `local:__virtual_cli__`; only external profile spoofing is rejected.

- [ ] **Step 4: Run the reservation tests**

```bash
pytest -q Tests/MCP/test_local_store.py Tests/MCP/test_hub_tool_catalog.py -k "local or virtual_cli"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/local_store.py tldw_chatbook/MCP/hub_tool_catalog.py Tests/MCP/test_local_store.py Tests/MCP/test_hub_tool_catalog.py
git commit -m "security: reserve the virtual CLI permission principal"
```

## Task 3: Implement one model schema with ten independent permission identities

**Files:**

- Create: `tldw_chatbook/Agents/virtual_cli_provider.py`
- Modify: `tldw_chatbook/Agents/run_context.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Create: `Tests/Agents/test_virtual_cli_provider.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/Agents/test_agent_service.py`

- [ ] **Step 1: Write failing provider tests**

Pin:

- `list_catalog()` exposes exactly one `virtual_cli` model tool;
- its schema requires `{command, argv}`, uses the fixed command enum, and has no shell-string field;
- `hub_tools()` exposes ten records under `local:__virtual_cli__`, one per command, with stable distinct definition hashes;
- missing permission resolves to Ask;
- Allow/Ask/Off for `ls` has no effect on `cat`;
- global kill switch and local-tools disabled state fail closed;
- invalid command/argv fails before permission lookup/dispatch;
- `invoke()` re-resolves permission after review;
- control characters in core results are stripped and result bytes retain the existing local-tool cap;
- no code path calls `subprocess` except the existing Git cores.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_agent_service.py -k "virtual_cli or tool_call_identity"
```

Expected: FAIL.

- [ ] **Step 3: Carry call identity into generic provider invocation**

The review runtime already resolves verdict maps by `call_id` before tool name. Extend `Agents/run_context.py` with a second ContextVar:

```python
def current_tool_call_id() -> str: ...

@contextmanager
def use_tool_call_identity(run_id: str, call_id: str) -> Iterator[None]:
    with use_run_id(run_id):
        token = _CURRENT_TOOL_CALL_ID.set(call_id or "")
        try:
            yield
        finally:
            _CURRENT_TOOL_CALL_ID.reset(token)
```

In `AgentService._make_invoke_tool`, bind `call.call_id` around `registry.invoke_by_name`. Preserve `current_run_id()` behavior and add nested/concurrent context tests. Do not widen every `ToolProvider.invoke` signature.

- [ ] **Step 4: Implement `VirtualCliProvider`**

Use constants:

```python
VIRTUAL_CLI_TOOL_NAME = "virtual_cli"
VIRTUAL_CLI_SERVER_KEY = "local:__virtual_cli__"
VIRTUAL_CLI_SERVER_LABEL = "Virtual CLI (read-only)"
```

The provider owns one `VirtualCliRegistry`, one `HubTool` descriptor per command, and stamp keys `(run_id, call_id or command)`. `pending_gate_for(call)` validates the selected command and returns an `MCPPendingCall` whose:

- `llm_name` remains `virtual_cli` for provider dispatch;
- `tool_name` is the selected virtual command;
- `server_key` is `local:__virtual_cli__`;
- `call_id` is copied from the model call;
- arguments show the selected command and argv.

`invoke()` obtains `current_run_id()` and `current_tool_call_id()`, revalidates command/argv, rechecks the kill switch and current command state, consumes only the matching stamp, then dispatches the registry. A missing call id falls back to `(run_id, command)` for the single fence-call path. Strip C0/C1 terminal controls (preserving newline/tab) from the direct core result and apply the same 32 KiB UTF-8 result cap as `LocalToolProvider`; Git already disables color, and no Rich markup interpretation is enabled.

- [ ] **Step 5: Keep permission persistence on the existing store**

Inject the same `resolve_state`, `is_session_approved`, `approve_for_session`, `persist_approval`, and audit callbacks used by `LocalToolProvider`, but call them with the selected command's HubTool. No second JSON file or permission service is added.

- [ ] **Step 6: Run provider and context tests**

```bash
pytest -q Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_agent_service.py -k "virtual_cli or tool_call_identity"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/virtual_cli_provider.py tldw_chatbook/Agents/run_context.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_agent_service.py
git commit -m "feat: gate virtual CLI commands independently"
```

## Task 4: Preserve per-call approval identity for repeated `virtual_cli` batches

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/Chat/test_console_virtual_cli_approval.py`
- Modify: `Tests/UI/test_console_mcp_approval.py`

- [ ] **Step 1: Write the failing repeated-call regression**

Build one native batch:

```python
calls = [
    ToolCall("virtual_cli", {"command": "ls", "argv": ["."]}, call_id="c1"),
    ToolCall("virtual_cli", {"command": "cat", "argv": ["README.md"]}, call_id="c2"),
]
```

Approve `c1`, deny `c2`, and assert only `ls` executes. Reverse decisions and assert only `cat` executes. Then approve `ls` for session and prove it does not approve `cat`. Pin that both full cards remain visible and independently addressable.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Chat/test_console_virtual_cli_approval.py Tests/UI/test_console_mcp_approval.py -k virtual_cli
```

Expected: FAIL at the current local-provider name-collapsing seam.

- [ ] **Step 3: Add a provider-specific review hook without weakening existing tools**

Add `build_virtual_cli_review_hook(provider, request_approvals)` beside `build_local_review_hook`. At entry, clear only this run's virtual stamps. Pass each full `ToolCall` to `pending_gate_for`, preserving `call_id`. Send all pending rows in one approval round trip. Apply decisions by `call_id` first, selected command second only for id-less fence calls. Persist `always_allow` and grant `approve_session` against the selected command's HubTool only; `approve_once` remains call-scoped. Return per-call refusal keys so the generic agent runtime prevents denied calls from reaching `invoke()`.

Do not change existing same-name approval semantics for ordinary MCP/local tools in this task except to fix a proven shared bridge defect.

- [ ] **Step 4: Make the approval UI key rows by call identity**

Audit `request_mcp_approvals` for any `llm_name` de-duplication. Its row/result maps must use `call_id or llm_name`; display labels may still group by source but must not merge actionable rows. Preserve compatibility for id-less fence calls.

- [ ] **Step 5: Run the repeated-call and existing approval suites**

```bash
pytest -q Tests/Chat/test_console_virtual_cli_approval.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_console_headless_approval.py
```

Expected: PASS with no regression to builtin/MCP approval behavior.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_virtual_cli_approval.py Tests/UI/test_console_mcp_approval.py
git commit -m "fix: keep repeated virtual CLI approvals per call"
```

## Task 5: Register the model tool by default under existing local-tool gates

**Files:**

- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Create: `Tests/Agents/test_virtual_cli_integration.py`
- Modify: `Tests/Agents/test_local_tools_integration.py`

- [ ] **Step 1: Write failing catalog/gate matrix tests**

Cover:

| Local tools | Global block-all | Expected catalog/result |
| --- | --- | --- |
| enabled | off | `virtual_cli` discoverable; selected command still Ask/Allow/Off gated |
| enabled | on | schema absent or invocation blocked; no dispatch |
| disabled | off/on | schema absent and invocation blocked |

Also prove discoverability at missing state does not execute: a direct invocation without a review stamp must return an Ask/refusal result.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Agents/test_virtual_cli_integration.py Tests/Agents/test_local_tools_integration.py -k virtual_cli
```

Expected: FAIL.

- [ ] **Step 3: Compose the provider at the same per-run boundary as local tools**

Create `VirtualCliProvider` from the active workspace/scratch binding and the same permission callbacks used by `LocalToolProvider`. Register it with the per-run `ToolCatalogRegistry`, add `virtual_cli` to allowed tools, and add `build_virtual_cli_review_hook` to `build_combined_review_hook`.

The root passed to `VirtualCliRegistry` must be the exact active local-tool root for that run. Read-only workspace bindings advertise the virtual tool because every v1 command is read-only; no raw CLI state is consulted.

- [ ] **Step 4: Recheck all gates at invocation**

Catalog construction is not authorization. The provider must call fresh local-tools/global-kill-switch/current-command resolution immediately before core dispatch so a setting changed after model schema creation fails closed.

- [ ] **Step 5: Run integration tests**

```bash
pytest -q Tests/Agents/test_virtual_cli_integration.py Tests/Agents/test_local_tools_integration.py -k virtual_cli
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py Tests/Agents/test_virtual_cli_integration.py Tests/Agents/test_local_tools_integration.py
git commit -m "feat: register virtual CLI for local agent runs"
```

## Task 6: Present independent virtual-command permissions in Tools

**Files:**

- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Modify: `Tests/UI/test_mcp_workbench.py`

- [ ] **Step 1: Write mounted Tools workbench tests**

Assert the canonical Tools destination shows a distinct `Virtual CLI (read-only)` group with ten rows, each cycling Allow → Ask → Off independently and persisting under `local:__virtual_cli__::<command>`. Assert changing `cat` does not mutate `fs_read` or any other virtual command. Copy must say permissions are independent from equivalent `fs_*`/Git tools and that no host shell runs. State words remain readable without color, rows stay keyboard reachable at supported terminal widths, and focus does not shift row dimensions.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/UI/test_mcp_workbench.py -k virtual_cli
```

Expected: FAIL.

- [ ] **Step 3: Project internal virtual HubTools beside existing local tools**

Extend the existing `_local_agent_hub_tools()`/catalog collection path to add `VirtualCliProvider.hub_tools()` with `executable=False` in the workbench projection. Keep the same permission-state resolver/editor. Always show the group when local agent tools are a supported feature, even if a current agent run has no provider instance; visibility explains policy and is not execution authority.

- [ ] **Step 4: Run mounted workbench tests**

```bash
pytest -q Tests/UI/test_mcp_workbench.py -k "local_agent or virtual_cli"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/test_mcp_workbench.py
git commit -m "feat: expose virtual CLI command permissions in Tools"
```

## Task 7: Document and verify the read-only virtual CLI

**Files:**

- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/mcp.md`
- Modify: `backlog/tasks/task-22509 - Read-only-virtual-CLI-with-independent-command-permissions.md`

- [ ] **Step 1: Document the exact contract**

Document the ten command forms, structured `{command, argv}` schema, no shell syntax/expansion/fallback, model-only status, workspace and sensitive-path policy, existing output caps, global gates, and independent virtual-command permissions. Include an example where virtual `cat` is Allow while `fs_read` is Off and explain that this is intentional separate authority.

- [ ] **Step 2: Run the focused virtual CLI suite**

```bash
pytest -q \
  Tests/Tools/test_virtual_cli_impls.py \
  Tests/Agents/test_virtual_cli_provider.py \
  Tests/Agents/test_virtual_cli_integration.py \
  Tests/Chat/test_console_virtual_cli_approval.py \
  Tests/MCP/test_local_store.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_console_mcp_approval.py \
  -k virtual_cli
```

Expected: PASS.

- [ ] **Step 3: Run static and whitespace checks**

```bash
ruff check tldw_chatbook/Tools/virtual_cli_impls.py tldw_chatbook/Agents/virtual_cli_provider.py tldw_chatbook/Agents/run_context.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
git diff --check
```

Expected: PASS. Do not run the full pytest suite unless the user explicitly requests it.

- [ ] **Step 4: Perform mounted and real-workspace verification**

With isolated config/data and a disposable repository, verify each Tools row, Ask approval, mixed repeated-call batch, session decision isolation, outside-root/sensitive-path refusal, and Git denylist behavior. Inspect the actual model request to confirm there is one `virtual_cli` schema and no raw-shell schema from this task.

- [ ] **Step 5: Self-review against ADR-094**

Search for `shell=True`, shell-string parsing, virtual-to-raw fallback, duplicated filesystem/Git implementations, shared permission names, unreserved external profile records, default-Allow behavior, and catalog-only gating.

- [ ] **Step 6: Complete Backlog hygiene after evidence exists**

Move TASK-22509 In Progress immediately before implementation and add this plan path through the CLI. At completion, check all criteria, add concise Implementation Notes/evidence and ADR-094, then set Done. Record a lesson only for a genuinely generalizable incident.

- [ ] **Step 7: Commit documentation and task completion**

```bash
git add Docs "backlog/tasks/task-22509 - Read-only-virtual-CLI-with-independent-command-permissions.md"
git commit -m "docs: describe the read-only virtual CLI"
```
