# Model shell_exec over Armed Raw CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a model request the same one-shot host shell as the direct user raw CLI, but only while raw CLI is persistently unlocked and armed and only through command-visible Run once/session/Deny authorization with no persistent silent Allow.

**Architecture:** `RawShellToolProvider` contributes one conditional `shell_exec` schema and adapts validated tool arguments to the app-owned `RawCliRuntime`; it contains no subprocess logic. A custom resolver projects persistent `Allow` as `Ask`, honors only Ask/Off plus a process-memory conversation-session grant, and rechecks every global/raw gate at invocation and runtime admission. Approval rows retain `call_id`, render the full command and host-authority warning, and revoke on disarm. Raw executor progress updates the already-created agent TOOL marker for that call; final output then follows the ordinary bounded tool-result/run-log/provider-history path.

**Tech Stack:** Python 3.11, existing Agent provider/catalog/review hooks, MCP permission store and approval card UI, `RawCliRuntime` from TASK-18926, per-call identity support from TASK-22509, Console store/transcript widgets, pytest.

**Backlog task:** `TASK-22510`

**ADR required:** yes

**ADR path:** `backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md`

**Reason:** ADR-093 establishes conditional model exposure, Ask/Off-only permission, process-memory session grants, command-visible approval, shared executor reuse, and disarm race behavior.

**Prerequisites:** TASK-18926 and TASK-22509 are complete. Do not start by copying their executor, output, or call-identity code.

---

## Task 1: Define the `shell_exec` schema and custom raw permission resolver

**Files:**

- Create: `tldw_chatbook/Agents/raw_shell_tool_provider.py`
- Create: `Tests/Agents/test_raw_shell_tool_provider.py`
- Modify: `Tests/MCP/test_permission_resolution.py`

- [ ] **Step 1: Write failing schema and state tests**

Pin one tool schema:

```json
{
  "command": "required non-empty string <= 16 KiB",
  "shell": "auto | bash | powershell | cmd (default auto)",
  "initial_directory": "optional absolute existing directory",
  "timeout_seconds": "optional number > 0 and <= 300"
}
```

Test malformed types, NUL, oversize command, relative/missing cwd, unknown shell, and excessive timeout fail before permission/launch. Test permission projection:

| Persisted/effective state | Runtime raw state | Model resolver |
| --- | --- | --- |
| Off | armed | Off |
| Ask/missing | armed | Ask |
| Allow (including hand-edited) | armed | Ask |
| any | locked/unarmed | unavailable/refused |

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Agents/test_raw_shell_tool_provider.py Tests/MCP/test_permission_resolution.py -k raw_shell
```

Expected: FAIL because the provider/resolver are absent.

- [ ] **Step 3: Implement the provider shell without executor logic**

Use the already-reserved local principal to avoid another permission store/profile boundary:

```python
RAW_SHELL_TOOL_NAME = "shell_exec"
RAW_SHELL_SERVER_KEY = "local:__local__"
RAW_SHELL_SERVER_LABEL = "Raw CLI (unsafe host shell)"
```

`RawShellToolProvider` implements `ToolProvider`, lists/loads one schema when `catalog_enabled()` is true, and exposes one `HubTool` for the Tools UI even while unavailable. Its constructor receives `RawCliRuntime`, Console session id, initial-directory resolver, current persistent state resolver, global/local kill-switch callables, permission callbacks, and an optional progress sink.

- [ ] **Step 4: Implement Ask/Off-only resolution**

Add a pure helper:

```python
def resolve_raw_shell_state(effective: EffectiveToolState) -> Literal["ask", "off"]:
    return "off" if effective.state == "off" else "ask"
```

Use it both for review and invocation. Never call generic `persist_approval(..., "allow")` for raw shell. A process-memory session grant may bypass Ask for its exact Console session, but it is not an `EffectiveToolState`, is never serialized, and does not change Tools UI state.

- [ ] **Step 5: Run schema/state tests**

```bash
pytest -q Tests/Agents/test_raw_shell_tool_provider.py Tests/MCP/test_permission_resolution.py -k raw_shell
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/raw_shell_tool_provider.py Tests/Agents/test_raw_shell_tool_provider.py Tests/MCP/test_permission_resolution.py
git commit -m "feat: define Ask-only model raw shell provider"
```

## Task 2: Add command-visible per-call approval and process-memory session grants

**Files:**

- Modify: `tldw_chatbook/Agents/raw_shell_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_raw_cli.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
- Create: `Tests/Chat/test_console_raw_shell_approval.py`
- Modify: `Tests/UI/test_console_mcp_approval.py`
- Modify: `Tests/UI/test_chat_approval_card.py`
- Modify: `Tests/UI/test_approval_argument_budget.py`

- [ ] **Step 1: Write failing approval tests**

For every pending row, assert the mounted card safely shows:

- complete multiline command;
- resolved shell selector;
- absolute initial directory;
- timeout;
- “full authority of the OS user” warning;
- local-log persistence warning;
- the fact that session approval covers future raw commands, not only this displayed command.

The first actionable focus target is a decision control already set to Deny, the complete command is keyboard-scrollable without truncation, and warning/scope meaning does not depend on color. Pressing Enter on initial focus may open the decision control; it must never submit approval directly.

Add regression coverage for the generic card's current 80-character argument-summary budget: ordinary approval rows retain that budget and their existing `approve_once` default, while only exact raw `shell_exec` rows receive the dedicated full-command view and Deny default.

Assert the only decisions are `approve_once`, `approve_session`, and `deny`; no `always_allow`. Use a batch of two `shell_exec` calls with different `call_id`/commands, approve one and deny one, and prove the undisplayed/denied command cannot execute.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Chat/test_console_raw_shell_approval.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_chat_approval_card.py Tests/UI/test_approval_argument_budget.py -k raw_shell
```

Expected: FAIL.

- [ ] **Step 3: Make danger copy a first-class bounded approval field**

Extend `MCPPendingCall` with bounded optional `full_command`, `warning`, and `scope_notice` strings. For the exact `(local:__local__, shell_exec)` identity, render `full_command` in a dedicated keyboard-scrollable multiline widget with `markup=False`, bounded by the provider schema's 16 KiB command limit rather than `_ARGS_SUMMARY_LIMIT`. Render warning and scope notice as adjacent markup-off text with explicit labels.

Add a row-level default-decision helper: raw `shell_exec` rows start at `deny`; every existing MCP/builtin/local row keeps the current `approve_once` default. Focus the raw row's already-Deny decision control first and keep the submit action separate. Preserve optional-field defaults so non-raw pending calls and cards remain behaviorally unchanged.

`RawShellToolProvider.pending_gate_for(call)` must copy `call.call_id`, include validated arguments, set:

```python
options=("approve_once", "approve_session", "deny")
```

and never truncate the command below the schema's 16 KiB cap. The card may scroll; it may not summarize a command the user is authorizing.

- [ ] **Step 4: Store session grants only in `RawCliRuntime`**

Add lock-protected methods:

```python
def grant_model_session(self, console_session_id: str) -> None: ...
def model_session_granted(self, console_session_id: str) -> bool: ...
def revoke_model_sessions(self) -> tuple[str, ...]: ...
```

The key is the Console conversation session id. Disarm and shutdown clear all grants; a new app process constructs an empty set. Do not write them to settings, session snapshots, AgentRunsDB, MCP permissions, or workspace state.

- [ ] **Step 5: Implement a raw-shell review hook keyed by call id**

Add `build_raw_shell_review_hook(provider, request_approvals)`. Clear this run's stamps at entry. Build one row per `ToolCall`. Apply `approve_once` only to `(run_id, call_id)`; apply `approve_session` by calling `runtime.grant_model_session(session_id)` only after the user's decision. Return denial keyed by `call_id or llm_name` so the generic runtime blocks that exact call before invoke.

For id-less fence calls, one call is the entire batch; name fallback is acceptable and fail-closed.

- [ ] **Step 6: Run approval tests**

```bash
pytest -q Tests/Chat/test_console_raw_shell_approval.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_chat_approval_card.py Tests/UI/test_approval_argument_budget.py -k "raw_shell or repeated_call"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/raw_shell_tool_provider.py tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Agents/mcp_tool_provider.py tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py Tests/Chat/test_console_raw_shell_approval.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_chat_approval_card.py Tests/UI/test_approval_argument_budget.py
git commit -m "feat: approve model raw shell calls per command"
```

## Task 3: Register `shell_exec` only while every catalog gate permits it

**Files:**

- Modify: `tldw_chatbook/Agents/raw_shell_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/Agents/test_raw_shell_integration.py`
- Modify: `Tests/Agents/test_local_tools_integration.py`

- [ ] **Step 1: Write the failing catalog and invocation matrix**

Cover all four gates independently:

| Saved unlock | Armed | Local tools | Block all tools | Schema |
| --- | --- | --- | --- | --- |
| off | any | any | any | absent |
| on | no | on | off | absent |
| on | yes | off | off | absent |
| on | yes | on | on | absent |
| on | yes | on | off | present |

For every row, also construct a stale provider/schema from the previously enabled state and call `invoke()` after toggling the gate. Assert no executor call occurs.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Agents/test_raw_shell_integration.py Tests/Agents/test_local_tools_integration.py -k raw_shell
```

Expected: FAIL.

- [ ] **Step 3: Compose the provider per run**

Alongside local and virtual providers, create/register `RawShellToolProvider` only when the latest saved unlock, runtime arm, local-tools setting, and global block-all switch permit it. Add `build_raw_shell_review_hook` to the combined review hook. Use the active binding/scratch only to choose the default initial directory; do not pass it as confinement authority.

- [ ] **Step 4: Recheck at the last safe moment**

Inside `invoke()` validate args, confirm its matching approval/session grant, re-read all four gates, then call `RawCliRuntime.execute`. Reuse TASK-18926's `admit_worker(tree)` callback: the executor spawns a waiting worker, then the runtime callback acquires the runtime lock, rechecks saved permission/armed state, and calls `tree.admit()` before releasing that lock. The provider performs no independent spawn/admission sequence. This closes schema-toggle, approval-toggle, and disarm-before-admission races: whichever of disarm or admission linearizes first either refuses the waiting worker or registers it for cancellation.

- [ ] **Step 5: Return an ordinary bounded model tool result**

Map `RawCliResult` to `ToolResult` without changing the executor:

- exit 0: `ok=True`;
- nonzero exit: `ok=False` with exit code and bounded stdout/stderr;
- timeout/cancel/refusal/spawn/containment failure: `ok=False` with stable category and bounded details;
- always include `truncated` and `cleanup_proven` facts;
- never include a larger result than the existing agent tool-result cap.

The normal AgentService path writes tool_call/tool_result records and appends the result to model history. Do not create an `agent_kind="local_command"` row for model calls.

- [ ] **Step 6: Run integration tests**

```bash
pytest -q Tests/Agents/test_raw_shell_integration.py Tests/Agents/test_local_tools_integration.py -k raw_shell
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/raw_shell_tool_provider.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Agents/test_raw_shell_integration.py Tests/Agents/test_local_tools_integration.py
git commit -m "feat: expose shell_exec only while raw CLI is armed"
```

## Task 4: Revoke pending approvals and active commands on disarm

**Files:**

- Modify: `tldw_chatbook/Chat/console_raw_cli.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Create: `Tests/Chat/test_console_raw_shell_revocation.py`
- Modify: `Tests/UI/test_settings_raw_cli.py`

- [ ] **Step 1: Write deterministic race tests**

Use events/barriers rather than sleeps for:

1. approval card pending → Disarm → card resolves denied and shell never starts;
2. approval returned → Disarm before provider invocation → runtime recheck refuses;
3. worker registered → Disarm → cancellation begins and bounded result reports cleanup certainty;
4. command exits → late Disarm → exited result is unchanged;
5. saved unlock toggled Off → same behavior as Disarm plus schema unavailable for later turns.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Chat/test_console_raw_shell_revocation.py Tests/UI/test_settings_raw_cli.py -k "disarm or raw_shell"
```

Expected: FAIL.

- [ ] **Step 3: Add one raw-approval revocation seam**

Register a process-local callback from `RawCliRuntime` to the Console controller. On disarm, the controller resolves only pending rows with `server_key=local:__local__` and `tool_name=shell_exec` as denied, preserving unrelated approvals. Reuse the existing round/run revocation machinery; do not reach into approval-card widgets from Settings.

Order under disarm:

1. set `_armed=False`;
2. clear model session grants and pending approval stamps;
3. revoke pending raw approval rounds;
4. snapshot/signal active invocation cancellation;
5. return immediately with cleanup in progress; completion updates arrive through ordinary event/result paths.

- [ ] **Step 4: Run revocation tests**

```bash
pytest -q Tests/Chat/test_console_raw_shell_revocation.py Tests/UI/test_settings_raw_cli.py -k "disarm or raw_shell"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Chat/test_console_raw_shell_revocation.py Tests/UI/test_settings_raw_cli.py
git commit -m "security: revoke model shell authority on disarm"
```

## Task 5: Stream model executor progress into the existing agent tool marker

**Files:**

- Modify: `tldw_chatbook/Agents/raw_shell_tool_provider.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/UI/Console_Modules/raw_cli.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Create: `Tests/Chat/test_console_raw_shell_progress.py`
- Modify: `Tests/Agents/test_agent_runtime.py`
- Modify: `Tests/Agents/test_agent_service_on_step.py`
- Modify: `Tests/UI/test_console_raw_cli_transcript.py`

- [ ] **Step 1: Write failing progress-correlation tests**

For two concurrent model raw calls, assert each stdout/stderr event updates only the TOOL marker created for its `(run_id, call_id)`. Assert there is one marker per call—not a raw-progress duplicate plus an agent-step duplicate. Verify the ordinary runtime emits the same `call_id` on its `STEP_TOOL_CALL` and `STEP_TOOL_RESULT`, navigation away/back continues updating the app-owned store, the final result updates the mapped marker rather than appending another marker, and late progress after terminal completion is ignored. Existing non-shell tool-marker behavior must remain unchanged.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/Chat/test_console_raw_shell_progress.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service_on_step.py Tests/UI/test_console_raw_cli_transcript.py -k "model or call_id"
```

Expected: FAIL because ordinary tool-call steps are not yet call-addressable and the provider has no marker progress sink.

- [ ] **Step 3: Make ordinary tool steps call-addressable**

In `agent_runtime.py`, propagate each `ToolCall.call_id` into both the ordinary `STEP_TOOL_CALL` and matching `STEP_TOOL_RESULT` `AgentStep` emissions. If a legacy/id-less fence call reaches this path, assign one deterministic correlation value from its run id and step index and use that same value for the pair. Do not change provider-visible tool names or model history.

- [ ] **Step 4: Create and retain exactly one mapped marker**

Change the bridge's local marker-append seam to return the created marker id. On a `shell_exec` `STEP_TOOL_CALL`, create one provisional TOOL marker and store `(run_id, call_id) -> marker_id`. Inject a `progress_sink(run_id, call_id, event)` into `RawShellToolProvider`; route it through the wired `ConsoleRawCliController`, marshal updates onto the app/Textual thread, and update that exact marker through the `ConsoleChatStore.update_tool_marker` seam delivered by TASK-18926.

When the matching `STEP_TOOL_RESULT` arrives, update/finalize the mapped marker instead of appending a second TOOL marker, then remove the mapping. Also remove it from terminal/finally paths so stale mappings cannot absorb later runs. Preserve the current append behavior for every non-`shell_exec` tool step.

The runtime must emit the call step synchronously before invoking the provider, so the marker mapping exists before the first executor event. Still fail safely if a marker is absent (navigation/recovery race): retain execution/logging and skip only the transient live projection. Keep the screen itself as a thin proxy so this feature does not increase the Console screen-size ratchet.

- [ ] **Step 5: Preserve ordinary final history and run logs**

Progress is session-only UI state. The final provider `ToolResult` remains the single source for AgentService's bounded role=`tool` history and run-log tool_result. Generic diagnostics remain content-free. Do not write every stream event as a run-log record.

- [ ] **Step 6: Run progress tests**

```bash
pytest -q Tests/Chat/test_console_raw_shell_progress.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service_on_step.py Tests/UI/test_console_raw_cli_transcript.py -k "model or call_id"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/raw_shell_tool_provider.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/UI/Console_Modules/raw_cli.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/Chat/test_console_raw_shell_progress.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service_on_step.py Tests/UI/test_console_raw_cli_transcript.py
git commit -m "feat: stream model shell progress into agent markers"
```

## Task 6: Show raw shell availability and Ask/Off-only controls in Tools

**Files:**

- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Modify: `Tests/UI/test_mcp_workbench.py`

- [ ] **Step 1: Write mounted Tools tests**

Assert `shell_exec` remains visible in Tools in Locked, Unlocked/not armed, and Armed states, with clear text-labeled availability copy. Its permission control cycles Ask ↔ Off only. Inject a hand-edited stored Allow and assert the UI displays Ask and runtime asks. Ensure the full host-authority warning is adjacent to the row and no workspace-confinement copy appears. The row remains keyboard reachable at supported terminal widths and does not rely on red alone.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -q Tests/UI/test_mcp_workbench.py -k raw_shell
```

Expected: FAIL.

- [ ] **Step 3: Add an always-visible non-executable projection**

Project `RawShellToolProvider.hub_tool()` into the workbench regardless of current schema registration. Attach availability facts (`locked`, `unarmed`, `armed`) derived from saved/runtime state. Override the permission-cycle/save adapter only for the exact `(local:__local__, shell_exec)` identity so it cannot emit Allow; preserve ordinary Allow/Ask/Off behavior for every other tool.

- [ ] **Step 4: Run Tools tests**

```bash
pytest -q Tests/UI/test_mcp_workbench.py -k "raw_shell or local_agent"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/test_mcp_workbench.py
git commit -m "feat: show Ask-only raw shell policy in Tools"
```

## Task 7: Document and verify model raw shell authorization

**Files:**

- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `backlog/tasks/task-22510 - Model-shell_exec-over-the-armed-raw-CLI.md`

- [ ] **Step 1: Document direct-user versus model authorization**

Explain that `! ` is user-authored and runs immediately when armed, while `shell_exec` is model-authored and additionally needs local tools, block-all Off, Ask/Off state, and Run once/session approval. Explain session grant scope, disarm/restart clearing, full command display, no persistent Allow, local logging, non-interactive stdin, timeout, and cleanup limitations.

- [ ] **Step 2: Run the focused model raw-shell suite**

```bash
pytest -q \
  Tests/Agents/test_raw_shell_tool_provider.py \
  Tests/Agents/test_raw_shell_integration.py \
  Tests/Chat/test_console_raw_shell_approval.py \
  Tests/Chat/test_console_raw_shell_revocation.py \
  Tests/Chat/test_console_raw_shell_progress.py \
  Tests/UI/test_console_mcp_approval.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_settings_raw_cli.py \
  -k "raw_shell or shell_exec"
```

Expected: PASS.

- [ ] **Step 3: Run static and whitespace checks**

```bash
ruff check tldw_chatbook/Agents/raw_shell_tool_provider.py tldw_chatbook/Chat/console_raw_cli.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/UI/Console_Modules/raw_cli.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
git diff --check
```

Expected: PASS. Do not run the full pytest suite unless the user explicitly asks.

- [ ] **Step 4: Perform mounted end-to-end verification with isolated state**

Using isolated config/data, verify schema absence/presence across every gate, exact approval card content, Run once, Allow for this Console session, Deny, mixed repeated-call batch, live output, Stop, Disarm while pending/running, navigation, and restart clearing both arm and session grants. Inspect the real provider request/history and local run log to prove model results are ordinary bounded tool messages and direct-user records remain model-excluded.

- [ ] **Step 5: Self-review against ADR-093**

Search for a copied executor, persistent raw Allow/session grant, schema-only gate, incomplete command approvals, call-name de-duplication, post-disarm launch window, ambient environment inheritance, command/output generic logs, and model calls creating `local_command` rows.

- [ ] **Step 6: Complete Backlog hygiene after evidence exists**

Move TASK-22510 In Progress immediately before implementation and attach this plan via the CLI. At completion, check every criterion, add concise Implementation Notes/evidence and ADR-093, then set Done. Add a lesson only if an actual generalizable incident occurred.

- [ ] **Step 7: Commit documentation and task completion**

```bash
git add Docs "backlog/tasks/task-22510 - Model-shell_exec-over-the-armed-raw-CLI.md"
git commit -m "docs: explain model raw shell authorization"
```
