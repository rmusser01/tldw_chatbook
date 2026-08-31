# MCP Hub Local Tool Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the MCP Hub safely test descriptor-approved `local:__local__` tools with one-click Ask approval, fresh workspace authority, bounded execution ownership, and one redacted terminal audit outcome.

**Architecture:** `UnifiedMCPControlPlaneService` owns immutable Test Tool previews and the prepared-execution entry point. A focused `hub_test_execution.py` module owns preview/argument primitives plus the in-flight coordinator, while `local_server_tools.py` supplies a dedicated, closable Hub-local provider composition and `LocalToolProvider.invoke_detailed()` supplies structured local terminal facts without changing normal callers. The existing Workbench and Inspector become clients of that service boundary: they render a service-issued preview and send an explicit `run` or `approve_once` intent, but never decide authorization themselves.

**Tech Stack:** Python 3.11+, asyncio/thread workers, Textual 8.x, pytest/pytest-asyncio, Ruff, the existing MCP permission store/execution log, `DirectoryChain` workspace authority, and `LocalToolProvider`.

**Spec:** `Docs/superpowers/specs/2026-08-30-mcp-hub-local-tool-execution-design.md`

**ADR:** `backlog/decisions/032-local-agent-tool-permission-boundary.md` (TASK-3605 addendum)

**Backlog task:** `backlog/tasks/task-3605 - Enable-fail-closed-MCP-Hub-execution-for-local-agent-tools.md`

---

## Completed design prerequisites

Before this implementation plan was written, commit `029f851084` completed the required architecture-document work:

- amended `backlog/decisions/032-local-agent-tool-permission-boundary.md` with the TASK-3605 operator-only Hub policy;
- corrected `Docs/superpowers/specs/2026-07-13-mcp-hub-redesign-design.md` so configured Off is blocked and Ask is one-click **Approve & run once**, superseding the earlier second-confirmation proposal;
- linked the ADR and approved TASK-3605 design from the Backlog task.

Execution must verify those prerequisite changes are still present before Task 1. Do not create a second ADR or reintroduce the superseded confirmation language.

## File map

- Create `tldw_chatbook/MCP/hub_test_execution.py`: immutable preview/outcome types, exact JSON canonicalization, authority fingerprinting, bounded preview registry, and service-owned coordinator.
- Create `Tests/MCP/test_hub_test_execution.py`: isolated registry, canonicalization, authority, timeout, cancellation, and late-worker ownership tests.
- Modify `tldw_chatbook/Agents/local_tool_provider.py`: compatible `invoke_detailed()` seam and structured local reason/terminal types.
- Modify `Tests/Agents/test_local_tool_provider.py`: detailed-result coverage and byte-for-byte ordinary `invoke()` compatibility.
- Modify `tldw_chatbook/MCP/local_server_tools.py`: dedicated closable Hub-local provider factory, descriptor filtering, one-shot approval callback, fresh root guard, and Watchlists resolver cleanup.
- Modify `Tests/MCP/test_local_server_tools.py`: Hub composition, eligibility, kill-switch carve-out, redaction-root, callback, and cleanup tests.
- Modify `tldw_chatbook/MCP/unified_control_plane_service.py`: local executable projection, preview APIs, prepared admission, local dispatch, coordinator integration, and sole audit finalization.
- Modify `Tests/MCP/test_control_plane_permissions.py`: preview gate, identity/definition/authority transition, and Ask/Off behavior.
- Modify `Tests/MCP/test_control_plane_tool_execute.py`: local execution, audit, timeout, cancellation, cleanup, and compatibility tests.
- Modify `Tests/MCP/test_unified_control_plane_service.py`: service construction, local projection, and lifecycle compatibility coverage.
- Modify `Tests/MCP/test_hub_tool_catalog.py`: exact executable projection and identity coverage.
- Modify `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`: preview-backed Test Tool panel, explicit one-click intent, active-state rendering, and removal of the armed-confirm state.
- Modify `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`: request/revoke previews, invoke only `execute_prepared_hub_test()`, render typed outcomes, and delegate duplicate ownership to the service.
- Modify `Tests/UI/test_mcp_inspector.py`: pure Inspector preview, intent-message, and armed-state removal coverage.
- Modify `Tests/UI/test_mcp_workbench.py`: executable projection and mounted one-click Ask/Allow/Off/remount flows.
- Modify `Tests/UI/test_mcp_tools_mode.py` only if the executable row state needs a pure-canvas assertion.
- Modify `backlog/tasks/task-3605 - Enable-fail-closed-MCP-Hub-execution-for-local-agent-tools.md`: final checked acceptance criteria and implementation notes after verification.

## Task 1: Add a compatible structured local-provider result seam

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Test: `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Write failing compatibility and reason tests**

Add focused tests proving that `invoke_detailed()` distinguishes `unknown_tool`, `invalid_arguments`, `permission_off`, `permission_unresolved`, `approval_refused`, `approval_timeout`, `root_changed`, `authority_unavailable`, `handler_returned`, and `handler_raised`; records `approval_consumed` and `dispatch_started`; and returns only `not_started`, `returned`, or `raised` provider terminals.

Also parameterize representative calls across two freshly constructed, identically configured providers and assert:

```python
detailed = detailed_provider.invoke_detailed(tool_id, copy.deepcopy(arguments))
ordinary = ordinary_provider.invoke(tool_id, copy.deepcopy(arguments))
assert ordinary == detailed.result
```

Do not compare sequential Ask calls on the same provider because one-shot approval consumption and decision recording are intentionally stateful.

- [ ] **Step 2: Run the new tests and verify the seam is absent**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Agents/test_local_tool_provider.py -k "invoke_detailed or ordinary_invoke_matches"
```

Expected: FAIL because `invoke_detailed` and the structured types do not exist.

- [ ] **Step 3: Add the narrow result types**

Add code-owned enums and a frozen result record near `LocalToolSpec`:

```python
class LocalToolInvocationReason(str, Enum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    PERMISSION_OFF = "permission_off"
    PERMISSION_UNRESOLVED = "permission_unresolved"
    APPROVAL_REFUSED = "approval_refused"
    APPROVAL_TIMEOUT = "approval_timeout"
    ROOT_CHANGED = "root_changed"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    HANDLER_RETURNED = "handler_returned"
    HANDLER_RAISED = "handler_raised"


class LocalProviderTerminal(str, Enum):
    NOT_STARTED = "not_started"
    RETURNED = "returned"
    RAISED = "raised"


@dataclass(frozen=True, slots=True)
class LocalToolInvocationResult:
    result: ToolResult
    final_gate: str
    approval_consumed: bool
    reason_code: LocalToolInvocationReason
    dispatch_started: bool
    provider_terminal: LocalProviderTerminal
```

Keep these local-provider-specific; do not widen the global `ToolResult` protocol.

- [ ] **Step 4: Refactor one private implementation behind both public methods**

Move the current `invoke()` body to `_invoke_detailed()`. Preserve every existing refusal string, redaction, record-decision call, root check, authority scope, and handler exception mapping. Have:

```python
def invoke(self, tool_id: str, args: dict) -> ToolResult:
    return self._invoke_detailed(tool_id, args).result

def invoke_detailed(self, tool_id: str, args: dict) -> LocalToolInvocationResult:
    return self._invoke_detailed(tool_id, args)
```

Do not infer reason codes from result text. Carry the verdict and approval-consumption fact through the existing `_verdict_for()` path as structured internal data.

- [ ] **Step 5: Run provider tests**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Agents/test_local_tool_provider.py
```

Expected: PASS, including all legacy `invoke()` assertions.

- [ ] **Step 6: Commit the provider seam**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "refactor(agents): expose typed local tool outcomes"
```

## Task 2: Build the closable Hub-local provider composition and executable projection

**Files:**
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Test: `Tests/MCP/test_local_server_tools.py`
- Test: `Tests/UI/test_mcp_workbench.py`

- [ ] **Step 1: Write failing composition tests**

Cover these exact outcomes:

- the Hub factory includes only `LocalToolExposure.CONSOLE_AND_EXTERNAL_MCP` specs;
- the ordinary full local catalog remains the inspection source, so Console-only Watchlists article tools remain visible but `executable=False` in the Hub projection;
- `todo_*` is absent because there is no session store;
- `local:__virtual_cli__` stays non-executable;
- `[console] local_tools_enabled=false` keeps the full local inspection group visible but marks every row `executable=False`, while `[mcp] expose_local_tools` has no effect;
- root resolution or filtered-provider construction failure keeps the full local inspection rows visible and non-executable and leaves unrelated built-in/external rows unchanged;
- the factory injects `kill_switch=lambda: False`, a fresh permission resolver, `result_redaction_root`, no provider decision recorder, and a caller-supplied one-shot approval callback;
- closing the provider handle closes an opened lazy Watchlists database exactly once, including exception cleanup.

- [ ] **Step 2: Run the focused projection tests and verify failure**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_local_server_tools.py \
  Tests/UI/test_mcp_workbench.py -k "hub_local or local_agent_tools_as_own_group or virtual_cli"
```

Expected: FAIL because the Hub factory does not exist and current local rows are all non-executable.

- [ ] **Step 3: Add a closable factory handle**

Add a small context-managed handle in `local_server_tools.py`:

```python
@dataclass(slots=True)
class HubLocalProviderHandle:
    provider: LocalToolProvider
    authority: DirectoryChain
    resolver: _LazyWatchlistsDBResolver

    def close(self) -> None:
        self.resolver.close()
```

Add `_LazyWatchlistsDBResolver.close()` under its existing lock. The Hub builder captures `DirectoryChain`, filters the shared descriptors, sets `kill_switch=lambda: False`, threads the exact `approval_callback`, uses `result_redaction_root=authority.canonical_root`, omits `record_decision`, and returns the handle. Do not change `build_server_local_provider()` semantics.

- [ ] **Step 4: Move local executable projection behind the service**

Add `local_hub_tools()` on `UnifiedMCPControlPlaneService`. Build the full inspectable catalog through the ordinary local-provider composition. Only when `[console] local_tools_enabled=true`, build one separate filtered Hub handle to establish executable shared identities, merge eligibility by `(server_key, name, definition_hash)`, and close both compositions in `finally`. The filtered factory must not become the inspection source or Console-only tools would disappear. A disabled configuration or any filtered-provider/root failure returns the full rows with `executable=False`; it must not fail the complete catalog refresh or remove unrelated groups. Update `MCPWorkbench._local_agent_hub_tools()` to call this service seam and keep only Virtual CLI/raw-shell projection locally.

- [ ] **Step 5: Run composition and mounted catalog tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_local_server_tools.py \
  Tests/UI/test_mcp_workbench.py -k "local_agent or hub_local or virtual_cli"
```

Expected: PASS with shared descriptors executable, Console-only rows visible but disabled, session tools absent, and Virtual CLI unchanged.

- [ ] **Step 6: Commit provider composition and projection**

```bash
git add \
  tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/unified_control_plane_service.py \
  tldw_chatbook/UI/MCP_Modules/mcp_workbench.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/UI/test_mcp_workbench.py
git commit -m "feat(mcp): project executable Hub local tools"
```

## Task 3: Add immutable admission primitives and a bounded preview registry

**Files:**
- Create: `tldw_chatbook/MCP/hub_test_execution.py`
- Create: `Tests/MCP/test_hub_test_execution.py`

- [ ] **Step 1: Write failing pure-unit tests**

Cover:

- canonical arguments accept only an object with string keys and JSON values;
- booleans stay booleans, integer/float values remain finite, `NaN`/infinity/custom objects/non-string keys fail;
- encoding is sorted, compact, and round-trips to the deep-copied dispatch object;
- authority fingerprints change when the canonical locator or any `DirectoryIdentity` field changes;
- preview nonces are service-minted, opaque, single-use, bounded, and TTL-expiring;
- revoke, consume, expiry, and capacity eviction remove the preview;
- concurrent consume calls yield exactly one winner.

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
../../.venv/bin/python -m pytest -q Tests/MCP/test_hub_test_execution.py
```

Expected: collection FAIL because `hub_test_execution.py` does not exist.

- [ ] **Step 3: Implement immutable records and exact canonicalization**

Use frozen, slotted dataclasses:

```python
@dataclass(frozen=True, slots=True)
class ToolTestAdmissionPreview:
    nonce: str
    server_key: str
    tool_name: str
    definition_hash: str
    rendered_gate: str
    authority_fingerprint: str | None
    safe_authority_label: str | None


@dataclass(frozen=True, slots=True)
class RegisteredToolTestPreview:
    public: ToolTestAdmissionPreview
    authority: DirectoryChain | None
    expires_at: float
```

`canonicalize_arguments()` must return `(canonical_bytes, deep_copy)` using `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)` followed by `json.loads()`.

- [ ] **Step 4: Implement the synchronized registry**

Use a `threading.Lock`, `secrets.token_urlsafe()`, `time.monotonic()`, a small fixed maximum, and insertion order. Expose only `issue()`, `consume()`, `revoke()`, and `clear()`; `consume()` removes before returning. Never expose the canonical locator or identity chain through the public preview.

- [ ] **Step 5: Run pure-unit tests**

```bash
../../.venv/bin/python -m pytest -q Tests/MCP/test_hub_test_execution.py
```

Expected: PASS.

- [ ] **Step 6: Commit admission primitives**

```bash
git add tldw_chatbook/MCP/hub_test_execution.py Tests/MCP/test_hub_test_execution.py
git commit -m "feat(mcp): add Hub test admission previews"
```

## Task 4: Put every Hub Test Tool click behind prepared service admission

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/MCP/hub_test_execution.py`
- Test: `Tests/MCP/test_control_plane_permissions.py`
- Test: `Tests/MCP/test_control_plane_tool_execute.py`

- [ ] **Step 1: Write failing preview/admission tests**

Add service tests for:

- `prepare_hub_test(tool)` issuing the current definition/gate/authority preview;
- `revoke_hub_test_preview(nonce)` making the nonce unusable;
- `execute_prepared_hub_test(nonce, "run", args)` dispatching only rendered-Allow + fresh-Allow;
- `execute_prepared_hub_test(nonce, "approve_once", args)` dispatching rendered-Ask + fresh-Ask and rendered-Ask + fresh-Allow;
- Allow→Ask, Ask→Off, gate error, definition change, exact identity change, authority-chain change, expired/reused nonce, and wrong intent returning a typed blocked/stale outcome with zero handler calls;
- a stored Allow whose definition hash changes resolving to fresh Ask, refusing the rendered `run` click with zero approval consumption/dispatch and a refreshed preview;
- `[console] local_tools_enabled` changing on→off between render and click, click-time root resolution failure, and Hub-provider construction failure, each producing zero handler calls, a refreshed unavailable/blocked preview, and an audit only when service admission had already occurred;
- concurrent double-click consuming one nonce once;
- invalid form data failing before admission/audit;
- external/built-in callers still reaching the unchanged low-level `test_hub_tool()` only after prepared admission.

- [ ] **Step 2: Run tests and verify service methods are absent**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  -k "prepared_hub or admission_preview or preview_nonce"
```

Expected: FAIL for missing service APIs.

- [ ] **Step 3: Add the service-owned registry and public preview lifecycle**

Initialize one registry in `UnifiedMCPControlPlaneService.__init__`. Implement:

```python
def prepare_hub_test(self, tool: HubTool) -> ToolTestAdmissionPreview: ...
def revoke_hub_test_preview(self, nonce: str) -> None: ...
async def execute_prepared_hub_test(
    self, nonce: str, intent: Literal["run", "approve_once"], arguments: dict[str, Any]
) -> dict[str, Any] | LocalHubExecutionOutcome: ...
```

Resolve the live tool by exact identity, recapture authority, re-resolve the definition and gate, and compare every registered field after atomically consuming the nonce. Never convert a `run` intent into approval.

- [ ] **Step 4: Preserve the low-level compatibility seam**

Keep `test_hub_tool()` and `execute_hub_tool()` available for existing runtime/bridge callers and tests. Only the Workbench path will move to `execute_prepared_hub_test()`. After prepared admission, the non-local branch delegates to the already-audited `test_hub_tool()` and performs no outer audit append. The local branch never calls that delegate and uses the coordinator as its sole terminal audit finalizer.

- [ ] **Step 5: Run admission and legacy execution tests together**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  Tests/MCP/test_control_plane_bridge.py
```

Expected: PASS; legacy callers remain compatible and all prepared race tests are green.

- [ ] **Step 6: Commit prepared admission**

```bash
git add \
  tldw_chatbook/MCP/hub_test_execution.py \
  tldw_chatbook/MCP/unified_control_plane_service.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_control_plane_tool_execute.py
git commit -m "feat(mcp): gate Hub tests with immutable previews"
```

## Task 5: Execute local tests under service-owned timeout, cancellation, and audit ownership

**Files:**
- Modify: `tldw_chatbook/MCP/hub_test_execution.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Test: `Tests/MCP/test_hub_test_execution.py`
- Test: `Tests/MCP/test_control_plane_tool_execute.py`
- Test: `Tests/MCP/test_local_runtime_delegate.py`

- [ ] **Step 1: Write failing coordinator and local-dispatch tests**

Cover:

- persistent Allow and one-time Ask invoke the local provider once;
- Ask approval binds invocation ID, preview identity/hash, authority fingerprint, and digest of the exact canonical bytes, consumes once, and is never persisted;
- `LocalHubExecutionOutcome` derives reason/final gate/approval/dispatch/provider terminal from `invoke_detailed()` without matching text;
- lifecycle-default timeout applies unless the provider declares a longer tool-specific timeout floor, and the longer floor wins;
- an already-cancelled request and cancellation during pre-dispatch permission review produce zero handler calls;
- `bounded_abandonable` timeout seals one timeout outcome/audit while a late worker return/raise is cleanup-only;
- bounded caller cancellation seals one cancellation terminal/audit before presentation detaches; a late worker return/raise is cleanup-only and cannot duplicate or replace it;
- injected eligible `definitive_after_start` work survives caller cancellation after dispatch and reports its actual terminal result;
- injected eligible `definitive_after_start` work cannot receive a timeout/cancellation terminal after dispatch;
- duplicate exact tool admission is refused across a remount while active; reservation exists before dispatch and releases only after the owning finalizer finishes;
- click-time disabled configuration, root failure, and provider-construction failure produce typed zero-dispatch outcomes and one post-admission audit attempt;
- provider/root/Watchlists cleanup runs under success, refusal, timeout, cancellation, and exception;
- a table-driven result/audit matrix distinguishes persistent Allow, one-time Ask approval, configured Off, unresolved permission, eligibility/configuration mismatch, cancellation, timeout, provider refusal, crash, and success;
- every matrix row uses the same bounded result for display and audit, redacts absolute roots, safe workspace-label authority details, secret-shaped arguments/results, and unexpected exception text, and never stores canonical argument bytes or authority identities;
- exactly zero or one best-effort terminal append is attempted, append failure does not mask the tool outcome, and ambiguous append is not retried;
- the raw `local:__local__` `tools/call` route still raises `RawToolCallRefusedError`.

- [ ] **Step 2: Run the coordinator tests and verify failure**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_hub_test_execution.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  Tests/MCP/test_local_runtime_delegate.py \
  -k "local_hub or definitive or late_worker or raw_tool_call"
```

Expected: FAIL because local prepared execution and coordinator ownership are incomplete.

- [ ] **Step 3: Implement `LocalHubExecutionCoordinator`**

The coordinator owns strong task/future references and exact active keys. Expose a read-only service seam `hub_test_active(server_key: str, tool_name: str) -> bool` backed by that registry; the UI may render it but may not mutate it. Reserve the exact key before dispatch, refuse a second admission while it is present, and release it only from the owning finalizer after terminal construction, audit attempt, and dependency cleanup. Use a single BaseException-safe finalizer to seal the `LocalHubExecutionOutcome`, append audit at most once, close dependencies, and release the key. For bounded work, retain a cleanup callback/future after timeout or cancellation so the worker's eventual result or exception is consumed without a second UI/audit finalization. For definitive work, shield and await the actual handler completion after dispatch and never synthesize timeout/cancellation after `dispatch_started=True`.

- [ ] **Step 4: Implement the local prepared branch**

Create a fresh Hub-local provider handle only after preview admission. Inject a private one-shot approval callback for `approve_once`; call `invoke_detailed()` through `asyncio.to_thread()`; use the descriptor's `ToolExecutionPolicy` and effective timeout floor; convert structured provider facts into one `LocalHubExecutionOutcome`; redact before both display and audit; close the provider handle in the coordinator finalizer.

Check cancellation before provider construction and again immediately before dispatch. Pre-dispatch cancellation is a zero-handler terminal; bounded post-dispatch cancellation is coordinator-sealed and audited before detachment. Derive the effective timeout as `max(hub_lifecycle_timeout, provider_timeout_floor)` for bounded tools. Run all result and audit fields through the existing bounded logging/redaction helpers before constructing the common display/audit payload.

- [ ] **Step 5: Run local execution and raw-refusal suites**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_hub_test_execution.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  Tests/MCP/test_local_runtime_delegate.py \
  Tests/MCP/test_local_server_tools.py
```

Expected: PASS with no duplicate audit record and no raw-call bypass.

- [ ] **Step 6: Commit coordinator execution**

```bash
git add \
  tldw_chatbook/MCP/hub_test_execution.py \
  tldw_chatbook/MCP/unified_control_plane_service.py \
  tldw_chatbook/MCP/local_server_tools.py \
  Tests/MCP/test_hub_test_execution.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  Tests/MCP/test_local_runtime_delegate.py \
  Tests/MCP/test_local_server_tools.py
git commit -m "feat(mcp): execute Hub local tools under owned coordination"
```

## Task 6: Replace the Workbench's armed confirmation with preview-backed one-click intent

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Modify: `Tests/UI/test_mcp_inspector.py`
- Modify: `Tests/UI/test_mcp_workbench.py`
- Modify: `Tests/UI/test_mcp_tools_mode.py` if needed

- [ ] **Step 1: Replace obsolete armed tests with failing one-click tests**

Update the mounted coverage to prove:

- panel open requests one preview and initially disables Run while preparing;
- Allow labels the action **Run** and emits intent `run` once;
- Ask labels the action **Approve & run once** and emits intent `approve_once` on the first press;
- Off/unresolved renders Blocked/Unavailable without dispatch;
- editing arguments does not create a second confirmation step; the exact current arguments are sent for service canonicalization;
- tool switch, Close, Escape, source/mode switch, and remount revoke the old preview;
- a stale service outcome refreshes the panel instead of executing;
- remount reads active state from the service coordinator;
- rapid double press results in one service admission;
- existing local-profile and built-in one-click flows still work through the same prepared entry point;
- focus tests use stable keyboard interaction where recomposition can replace a widget, per `lessons-testing-evidence.md`.

- [ ] **Step 2: Run the mounted tests and verify they fail against armed-confirm behavior**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_mcp_inspector.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_mcp_tools_mode.py \
  -k "test_tool and (one_click or preview or local_agent or remount or double_run)"
```

Expected: FAIL because Ask still arms **Confirm run** and the widget owns duplicate/gate state.

- [ ] **Step 3: Make the Inspector a preview renderer, not an authorizer**

Replace `_test_run_armed`, `require_confirm()`, `disarm_test_run()`, armed notices, and edit-to-disarm handlers with one immutable preview slot. Add methods such as:

```python
def show_test_preview(self, preview: ToolTestAdmissionPreview) -> None: ...
def clear_test_preview(self) -> str | None: ...
def show_test_active(self, active: bool) -> None: ...
```

Extend `ToolTestRequested` to carry `preview_nonce`, explicit `intent`, and current form arguments. Keep the panel's stable widget IDs where practical to minimize CSS/test churn.

- [ ] **Step 4: Route Workbench panel lifecycle through the service**

On open, request a preview off the UI loop and render it only if the same tool/panel is still current. On Close/switch/remount, revoke the nonce best-effort. On Run, call only `execute_prepared_hub_test()`; remove `_resolve_test_gate()` authorization and widget-local `_tool_test_in_flight` as authoritative guards. Render the typed outcome and refresh a stale preview. Update all test fakes to implement the prepared methods. A missing prepared service method must render an unavailable state and fail closed; there is no compatibility dispatch to `test_hub_tool()` from the Workbench.

- [ ] **Step 5: Run the full Workbench test file**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_mcp_inspector.py \
  Tests/UI/test_mcp_workbench.py
```

Expected: PASS after updating superseded two-press assertions to the accepted one-click contract.

- [ ] **Step 6: Commit the one-click UI**

```bash
git add \
  tldw_chatbook/UI/MCP_Modules/mcp_inspector.py \
  tldw_chatbook/UI/MCP_Modules/mcp_workbench.py \
  Tests/UI/test_mcp_inspector.py \
  Tests/UI/test_mcp_workbench.py \
  Tests/UI/test_mcp_tools_mode.py
git commit -m "feat(ui): make Hub Ask approval one click"
```

## Task 7: Run the integrated security and lifecycle verification matrix

**Files:**
- Test: `Tests/Agents/test_local_tool_provider.py`
- Test: `Tests/MCP/test_hub_test_execution.py`
- Test: `Tests/MCP/test_local_server_tools.py`
- Test: `Tests/MCP/test_control_plane_permissions.py`
- Test: `Tests/MCP/test_control_plane_tool_execute.py`
- Test: `Tests/MCP/test_unified_control_plane_service.py`
- Test: `Tests/MCP/test_hub_tool_catalog.py`
- Test: `Tests/MCP/test_control_plane_bridge.py`
- Test: `Tests/MCP/test_local_runtime_delegate.py`
- Test: `Tests/UI/test_mcp_tools_mode.py`
- Test: `Tests/UI/test_mcp_inspector.py`
- Test: `Tests/UI/test_mcp_workbench.py`
- Test: `Tests/Architecture/test_persistent_diagnostic_inventory.py`

- [ ] **Step 1: Run the complete targeted feature matrix**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Agents/test_local_tool_provider.py \
  Tests/MCP/test_hub_test_execution.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  Tests/MCP/test_unified_control_plane_service.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/MCP/test_control_plane_bridge.py \
  Tests/MCP/test_local_runtime_delegate.py \
  Tests/UI/test_mcp_tools_mode.py \
  Tests/UI/test_mcp_inspector.py \
  Tests/UI/test_mcp_workbench.py
```

Expected: PASS. If a joined-only UI failure appears, preserve the failing interleaving and follow the stable-focus/recomposition lessons rather than weakening the product assertion.

- [ ] **Step 2: Run static checks on every touched Python file**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/MCP/hub_test_execution.py \
  tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/unified_control_plane_service.py \
  tldw_chatbook/UI/MCP_Modules/mcp_inspector.py \
  tldw_chatbook/UI/MCP_Modules/mcp_workbench.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/MCP/test_hub_test_execution.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_control_plane_tool_execute.py \
  Tests/MCP/test_unified_control_plane_service.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/MCP/test_local_runtime_delegate.py \
  Tests/UI/test_mcp_tools_mode.py \
  Tests/UI/test_mcp_inspector.py \
  Tests/UI/test_mcp_workbench.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/MCP/hub_test_execution.py \
  tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/unified_control_plane_service.py \
  tldw_chatbook/UI/MCP_Modules/mcp_inspector.py \
  tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
```

Expected: clean.

- [ ] **Step 3: Run compilation, diagnostics, and diff hygiene**

```bash
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/MCP/hub_test_execution.py \
  tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/unified_control_plane_service.py \
  tldw_chatbook/UI/MCP_Modules/mcp_inspector.py \
  tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -m pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py
git diff --check origin/dev...HEAD
```

Expected: clean. If the diagnostic inventory changes because new diagnostics are intentional, inspect the semantic delta before running its documented `--write` flow; never hand-edit the inventory.

- [ ] **Step 4: Review security invariants explicitly**

Inspect the final diff and confirm:

- no route makes `local:__local__` a transport profile;
- every Hub dispatch begins with a service-owned preview consume;
- no bare-name allowlist determines eligibility;
- no refusal text is parsed for outcome classification;
- no absolute root reaches UI or audit;
- one-time approval is neither session nor persistent permission state;
- provider and service cannot both append a local terminal audit;
- late bounded worker completion cannot update UI/audit;
- definitive-after-start execution remains service-owned;
- no full test sweep is run unless the user requests it or the merge gate requires it.

- [ ] **Step 5: Request independent code review**

Use `superpowers:requesting-code-review` against `origin/dev...HEAD`. Resolve every valid finding with a focused regression test before continuing.

- [ ] **Step 6: Commit any verification-only corrections**

Stage only the files changed by the correction, naming each path explicitly, then run:

```bash
git diff --cached --check
git commit -m "test(mcp): close Hub local execution verification"
```

Skip this commit when verification produces no changes.

## Task 8: Close the Backlog task and prepare the PR

**Files:**
- Modify: `backlog/tasks/task-3605 - Enable-fail-closed-MCP-Hub-execution-for-local-agent-tools.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` only if implementation reveals a genuinely new, incident-backed lesson.

- [ ] **Step 1: Confirm every acceptance criterion from evidence**

Map each checked criterion to a passing test or explicit static inspection. Do not check a criterion based only on code review.

- [ ] **Step 2: Add concise implementation notes**

Document the provider seam, preview registry, authority binding, coordinator ownership, one-click UI, audit/redaction behavior, touched files, focused test counts, static checks, independent review, and any plan deviation. Re-link ADR-032 and the design/plan.

- [ ] **Step 3: Mark the task Done only after all Definition of Done gates pass**

Use the Backlog CLI first so it cannot overwrite hand-authored notes later:

```bash
backlog task edit 3605 -s Done --notes "Implemented fail-closed MCP Hub execution for descriptor-approved local tools with service-owned previews, one-click Ask approval, fresh workspace authority, structured outcomes, owned completion, and one terminal redacted audit path."
```

Then inspect the exact printed file path, restore/expand notes if the CLI replaced them, and check every `- [ ]` to `- [x]` with `apply_patch`.

- [ ] **Step 4: Verify task rendering and clean tree**

```bash
backlog task 3605 --plain
git diff --check
git status --short
```

Expected: TASK-3605 is Done, all criteria checked, implementation plan/notes present, and only intended closeout files are modified.

- [ ] **Step 5: Commit closeout**

```bash
git add \
  'backlog/tasks/task-3605 - Enable-fail-closed-MCP-Hub-execution-for-local-agent-tools.md' \
  Docs/superpowers/plans/2026-08-30-mcp-hub-local-tool-execution.md
git commit -m "docs: close TASK-3605"
```

- [ ] **Step 6: Rebase and re-run exact-head gates before PR/merge**

Fetch `origin/dev`; if it advanced, rebase, rerun the targeted matrix, static/diff/diagnostic checks, and inspect Backlog task-ID/path integrity. Create the PR only after exact-head verification. Address every valid Qodo/reviewer comment with a focused test, then merge normally when required checks and review threads are resolved.
