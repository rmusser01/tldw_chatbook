# MCP Hub Redesign — Phase 5 (Chat Bridge + Audit Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MCP tools into the Console agent runtime as an `MCPToolProvider` (task-201) with call-time permission gating and a real batch-approval flow, and ship the Audit mode (Executions over the JSONL log + read-only server Findings).

**Architecture:** The bridge targets the Console/agent-runtime stack ONLY (user decision 2026-07-17; the legacy Chat tab keeps its manual-paste loop; server-source execution is deferred to Phase 6). A new sync `MCPToolProvider` registers beside Builtin/Skill providers in the per-run `ToolCatalogRegistry`; its `invoke()` bridges from the agent worker thread to Textual's MAIN event loop via `asyncio.run_coroutine_threadsafe` (MCP client sessions are main-loop-bound; the agent thread is `asyncio.to_thread`, so the main loop is free — no deadlock). Approval is genuinely new plumbing: the agent loop gains an optional pre-dispatch batch-review hook; the Console controller shows a batch approval card and the worker thread blocks on a `threading.Event` with timeout. Audit mode is a standard workbench canvas over `MCPExecutionLog.read_recent()` plus a fail-soft server Findings listing.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, threading.Event bridging, existing MCP control-plane seams (Phase 3/4).

## Global Constraints

- Bridge scope: **local: + builtin: tools only** (server-source execution = Phase 6, recorded). Legacy Chat tab untouched.
- Assembly is send-time (per-run registry composition, already fresh per run): eligible = tools of servers with a discovery snapshot (local) or the built-in inventory ∩ effective state ≠ deny ∩ **kill switch off** (switch on → provider not registered at all). No connect calls during composition; execution reuses `execute_external_tool`'s connect-if-needed (post-approval execution is not "silent"). When ≥1 eligible local server has `is_connected=False`, the Console inspector shows a `ConsoleDisplayRow("MCP", "N servers enabled, not connected", status="blocked")` — the spec's affordance, in the established row idiom.
- Naming (spec §11 verbatim): LLM-facing name `mcp__<server>__<tool>` sanitized to `[a-zA-Z0-9_-]`, ≤64 chars, truncate-with-hash (sha256 hex8 suffix), post-sanitization collisions get numeric suffixes. Pretty name in UI.
- Gate at call time, fresh per call: allow → run; deny → defensive refusal `ToolResult(ok=False, error="blocked by MCP permissions (set to Off)")`; ask → batch approval. Rug-pull/risk-floor resolve through the existing `gate_tool_test` (a live HubTool is available from the provider's own catalog).
- Batch-vs-invoke division of labor: the T4 hook (via T6's closure) resolves gates for the WHOLE turn batch up front — ask-calls go to ONE approval card; the closure applies side effects (session cache / always-allow write-through) and stamps per-call verdicts onto the provider via `apply_batch_decisions`. `invoke()` then consults those stamped verdicts first and only falls back to its own single-call gate + `approval_callback` when unstamped (defensive path; no callback → fail closed to deny). No double-prompting.
- Approval semantics (spec §11 verbatim): one approval per assistant turn listing ALL pending calls; per-call decisions + "Approve all"; options **Approve once / Approve for session / Always allow / Deny**. Approve-for-session = app-run lifetime, in-memory, keyed (server_key, tool_name), living ON the app-level control-plane singleton (`unified_mcp_service` — screens are rebuilt on navigation, the service is not). Always-allow → `set_tool_state(..., "allow", tool=hub_tool)` (stores the definition hash — rug-pull guard applies). Timeout (default 120s, `[mcp] approval_timeout_seconds`) → deny with the model-facing result `"user did not approve within the time limit; do not retry"`. Stop/new send/conversation switch/screen unmount while pending → deny. Every decision AND execution appends to the execution log (`initiator="agent"`; decision vocabulary: `allowed` (explicit/inherited allow), `approved` (once/session/always), `denied`, `denied-timeout`).
- Per-call tool timeout: `[mcp] tool_call_timeout_seconds` default 60 (distinct from the 45s lifecycle timeout), coerced reads (the "false"-string lesson → use existing coercion helpers for numerics too: try/except float fallback).
- Concurrency invariants: `MCPToolProvider.invoke()` runs on the agent worker thread and must NEVER touch Textual widgets; all MCP I/O submits to the main loop captured at provider construction; `.result(timeout)` bounds every wait; a dead/closed loop → `ToolResult(ok=False, error=...)`, never a hang. The approval Event wait must re-check cancellation every ≤1s (poll loop) so Stop stays responsive.
- App exit: `app.py on_unmount` gains best-effort MCP teardown (`local_mcp_control_service.client.disconnect_all()` when sessions exist) — currently missing entirely.
- Audit ▸ Executions: table ts | tool | initiator | decision | duration | outcome over `read_recent(200)`; filters decision/server/initiator/text (client-side); inspector entry detail (pretty-printed excerpt — UX-B8) + drill actions "Open tool" (→ Tools mode + selection) and "Adjust permission" (→ Permissions mode + selection). Audit ▸ Findings: server source only, read-only via the governance findings payload, fail-soft (absent on error); local/builtin show the standard server-only empty state.
- Textual 8.2.7 discipline (verified): namespace= on MCP* messages; await remove_children; no uncaught worker exceptions; markup=False on derived text; every Button tooltipped; dual-layer geometry CSS (bundle via build_css.py only); real-bundle harness assertions for new interactive widgets; DataTable two-click selection in tests; row-key cursor re-seat on rebuild (Phase 4 Critical — reuse the helper pattern).
- Test commands (FOREGROUND; no `timeout` cmd; worktree has no venv): `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest <paths> -q` from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase5`.
- Known dev baseline: 2 pre-existing Library snapshot-timeout failures in `test_destination_visual_parity_correction.py` — not regressions, do not fix.
- Verified seam facts (planning session 2026-07-17): app singletons `app.unified_mcp_service` / `local_mcp_control_service` built once in `app.py:3670-3706`; screens read them via `getattr(self.app_instance, "unified_mcp_service", None)`; `ToolProvider` Protocol is SYNC (`list_catalog/load_schema/invoke`, `Agents/tool_catalog.py:60-67`); per-run registry composed in `console_agent_bridge._compose_run_registry_and_allowed` (~:472); the per-turn call batch materializes at `agent_runtime.py:267` before sequential dispatch; `ChatApprovalCard`/`TaskResumeState.pending_approval` is DEAD scaffolding (no producer, no button handlers) — Phase 5 makes it live; `BuiltinToolProvider.invoke`'s `asyncio.run` pattern is NOT reusable for MCP (loop-bound sessions); `run_coroutine_threadsafe` precedent at `console_provider_gateway.py:520`.

## File Structure

- Create `tldw_chatbook/MCP/tool_naming.py` (T1), `tldw_chatbook/Agents/mcp_tool_provider.py` (T3), `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py` (T7).
- Modify `unified_control_plane_service.py` (T2), `Agents/agent_runtime.py` + `agent_models.py` (T4), `Widgets/Chat_Widgets/chat_approval_card.py` + `Chat/console_chat_controller.py` + `UI/Screens/chat_screen.py` (T5), `Chat/console_agent_bridge.py` + `Chat/console_display_state.py` + `app.py` (T6), `mcp_workbench.py` (T7/T8), CSS (T9).

---

### Task 1: Tool naming (`MCP/tool_naming.py`) — pure module

**Files:** Create `tldw_chatbook/MCP/tool_naming.py`; Test `Tests/MCP/test_tool_naming.py`

**Interfaces:**
- Produces (exact — T3 depends):
  - `def sanitize_component(text: str) -> str` — keep `[a-zA-Z0-9_-]`, replace runs of anything else with `_`, strip leading/trailing `_`, empty → `"x"`.
  - `def llm_tool_name(server_key: str, tool_name: str) -> str` — `f"mcp__{sanitize_component(server_label_part)}__{sanitize_component(tool_name)}"` where `server_label_part` is server_key with its `local:`/`builtin:` prefix dropped; if the result exceeds 64 chars, truncate to 55 and append `_` + first 8 hex of `sha256(f"{server_key}::{tool_name}")`.
  - `def dedupe_names(names: list[str]) -> list[str]` — stable; post-sanitization duplicates get `_2`, `_3`… suffixes (re-truncating with hash if the suffix overflows 64).
- [ ] **Steps 1-5**: failing tests (sanitization table incl. unicode/spaces/dots; 64-char truncate-with-hash determinism; collision suffixes stable and unique; round-trip uniqueness property for distinct inputs) → RED → implement (~70 lines) → GREEN (`Tests/MCP/ -q`) → commit `feat(mcp): LLM-safe tool name sanitization with truncate-hash and collision suffixes`.

---

### Task 2: Control-plane bridge seam + config knobs + session approvals

**Files:** Modify `tldw_chatbook/MCP/unified_control_plane_service.py`; Test `Tests/MCP/test_control_plane_bridge.py`

**Interfaces:**
- Produces (exact — T3/T5 depend):
  - `async execute_hub_tool(self, server_key: str, tool_name: str, arguments: dict | None = None, *, initiator: str = "test", decision: str = "allowed", timeout_seconds: float | None = None) -> dict` — generalization of `test_hub_tool`'s body: same local:/builtin: routing and error semantics, but records with the given `initiator`/`decision` and uses `timeout_seconds` (default → `_tool_call_timeout()`). `test_hub_tool` becomes a thin delegate (`initiator="test"`, `decision="allowed"`, lifecycle timeout) — its existing tests must stay green unmodified.
  - `def _tool_call_timeout(self) -> float` — `[mcp] tool_call_timeout_seconds`, default 60.0, try/except fallback (mirror `_lifecycle_timeout`).
  - `def approval_timeout_seconds(self) -> float` — `[mcp] approval_timeout_seconds`, default 120.0, same guard.
  - Session approvals (in-memory, app-run lifetime — the service is an app singleton; document non-persistence): `def approve_for_session(self, server_key: str, tool_name: str) -> None`, `def is_session_approved(self, server_key: str, tool_name: str) -> bool`, `def clear_session_approvals(self) -> None`.
  - `def record_tool_decision(self, server_key: str, tool_name: str, *, decision: str, initiator: str = "agent", error: str | None = None) -> None` — best-effort execution-log append for approval decisions that never execute (denied/timeout): `build_record(..., ok=False, duration_ms=0, decision=decision, error=error)`; never raises.
- [ ] **Steps 1-5**: failing tests (execute_hub_tool records initiator="agent"/decision="approved" on the JSONL record; test_hub_tool delegate keeps its exact old behavior — run the existing test file unmodified; timeout knob honored + garbage config falls back; session approval set/check/clear; record_tool_decision writes denied records and survives log failure) → RED → implement → GREEN (`Tests/MCP/ -q` incl. the untouched `test_control_plane_tool_execute.py`) → commit `feat(mcp): bridge execute seam, per-call/approval timeout knobs, session approvals`.

---

### Task 3: `Agents/mcp_tool_provider.py` — the task-201 provider

**Files:** Create `tldw_chatbook/Agents/mcp_tool_provider.py`; Test `Tests/Agents/test_mcp_tool_provider.py`

**Interfaces:**
- Consumes: T1 naming; T2 seams; `gate_tool_test`/`gate_tool_test_by_key`, `get_kill_switch`, `effective_tool_states`; `hub_tool_catalog` derivations (`local_tools_from_record`, `builtin_tools_from_inventory`); `ToolCatalogEntry/ToolSchema/ToolResult` from `Agents/agent_models.py`.
- Produces (exact — T4/T5/T6 depend):
  - `class MCPToolProvider:` `__init__(self, *, service, main_loop: asyncio.AbstractEventLoop, approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]] | None = None)` — `service` = the app's `unified_mcp_service`; `main_loop` captured at construction (the Textual loop). SYNC Protocol methods.
  - `@dataclass(frozen=True) class MCPPendingCall: llm_name: str; server_key: str; tool_name: str; server_label: str; arguments: dict; reason: str  # ask|config_changed|risk_floored`
  - `compose_catalog(self) -> None` — called once at registration (composition time, main loop): kill switch on → empty; else build eligible HubTools (local records via the service's catalog path + builtin inventory; skip `state == "deny"` via one `effective_tool_states` batch), build the name maps via T1 (`llm_tool_name` + `dedupe_names`), cache `list[ToolCatalogEntry]` + `{llm_name: (HubTool, EffectiveToolState)}`. `not_connected_count` property for T6's affordance. (Satisfies task-201's don't-re-list-per-lookup note.)
  - `list_catalog(self)` / `load_schema(self, tool_id)` — from the cache; schema passthrough of `input_schema` (None → `{"type": "object", "properties": {}}`).
  - `apply_batch_decisions(self, decisions: dict[str, str]) -> None` / `consume_decision(self, llm_name: str) -> str | None` — per-turn verdict stamps set by T6's closure (values `approve_once|approve_session|always_allow|deny|timeout`); consumed (removed) on first read so a later turn re-gates.
  - `pending_gate_for(self, llm_name: str, args: dict) -> MCPPendingCall | None` — used by T6's closure: resolves the gate on the main loop and returns a pending-call descriptor when the effective state is ask (None for allow/deny — the closure lets those flow to invoke's own gate).
  - `invoke(self, tool_id: str, args: dict) -> ToolResult` — WORKER THREAD. Order: (1) stamped verdict from `consume_decision` → apply side effects (session cache via T2 / `set_tool_state` with the live HubTool for always_allow) then execute, or refuse for deny/timeout (timeout uses the exact model-facing copy); (2) no stamp → fresh gate via main-loop submit (`gate_tool_test(hub_tool)`): deny → refusal ToolResult + `record_tool_decision(decision="denied", initiator="agent")`; session-approved or allow → execute; ask → `approval_callback` when set (single-call fallback list), else deny (fail closed). Execute = `asyncio.run_coroutine_threadsafe(service.execute_hub_tool(..., initiator="agent", decision=...), self._main_loop).result(timeout=tool_call_timeout + 5)`; every exception (incl. concurrent.futures.TimeoutError, closed loop) → `ToolResult(ok=False, error=str(exc)[:300])` — NEVER raises, NEVER hangs unbounded. Results: dict → `json.dumps(redact_mapping(...), default=str)` truncated to 4000 chars; non-text placeholder rule: a dict containing image/blob content types → `"[image result — not yet supported]"` (defensive key sniff: `type` in {"image","blob"} entries).
- [ ] **Steps 1-5**: failing tests (fake service + a real running loop in a thread for the submit path: compose filters deny + kill switch; name collision dedupe; invoke allow-path returns content; deny refusal + decision record; ask with no callback → fail closed; ask callback approve_once executes and records "approved"; always_allow calls set_tool_state with the HubTool; timeout from a hanging coroutine returns error ToolResult within bound; closed-loop error path) → RED → implement → GREEN (`Tests/Agents/ Tests/MCP/ -q`) → commit `feat(agents): MCPToolProvider bridging the agent thread to main-loop MCP execution (task-201)`.

---

### Task 4: Agent-runtime batch-review hook

**Files:** Modify `tldw_chatbook/Agents/agent_models.py` (LoopDeps), `tldw_chatbook/Agents/agent_runtime.py`, `tldw_chatbook/Agents/agent_service.py`; Test `Tests/Agents/test_agent_runtime_review_hook.py`

**Interfaces:**
- Produces: `LoopDeps` gains `review_tool_calls: Callable[[list[ToolCall]], dict[str, str]] | None = None` (keyed by `call.name`; values `"proceed"` or a refusal string). In `run_agent_loop`, immediately after `calls = list(turn.tool_calls)` (~:267) and the cancel check: when the hook is set and `calls` non-empty, call it ONCE with the full batch; for any call whose verdict is not `"proceed"`, skip dispatch and append the verdict string as that call's tool result (same role/tool_call_id shaping as normal results). Everything else (including `deps.invoke_tool`) unchanged; hook absent → identical behavior (all existing agent tests must stay green unmodified).
- `AgentService.run_turn`/`_run_one` thread the hook through from a new optional `AgentService(..., review_tool_calls=None)` ctor arg (mirroring how `should_cancel` flows). NOTE: the hook is generic runtime surface — MCP specifics stay in T5's closure.
- [ ] **Steps 1-5**: failing tests (hook receives the full parallel batch before any invoke; non-proceed verdict skips invoke and the model sees the verdict text as the tool result with correct call_id shaping; hook=None → byte-identical message flow vs. today, pinned by an existing-behavior test; hook exception → treated as proceed-nothing/deny-all? — resolve: hook exceptions are caught, logged, and treated as "proceed" for non-MCP safety [the hook must fail open for the runtime, closed inside the MCP closure]) → RED → implement → GREEN (`Tests/Agents/ -q` full) → commit `feat(agents): optional pre-dispatch batch review hook on the agent loop`.

---

### Task 5: Console batch-approval flow (make the dead card live)

**Files:** Modify `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Screens/chat_screen.py`; Test `Tests/UI/test_console_mcp_approval.py` (+ extend `Tests/Chat/` controller tests)

**Interfaces:**
- Consumes: T3 `MCPPendingCall` + the `approval_callback` seam; T2 `approval_timeout_seconds`, `approve_for_session`, `set_tool_state` (via the service), `record_tool_decision`.
- Produces:
  - `ChatApprovalCard` extended for batches: `set_batch(self, calls: list[dict], *, timeout_seconds: float) -> None` renders one row per pending call (`pretty "server · tool"` + argument summary ≤80 chars, markup=False) with per-row `Select` of Approve once / Approve for session / Always allow / Deny (default Approve once) + global Buttons `Approve all` (`#approval-approve-all`), `Submit` (`#approval-submit`), `Deny all` (`#approval-deny-all`) — all tooltipped; posts `ApprovalDecided(decisions: dict[str, str])` (namespace as needed). Existing single-approval API stays (unused but harmless).
  - Controller plumbing (`console_chat_controller.py`): `request_mcp_approvals(self, pending: list[MCPPendingCall]) -> dict[str, str]` — called FROM THE WORKER THREAD (this is the closure handed to `MCPToolProvider.approval_callback` in T6): creates a `threading.Event` + shared decisions dict, `app.call_from_thread` to surface the card (sets `pending_approval` state → `sync_task_resume_state()` — the dead scaffolding's producer at last), then polls `event.wait(1.0)` in a loop re-checking run cancellation and a deadline (`approval_timeout_seconds`); resolution paths: card decision → decisions dict + set; timeout → all remaining `"timeout"`; stop/new send/conversation switch/screen unmount → `"deny"` (wire the existing cancel/reset seams to set the event). Clears the card state afterwards (call_from_thread).
  - Decision application semantics live in T3's invoke (side effects) — the controller only transports decisions. Timeout deny result copy (model-facing, exact): `"user did not approve within the time limit; do not retry"`.
- [ ] **Steps 1-5**: failing tests (card renders N rows + buttons with tooltips; Approve all sets every row; ApprovalDecided carries per-name decisions; controller round-trip with a scripted card interaction on the pilot thread while a real worker thread blocks — decisions arrive, worker unblocks; timeout path returns "timeout" for all within deadline+slack; cancellation path returns deny; pending card cleared after resolution) → RED → implement → GREEN (`Tests/UI/test_console_mcp_approval.py` + touched controller suites + `Tests/UI/test_chat_approvals_and_resume.py` unmodified) → commit `feat(console): live batch approval flow for MCP tool calls`.

---

### Task 6: Bridge wiring (registry, affordance, teardown)

**Files:** Modify `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_display_state.py`, `tldw_chatbook/UI/Screens/chat_screen.py` (inspector row + provider construction), `tldw_chatbook/app.py` (on_unmount teardown); Test extend `Tests/Chat/test_console_agent_bridge.py`, `Tests/UI/` chat-screen suites

**Interfaces:**
- `_compose_run_registry_and_allowed` registers an `MCPToolProvider` when: the app exposes `unified_mcp_service`, kill switch off, and `compose_catalog()` yields ≥1 entry. Provider constructed per run (cheap — catalog composed once) with `main_loop` = the running Textual loop captured BEFORE `asyncio.to_thread` (the bridge/controller passes it in), `approval_callback` = the T5 controller closure. MCP tool names join the run's allowed-tools set (respect existing collision precedence: builtins/runtime names win — exclude colliding MCP names like `_non_colliding_skill_entries` does).
- The `review_tool_calls` closure (T4 hook, built here, handed to AgentService): for each ToolCall in the batch owned by the MCP provider, `provider.pending_gate_for(name, args)`; collect the MCPPendingCalls; when any, ONE `controller.request_mcp_approvals(pending)` round-trip (T5); apply side effects + `provider.apply_batch_decisions(decisions)`; return verdicts — `"proceed"` for approved/allow/deny-flowing calls (deny refusals surface through invoke's gate so the refusal copy and log records stay single-sourced), and the timeout/deny model-facing strings ONLY when the whole call should be skipped without invoking (timeout → the exact spec copy). Non-MCP calls always `"proceed"`.
- Console inspector: `ConsoleDisplayRow("MCP", f"{n} tools ready" | f"{m} servers enabled, not connected", status=...)` fed from the provider's composition (or absent when no MCP service/kill switch on). Follow `_console_tool_count` wiring (`chat_screen.py:4886-4899`).
- `app.py on_unmount`: best-effort `disconnect_all()` on the local MCP client when sessions exist (pattern-match the sibling teardown blocks; guarded getattr + try/except).
- [ ] **Steps 1-5**: failing tests (registry includes MCP entries when eligible/absent when kill switch on; name collisions excluded builtin-first; inspector row renders both states; unmount teardown calls disconnect_all best-effort) → RED → implement → GREEN (bridge + chat screen suites + `Tests/Agents/ -q`) → commit `feat(console): register MCPToolProvider per run with kill-switch gate and connect affordance`.

---

### Task 7: Audit mode — Executions canvas

**Files:** Create `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py`; Modify `mcp_workbench.py` (replace the audit placeholder; feed from `_sync_children`); Test `Tests/UI/test_mcp_audit_mode.py`, extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- `MCPAuditMode(Vertical)`: filter bar (`#mcp-audit-filter-text` Input; `#mcp-audit-filter-decision` Select over the decision vocabulary + All; `#mcp-audit-filter-initiator` Select test/agent/system + All — Select mount-echo + width lessons apply: bundle-layer width rules from day one); DataTable `#mcp-audit-table` (When | Tool | Initiator | Decision | Duration | Outcome; row key = stable synthetic index; plain Text cells; newest first; cursor re-seat by key on rebuild); empty state "No tool executions recorded yet." `async update_entries(self, entries: list[dict]) -> None` (the raw `read_recent` dicts, client-side filtering with cached full list); `EntrySelected(index)` message. Duration via the T-A-batch ms→s helper (import from `mcp_inspector` or move it to a shared module-level util in this task).
- Workbench: `_sync_audit_mode` reads `service.execution_log.read_recent(200)` (guarded — no log → empty); inspector `show_audit_entry(entry: dict)` — pretty-printed detail (ts, tool, initiator, decision, duration, error, redacted args/excerpt via `json.dumps(indent=2)` in a scrollable block, markup=False — UX-B8) + Buttons `Open tool` (`#mcp-audit-open-tool`, → tools mode + select the row-key tool when present) and `Adjust permission` (`#mcp-audit-adjust-permission`, → permissions mode + select) — both tooltipped, both no-op-with-toast when the tool no longer exists.
- [ ] **Steps 1-5**: failing tests (rows newest-first; filters narrow by decision/initiator/text; empty state; selection → inspector detail with pretty JSON; drills switch mode + select or toast-fallback; real-bundle-CSS geometry for table+filters) → RED → implement → GREEN → commit `feat(mcp-hub): audit executions canvas with filters and drill-through inspector`.

---

### Task 8: Audit mode — server Findings sub-view

**Files:** Modify `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py`, `mcp_workbench.py`; Test extend `Tests/UI/test_mcp_audit_mode.py`

**Interfaces:**
- Audit canvas gains a sub-view strip (two toggle Buttons `Executions` / `Findings`, Library-idiom cycling style, default Executions). Findings: server source only — workbench fetches via the governance payload path used in Phase 4 T8 (reuse the cached-by-(source,target) approach; findings key from `server_unified_service` governance/audit payload — defensive raw-dict reads: severity/type/message/remediation when present), rendered as a DataTable (Severity | Type | Message) + inspector detail with suggested remediation (markup=False). Local/builtin source: the standard "Findings come from a tldw_server target." empty state. Fetch failure → fail-soft absent listing with a retry hint. NO client-side fix actions this phase (readiness drills deferred with Phase 6 polish — record).
- [ ] **Steps 1-5**: failing tests (sub-view toggle; findings render from a fake payload; local shows server-only copy; fetch failure fail-soft) → RED → implement → GREEN → commit `feat(mcp-hub): audit findings sub-view for server targets`.

---

### Task 9: CSS, bindings, full gate

**Files:** Modify `tldw_chatbook/UI/Screens/mcp_screen.py` (MCP_SHORTCUTS entries for audit if any), CSS source + rebuilt bundle; Test bundle-parity + full gate

- [ ] Dual-layer rules for `#mcp-audit-table` (height discipline), filter Selects (width — the Phase 3 0×0 lesson), approval-card geometry if needed; regenerate bundle; parity tests; fidelity check.
- [ ] Full gate:
`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/ Tests/Agents/ Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_tools_mode.py Tests/UI/test_mcp_schema_form.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_audit_mode.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_mcp_profile_form.py Tests/UI/test_mcp_server_mutations.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_destination_shells.py Tests/UI/test_chat_approvals_and_resume.py -q`
Expected: green (the 2 pre-existing Library failures live in `test_destination_visual_parity_correction.py`, outside this command).
- [ ] Commit `feat(mcp-hub): audit styles, Phase 5 gate`

**Post-task (controller-owned):** live QA — seed the P4 HOME + a real stdio MCP server profile that can execute (or builtin tools), drive a Console agent turn calling an MCP tool through allow/ask/deny (captures: approval card with batch rows, session approval surviving a second call, denied result in transcript, Audit Executions with agent-initiated records, Findings empty state, MCP inspector row affordance). User screenshot approval gates the PR.

## Out of scope (recorded)

Server-source tool execution + access-context plumbing + task-234 shape verification (Phase 6); legacy Chat-tab ToolExecutor bridge (superseded by the Console stack — spec §11's ToolExecutor framing adapted per user decision 2026-07-17); Findings inline remediation actions + readiness drills (Phase 6); approval persistence across app restarts (session cache is in-memory by spec); UX B-batch items not named here (Phase 6 polish); colored-state-words design decision (Phase 6 with the polish pass).
