# Mobile engine-binding feasibility spike (spec §4.3) — 2026-10-15

**Status:** complete. **Verdict: Binding 1 (parent-process service) is feasible
as-is for agent-mode chat** — one tool call plus one approval decision round-tripped
App-free against a scratch DB.

The probe is **throwaway**: `/tmp/mobile_binding_spike.py` (never committed; this
document is the durable record). It imports only the worktree package plus
`Tests/console_provider_doubles.py` helpers and uses a tmp SQLite DB.

## Environment

- Worktree `feat/served-auth-phase1a` @ `ab6bc49759` (Task 7's textual-free
  `Chat/approval_constants.py` included), macOS arm64, `.venv/bin/python`
  (CPython 3.12.11), sqlite 3.49.1.
- Runs were made with a fresh `HOME=/tmp/spike-home` +
  `TLDW_CONFIG_PATH=/tmp/spike-home/.config/tldw_cli/config.toml` **and** with the
  real user HOME; both passed identically. The isolation was **not required** — the
  probe path never hit the `RecoveryRequired` config fence (the 209 pre-existing
  failures in `Tests/Agents/test_local_tool_provider.py` are environmental and were
  not touched).
- No source changes were made or needed; this is a pure probe.

## What was constructed (Binding 1, all App-free)

- `ConsoleChatStore` with real persistence — `ChatPersistenceService(CharactersRAGDB(tmp/chat.db), workspace_registry=…)`
  (the `Tests/console_provider_doubles.persisted_console_store` idiom). A bare store
  is not enough: `submit_draft` refuses a manual send on a non-ephemeral session
  without the durable-turn adapter.
- A real `LocalWorkspaceRegistryService(WorkspaceDB(tmp/workspaces.db))` with one
  workspace and one folder binding; the Console session is created with that
  `workspace_id`. **Required:** sessions in the default workspace (no admitted
  binding) get **no** `fs_*`/`git_*` tools at all — an empty `admitted_roots`
  sequence drops the whole path-tool family (`Agents/local_tool_provider.py`,
  `__init__` spec selection). A workspace-bound chat is the realistic mobile case.
- Session posture: `store.set_session_project_instruction_state(session.id,
  ProjectInstructionControlState.legacy_disabled())` — a workspace-bound chat
  without the startup-`AGENTS.md` dispatch. That flow wants a UI confirm callback
  (`confirm_project_instruction_dispatch` / `select_project_instruction_binding`);
  with it armed and unwired the run cancelled before the provider call. Mobile v1
  ships without project instructions (spec's stated scope), so this matches.
- `AgentRunsDB(tmp/runs.db)` + `ConsoleAgentBridge(agent_runs_db=…, store=…,
  provider_gateway=<fake>)` — only the three required kwargs; `registry` defaults
  to `BuiltinToolProvider`, `message_store`/`ensure_run_hooks`/`app_config`/etc.
  stayed `None`.
- `ConsoleChatController(store=…, provider_gateway=…, provider="openai",
  model="gpt-4o-mini", agent_bridge=bridge, agent_runtime_enabled=True)` — every
  other constructor parameter left at its default.
- Post-construction instance seams (not constructor parameters, but mandatory):
  - `controller.app = HeadlessApp(...)` — a duck-typed object, **not** a Textual
    App: `call_from_thread` (only method the approval path calls),
    `unified_mcp_service` (real `UnifiedMCPControlPlaneService` built with all-`None`
    stores — a plain class, not a widget; with no permission store every tool
    resolves fail-closed to `"ask"`, which is exactly the approval surface), and
    `workspace_registry_service`. Other App attributes are read via
    `getattr(..., None)` and tolerate absence.
  - `controller.set_pending_approval = <callback>` — the card-mount bridge (below).
- Fake provider gateway implementing `ConsoleProviderGatewayProtocol`
  (`resolve_for_send` + `stream_chat`). **The conftest `StreamingGateway`
  (`Tests/Chat/conftest.py`) is unsuitable for agent mode** — justified: its
  `stream_chat` yields only `("hel", "lo")` text chunks and can never emit a native
  tool call, so it can only exercise plain chat. The probe's fake yields plain
  chunks plus one `ProviderToolCalls` item (OpenAI-shape dicts, the bridge's Plan-B
  contract, `Chat/console_provider_gateway.py:2294`) when `tools=` is passed, and a
  final text answer on the follow-up call that carries the tool result.

## Seam inventory — what had to be non-None

Constructor (`Chat/console_chat_controller.py:3907-3972`, ~50 params):

| Seam | Value used | Required for the agent round-trip? |
|---|---|---|
| `store` | persisted `ConsoleChatStore` | **Yes** (durable-turn refusal otherwise) |
| `provider_gateway` | agent-capable fake | **Yes** |
| `agent_bridge` | real `ConsoleAgentBridge` | **Yes** — `None` means plain chat only |
| `agent_runtime_enabled` | `True` (the default) | Yes in effect; left explicit |
| `provider` / `model` | `"openai"` / `"gpt-4o-mini"` | **Yes in practice** — must be a native-tools-capable execution key (`Agents/native_tools.py::NATIVE_TOOLS_PROVIDERS`; `llama_cpp` without `reasoning_replay` silently gets no tools) and a capability-known model id (an unknown id degrades the first-request plan to `find_tools`/`load_tools` discovery mode — the tool is never directly advertised) |
| all other params (`skills_service`, `rag_capture_provider`, `default_session_settings`, `library_provider_factory`, `context_repository`, `turn_context_provider`, `ensure_run_hooks`, `scratch_spaces`, `buddy_sink`, `confirm_/select_project_instruction_*`, …) | defaults/`None` | **No** — the round-trip ran with every one defaulted |

Post-construction instance seams:

| Seam | Why it must be set |
|---|---|
| `controller.app` (duck-typed) | `request_mcp_approvals` **fails closed immediately** when `app is None`: every key returns `deny` + `unresolved_keys` and no round is ever armed (`Chat/console_interrupt_rounds.py:3892-3908`). `unified_mcp_service` gates the local fs_* provider composition (`_compose_local_provider`) and the builtin gate; `workspace_registry_service` gates `capture_run_admitted_workspace_roots` (→ fs_* advertising). |
| `controller.set_pending_approval` | The mount callback the round is delivered through (`run_round` → `app.call_from_thread(setter, payload)`); without it the round arms but nothing observes it. |
| `controller.resolve_pending_approval` | Called by the service (not set) — the decision entry point. |

How escalation is observed and decided headless (replicated minimum):

1. The controller binds `functools.partial(self.request_mcp_approvals,
   session_id=…)` into the run's review hooks (`build_tool_review_hook` /
   `build_local_review_hook`). With a fresh permission store the local hub
   `local:__local__` inherits `global_default = "ask"`, so an `fs_list` call
   becomes an `MCPPendingCall` row and the hook calls the bridge from the agent
   worker thread.
2. `request_mcp_approvals` (`console_interrupt_rounds.py:3787`) registers a round,
   builds the payload (`round_id`, `calls` with `llm_name`/`call_id`/`options`/
   `arguments`, `timeout_seconds`, `deadline_monotonic`) and delivers it via
   `app.call_from_thread(controller.set_pending_approval, payload)`; then polls a
   `threading.Event` at 1.0 s granularity until decided / cancelled / timed out
   (ADR-067: default timeout 0 = wait indefinitely — right for a phone).
3. The headless service captures the payload (this is exactly what the mobile API
   would push to the phone) and answers with
   `controller.resolve_pending_approval({call_id: "approve_once"},
   round_id=payload["round_id"])` — lock-guarded merge into the round's shared
   decisions dict + `event.set()` (`console_interrupt_rounds.py:4659-4749`). The
   worker wakes, returns an `ApprovalDecisions` map
   (`Agents/approval_provenance.py` — a plain dict carrying `unresolved_keys` so a
   fail-closed deny is never audited as a user refusal), the hook stamps the
   provider (`apply_batch_decisions`), and `invoke()` executes the tool. The
   widget's mounted-card flow (`ChatScreen._set_console_pending_approval` →
   `ChatApprovalCard.ApprovalDecided` → the same `resolve_pending_approval`) is the
   same wire with a card in the middle; nothing about the round requires widgets.

## Agent round-trip result

**Exact outcome** (identical with isolated and real HOME; probe verdict
`AGENT_ROUND_TRIP_OK`, exit 0):

- `submit_draft` accepted; `run_state_for(session).status == COMPLETED`;
  the `agent_runs` row is `status="done"` with the final text persisted verbatim.
- Provider call 1: `tools=` passed, `fs_list` among 37 advertised tools.
- **One approval round raised**: `llm_name=fs_list`, `call_id=call-spike-1`,
  `server_key=local:__local__`, `reason="ask"`,
  `options=(approve_once, approve_session, always_allow, deny)`, `round_id`
  present. Lifecycle observed: 1 mount callback + 1 clear callback
  (`set_pending_approval(None)`) after resolution.
- Decision: `resolve_pending_approval` invoked from the main event loop (via
  `call_soon_threadsafe`, modeling the server answering the phone asynchronously)
  → round released within one poll tick.
- **The tool executed for real**: `fs_list` listed the bound folder — the tool
  result delivered to provider call 2 contains `src/` and `README.md`.
- Provider call 2: message list now carries the `role="tool"` result; final
  assistant answer streamed and persisted.
- Multi-open (spec §4.3 shared requirement): a second `CharactersRAGDB` opened on
  the same `chat.db` while the store's connection was live — ok (open-time schema
  migration + concurrent read; WAL multi-open behavior is fine).
- Teardown: `controller.shutdown()` within 10 s, store `quiesce_connections` +
  close, `AgentRunsDB.close()` — all clean.

How measured: the probe asserts each item above and prints a structured report
(captured in the spike working notes); the verdict conjunction is
accepted ∧ completed ∧ ≥1 round ∧ round_id present ∧ tool result seen by the
provider ∧ real folder listing in the result ∧ final answer persisted ∧ all
resolve calls succeeded.

## Transitive imports

`python -c "import sys; import tldw_chatbook.Chat.console_chat_controller as m;
print([n for n in sys.modules if n.startswith('textual')])"` — **98 textual
modules load transitively** in a plain process (no App instantiated; harmless, but
real import cost). Task 7 removed the `chat_approval_card` edge; two remain,
confirmed by blocking `textual` and reading the stack:

1. `console_chat_controller.py:119` → `console_chat_store.py:80` →
   `console_chat_fork.py:41` → `console_session_settings.py:111`
   (`from tldw_chatbook.UI.character_display_text import …`) → `UI/__init__.py`
   (installs textual compatibility shims) → `textual`.
2. `console_chat_controller.py:167` → `Agents/agent_service.py:146` →
   `agent_runtime.py:129` → `project_instruction_runtime.py:40` →
   `Agents/tool_catalog.py:44` → `library_rag_tool_provider.py:48` →
   `library_local_rag_search_service.py:29` → `library_rag_service.py:19`
   (`from tldw_chatbook.UI.destination_recovery import …`) → `textual`.

That is data, not failure: nothing in the round-trip instantiates a widget. If
Phase 1b wants the import gone, two targeted refactors (move
`UI.character_display_text` and `UI.destination_recovery` to non-UI modules, or
lazy-import them at the two call sites) would sever both chains — optional
follow-up, not a gate.

## Recommendation

**Binding 1 feasible as-is** (agent-mode included), with these named conditions
the Phase 1b plan should carry:

1. `MobileChatService` supplies a duck-typed App context: `call_from_thread`
   (direct invocation is fine off the UI loop), `unified_mcp_service` (real,
   headless-constructed; store-less is fail-closed-correct and still
   approval-capable), and `workspace_registry_service`. All are plain services —
   no Textual anywhere.
2. Approval surface = capture `set_pending_approval` payloads (round_id + calls),
   push to the phone, resolve via `controller.resolve_pending_approval`. Deny on
   disconnect is already fail-closed; the default indefinite wait matches push
   latency.
3. Provider/model selection must be native-tools-capable, and per-session fs_*
   tools require a workspace binding (default-workspace chats have no path tools —
   the mobile UX must not offer folder tools for those sessions).
4. Project-instruction dispatch stays out of mobile v1 (`legacy_disabled()`), or
   its confirm callbacks get wired to phone prompts later.
5. Test rigging: the shared `StreamingGateway` double cannot drive agent mode;
   Phase 1b tests need a `ProviderToolCalls`-capable fake (the probe's shape).

**What this costs Phase 1b:** the API shape can be frozen around the parent-process
model (send → stream events → approval payload → resolve → continuation) with no
child-process management and no parent↔child bridge for mobile-initiated
conversations. **Binding 2 evidence, for the record:** `Chat/console_runtime.py::
ensure_chat_controller`'s "viewless-from-birth" path (lines 4525-4541) clears view
hooks when no view is attached — but the runtime still binds a real `self._app`'s
callables (`_apply_chat_dictionaries_for_app`, `_default_session_settings_for_app`,
`_library_provider_for_app`, `provider_config`, `wake.wire(app=…)`), so Binding 2
means a full headless App instance plus process-per-session plumbing. It remains
the fallback only if v2 wants phone-side decisions on desktop-running sessions
(the spec's stated Phase 2 relay), not for v1.
