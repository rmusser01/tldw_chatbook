# Served Mobile Companion Surface (Phase 1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A phone-first page at `/m` — conversation list, streaming transcript, composer, and approvals for mobile-initiated agent chats — sharing Phase 1a's auth, installable as a PWA, with the full terminal one tap away.

**Architecture:** Per the spike verdict (Binding 1 feasible as-is, agent-mode included), a `MobileChatService` in the served parent process drives `ConsoleChatStore` + `ConsoleAgentBridge` + `ConsoleChatController` directly with a duck-typed App context — no child processes, no Textual App. The store is the streaming source of truth (the controller appends chunks via `append_stream_chunk`); the service polls run-state + message content on a tick and emits JSON events over one authenticated WebSocket. Thin REST routes cover history and send. A vanilla-JS page under `Web_Server/static/mobile/` consumes the API; completion-level cross-surface sync is client-side cursor polling.

**Tech Stack:** aiohttp (existing), vanilla JS/CSS (no build step, no framework, no service worker), WebAuthManager session + CSRF (Phase 1a), PWA manifest + generated PNG icons (stdlib-only generator script).

**Spec:** `Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md` (§4 is this phase; §3 landed in Phase 1a / PR #3127)
**Spike (authoritative construction recipe):** `Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md` — its "named conditions" 1-5 are binding on this plan. The throwaway probe `/tmp/mobile_binding_spike.py` (if still present) is reference-only; do not commit it.

## Global Constraints

- Spike conditions carried verbatim: (1) duck-typed App context = `call_from_thread` + real headless `UnifiedMCPControlPlaneService` + `workspace_registry_service`; (2) approval surface = capture `set_pending_approval` payloads, resolve via `controller.resolve_pending_approval`; (3) provider/model must be native-tools-capable (`Agents/native_tools.py::NATIVE_TOOLS_PROVIDERS`; unknown model ids degrade to discovery mode), and default-workspace sessions have NO `fs_*`/`git_*` tools — the UI must not offer folder tools for them; (4) project-instruction dispatch stays `legacy_disabled()` for mobile sessions; (5) tests use a `ProviderToolCalls`-capable gateway double — the conftest `StreamingGateway` cannot drive agent mode.
- All `/api/v1/mobile/*` routes sit behind the existing `_web_auth_middleware` (session + CSRF header for POSTs; NOT in `_public_auth_paths`); the WS authenticates via `authenticate_request(facts, websocket=True)` with the `csrf.<token>` + `chatbook-v1` subprotocols (constants `CSRF_HEADER_NAME`, `WEBSOCKET_PROTOCOL` in `Canvas/web_auth.py`).
- No secrets (tokens, cookies, CSRF) in logs or event payloads; message content is user data, not credentials.
- `[web_server] mobile_surface = true` (default on; escape hatch) read via `get_cli_setting` in `create_server`; `/m` and the API 404 when false.
- Vanilla JS only, no CDN, no service worker, no Tailwind; plain CSS in `Web_Server/static/mobile/` (outside TCSS token governance, same as the existing served shell).
- Targeted pytest runs only (never a full sweep without operator opt-in). Known environment baselines (pre-existing, not regressions): 11 failures in `Tests/Web_Server/test_web_auth_integration.py`, 4 in `test_web_server_dependency_gate.py`, 209 in `Tests/Agents/test_local_tool_provider.py` — all `RecoveryRequired`; new tests must avoid config-loading paths (`GET /`, textual.js, `create_server` with default canvas policy).
- Execute in the worktree `/Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a` on a branch off `feat/served-auth-phase1a` (Phase 1b depends on Phase 1a code; the PR targets `dev` after #3127 lands or stacks on it — coordinator decides at PR time).
- ADR required before code: Task 1 creates `backlog/decisions/234-served-mobile-companion-surface.md`.
- Testing venv: `/Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a/.venv/bin/pytest`.

---

### Task 1: ADR-234 and Backlog task

**Files:**
- Create: `backlog/decisions/234-served-mobile-companion-surface.md`
- Create via CLI: one Backlog task for Phase 1b (binary: `~/.local/bin/backlog-py`; the CLI also writes the task markdown under `backlog/tasks/` — commit both)

**Interfaces:**
- Consumes: nothing.
- Produces: ADR-234 path + Backlog task id referenced by later commits and Implementation Notes.

- [ ] **Step 1: Write ADR-234**

```markdown
# ADR-234: Served mobile companion surface

Date: 2026-10-15
Status: Accepted (implements spec Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md §4; binding per spike Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md)

## Context

The served terminal is a keyboard-first character grid — painful on a phone.
The operator's mobile use-cases are chat with agents, check-ins/approvals for
mobile-initiated conversations, and reading; full app usage stays on desktop.
The spike proved Binding 1 (parent-process service driving the existing
Console engine App-free) works for agent-mode including tool approvals.

## Decision

1. A phone-first page at `/m` (vanilla JS, PWA-installable, no service
   worker) served by the same process and the same Phase-1a auth; the
   terminal remains one tap away.
2. `MobileChatService` in the served parent process constructs
   ConsoleChatStore + ConsoleAgentBridge + ConsoleChatController directly
   with a duck-typed App context (call_from_thread + headless
   UnifiedMCPControlPlaneService + workspace_registry_service). One service
   instance; sends serialized per session.
3. Authenticated API under `/api/v1/mobile/`: REST for conversation list,
   cursor-paginated history, send, new conversation, cancel, approvals; one
   WebSocket (`/api/v1/mobile/stream`) carrying run/delta/approval/error
   JSON events. The store is the streaming source of truth; the service
   polls run-state and message content on a tick.
4. Approval surface: pending rounds captured from the controller's
   set_pending_approval seam, pushed as events, decided via
   resolve_pending_approval. Default posture fail-closed ("ask").
5. v1 limits (stated, not accidental): approvals only for mobile-initiated
   conversations; desktop activity reaches the phone via client cursor
   polling (completion-level), never live push; project-instruction
   dispatch stays disabled for mobile sessions; default-workspace sessions
   expose no fs_*/git_* tools.
6. Config: [web_server] mobile_surface = true (default on).

## Consequences

- New authenticated HTTP/WS surface reusing the existing session + CSRF
  middleware — no second auth path to audit.
- The parent process now opens the chat DBs (children already do; WAL
  multi-open verified by the spike). Children remain migration owners.
- The controller's ~50 constructor seams stay defaulted except the five
  the spike named; future Console changes may need the service updated.
- Phase 2 (passkeys, read-only library/notes, desktop->phone approval
  relay) is explicitly out of scope.

## Alternatives considered

- Binding 2 (headless child App per mobile session): reserved for the
  Phase-2 cross-surface relay; v1 needs no process management.
- Making the terminal mobile-friendly: ceiling is "less painful"; rejected
  as the primary path for the stated use-cases.
```

- [ ] **Step 2: Create the Backlog task**

```bash
~/.local/bin/backlog-py task create "Served mobile companion surface at /m: chat, approvals, PWA (Phase 1b)" \
  -d "Implements spec 2026-10-15 §4 per spike verdict (Binding 1): /m page, mobile API + WS, MobileChatService, PWA" \
  -s "In Progress"
```

Capture the printed task id as `<TASK-ID>`; add the ADR link:

```bash
~/.local/bin/backlog-py task edit <TASK-ID> --notes "ADR: backlog/decisions/234-served-mobile-companion-surface.md; spec §4; spike doc is the construction authority"
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/234-served-mobile-companion-surface.md "backlog/tasks/<TASK-ID> - ....md"
git commit -m "docs(adr): ADR-234 served mobile companion surface"
```

---

### Task 2: Agent-capable provider double

**Files:**
- Modify: `Tests/console_provider_doubles.py` (append; the file already provides `persisted_console_store` and `provider_resolution`)
- Test: covered indirectly by Task 3's service tests (this is test infrastructure; a direct unit test would only re-state the double)

**Interfaces:**
- Consumes: `provider_resolution(...)` (same file), `ProviderToolCalls` / `ProviderTurnMetadata` from `Chat/console_provider_gateway.py`.
- Produces: `AgentModeGateway` — a `ConsoleProviderGatewayProtocol` double:

```python
class AgentModeGateway:
    """Agent-capable gateway double: emits one native tool call when the
    caller passes tools=, then a final text answer once the tool result
    returns. resolve_for_send reports an execution key and model id that
    are native-tools-capable (NATIVE_TOOLS_PROVIDERS; capability-known id).
    The conftest StreamingGateway cannot drive agent mode (spike condition 5).
    """

    def __init__(self, *, tool_name: str = "fs_list",
                 final_answer: str = "MOBILE_AGENT_ROUND_TRIP_COMPLETE") -> None: ...
    async def resolve_for_send(self, selection) -> <ProviderResolution>: ...   # provider_resolution(ready=True, provider="openai", model="gpt-4o-mini", base_url="https://mobile.invalid/v1", visible_copy="")
    async def stream_chat(self, resolution, messages, **kwargs) -> AsyncIterator[str | ProviderToolCalls]: ...
    # stream_chat records calls in .stream_calls (tools_passed, tool_advertised,
    # saw_tool_result) and yields: text chunk -> ProviderToolCalls([{type:"function",
    # function:{name: tool_name, arguments: json {"path": "."}}}]) when tools is
    # passed and no role=="tool" message present; on the follow-up call carrying
    # the tool result it yields the final_answer text and ProviderTurnMetadata.
```

- [ ] **Step 1: Port the double** — transcribe the `/tmp/mobile_binding_spike.py` `AgentModeGateway` class (lines ~60-120) into `Tests/console_provider_doubles.py` with the constructor parameters above; keep the OpenAI-shape Plan-B tool-call dicts byte-equivalent to the probe's (the bridge's consumed contract, `Chat/console_provider_gateway.py:2294`).
- [ ] **Step 2: Verify import surface** — `.venv/bin/python -c "from Tests.console_provider_doubles import AgentModeGateway, persisted_console_store, provider_resolution; print('ok')"`
- [ ] **Step 3: Commit** — `git add Tests/console_provider_doubles.py && git commit -m "test(console): agent-capable provider gateway double (spike condition 5)"`

---

### Task 3: MobileChatService (engine binding per spike)

**Files:**
- Create: `tldw_chatbook/Web_Server/mobile_service.py`
- Test: `Tests/Web_Server/test_mobile_service.py`

**Interfaces:**
- Consumes (spike recipe, exact): `persisted_console_store` idiom → `ConsoleChatStore` over `ChatPersistenceService(CharactersRAGDB(path), workspace_registry=…)`; `LocalWorkspaceRegistryService(WorkspaceDB(path))`; `AgentRunsDB(path)` + `ConsoleAgentBridge(agent_runs_db=…, store=…, provider_gateway=…)`; `ConsoleChatController(store=…, provider_gateway=…, provider="openai", model="gpt-4o-mini", agent_bridge=…, agent_runtime_enabled=True)` with all other constructor params defaulted; `store.set_session_project_instruction_state(sid, ProjectInstructionControlState.legacy_disabled())`; `UnifiedMCPControlPlaneService` built with all-None stores (fail-closed "ask"); duck-typed `HeadlessAppContext` with `call_from_thread` (direct call — no UI loop), `unified_mcp_service`, `workspace_registry_service`; `controller.set_pending_approval = <capture callback>`.
- Consumes (read/write API): `store.sessions` (list of `ConsoleChatSession`: `.id`, `.title`, `.workspace_id`, `.persisted_conversation_id`), `store.create_session(...)` (line ~2247 — read its signature for title/workspace_id), `store.messages_for_session(session_id) -> list[ConsoleChatMessage]` (line ~12008), `store.get_message(message_id)`, `controller.submit_draft(draft, session_id=sid, origin=ConsoleSubmissionOrigin.MANUAL)`, `controller.run_state_for(session_id)` (line ~6118), `controller.resolve_pending_approval(decisions, round_id=…)`, `controller.shutdown()`.
- Produces:

```python
@dataclass(frozen=True)
class MobileEvent:            # WS wire shape; JSON-serializable via .to_wire()
    type: str                 # "run_started" | "delta" | "message_done" | "approval_requested" |
                              # "run_completed" | "run_failed" | "session_updated"
    session_id: str
    payload: dict             # delta: {"message_id", "offset", "text"}; approval: the captured
                              # round payload (round_id, calls[...] with llm_name/call_id/options/arguments,
                              # timeout_seconds); message_done/run_completed: final status
    seq: int                  # per-service monotonic

class MobileChatService:
    def __init__(self, *, data_dir: Path, provider_gateway) -> None: ...
    # lazy single construction of store/bridge/controller + HeadlessAppContext;
    # approval rounds land in self.pending_approvals (dict round_id -> payload)
    # via the set_pending_approval capture; a clear callback removes them.
    @property
    def sessions(self) -> list[dict]: ...          # [{id, title, workspace_id, has_folder_tools}] newest-first
    def session_messages(self, session_id: str, *, after_message_id: str | None = None) -> dict: ...
    # {"messages": [...], "cursor": last_message_id} — messages after the cursor only;
    #   each {id, role, content, status, created_at}
    async def create_session(self, *, title: str, workspace_id: str | None = None) -> dict: ...
    async def send(self, session_id: str, draft: str) -> None: ...   # serialized per session via asyncio.Lock;
    #   submits and returns; streaming is observed via events()
    def cancel(self, session_id: str) -> None: ...  # controller's stop path for the active run (find the
    #   stop/cancel entrypoint on the controller — the same one the TUI Stop button uses)
    def list_approvals(self) -> list[dict]: ...     # pending rounds across mobile sessions
    def decide_approval(self, round_id: str, decisions: dict[str, str]) -> bool: ...
    #   -> controller.resolve_pending_approval(decisions, round_id=round_id)
    async def events(self) -> AsyncIterator[MobileEvent]: ...  # fan-out via asyncio.Queue per subscriber;
    #   the pump task polls run_state_for(active sessions) + get_message(content length) at 0.5 s while
    #   any run is active, emitting incremental delta events (text beyond last offset) and
    #   message_done/run_completed transitions; approval_requested fires from the capture callback
    #   (thread-safe -> loop.call_soon_threadsafe into the queue)
    async def shutdown(self) -> None: ...           # controller.shutdown() + store quiesce + AgentRunsDB.close
```

- [ ] **Step 1: Write failing tests** — model on the spike's assertions (the service test IS the spike, productized):

```python
# Tests/Web_Server/test_mobile_service.py  (asyncio tests; tmp_path DBs; fresh-HOME not required per spike)
async def test_agent_round_trip_with_approval(tmp_path):
    svc = MobileChatService(data_dir=tmp_path, provider_gateway=AgentModeGateway())
    session = await svc.create_session(title="phone chat", workspace_id=<bound-workspace-id>)
    got = []
    async for event in svc.events():        # subscriber started in background task
        got.append(event)
    await svc.send(session["id"], "list the folder")
    # ...await run completion (poll got for run_completed with timeout 30 s)
    messages = svc.session_messages(session["id"])
    assert any(m["role"] == "assistant" and "MOBILE_AGENT_ROUND_TRIP_COMPLETE" in m["content"] for m in messages["messages"])
    approvals = [e for e in got if e.type == "approval_requested"]
    assert approvals and approvals[0].payload["calls"][0]["llm_name"] == "fs_list"
    assert svc.decide_approval(approvals[0].payload["round_id"], {approvals[0].payload["calls"][0]["call_id"]: "approve_once"}) is True

async def test_send_is_serialized_per_session(tmp_path): ...      # two concurrent sends -> both complete, in order
async def test_default_workspace_session_has_no_folder_tools(tmp_path): ...  # sessions[] has_folder_tools False when workspace_id is the console default
async def test_messages_cursor(tmp_path): ...                     # after_message_id returns only later messages
```

(The workspace fixture: one workspace + one folder binding via `LocalWorkspaceRegistryService(WorkspaceDB(tmp))`, mirroring the probe's `ws-mobile-spike` setup — read the probe lines for the exact binding calls.)

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Web_Server/test_mobile_service.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement `mobile_service.py`** per the Interfaces block; port construction verbatim from the spike doc §"What was constructed"; the delta pump and queue fan-out are the only new logic. Logging: session ids and statuses only — never message content, never credentials.
- [ ] **Step 4: Run to verify pass** — expected: all PASS within ~60 s (real agent round-trip).
- [ ] **Step 5: Commit** — `git commit -m "feat(web-server): MobileChatService — parent-process Console engine binding (ADR-234)"`

---

### Task 4: Mobile REST API

**Files:**
- Create: `tldw_chatbook/Web_Server/mobile_api.py`
- Modify: `tldw_chatbook/Web_Server/serve.py` (routes registration + `mobile_surface` gate + service lifecycle)
- Test: `Tests/Web_Server/test_mobile_api.py`

**Interfaces:**
- Consumes: `MobileChatService` (Task 3); middleware-injected `request["chatbook_browser_session"]`; Phase 1a's `_read_device_json` body guard; `get_cli_setting`.
- Produces (all authed; POSTs CSRF-enforced by the existing middleware):

| Route | Handler | Body/Query | Response |
| --- | --- | --- | --- |
| `GET /api/v1/mobile/conversations` | `handle_conversations` | — | `{"sessions": [...]}` from `svc.sessions` |
| `POST /api/v1/mobile/conversations` | `handle_new_conversation` | `{"title"?, "workspace_id"?}` | `201 {"session": {...}}` |
| `GET /api/v1/mobile/conversations/{sid}/messages` | `handle_messages` | `?after=<message_id>` | `{"messages": [...], "cursor"}` |
| `POST /api/v1/mobile/conversations/{sid}/messages` | `handle_send` | `{"text"}` (1..32 KiB) | `202 {"ok": true}` |
| `POST /api/v1/mobile/conversations/{sid}/cancel` | `handle_cancel` | — | `{"ok": bool}` |
| `GET /api/v1/mobile/approvals` | `handle_approvals` | — | `{"approvals": svc.list_approvals()}` |
| `POST /api/v1/mobile/approvals/{round_id}` | `handle_decide` | `{"decisions": {call_id: "approve_once"|"approve_session"|"always_allow"|"deny"}}` | `{"ok": bool}` |

All responses `Cache-Control: no-store`; unknown session → `404 {"error": "unknown_session"}`; `mobile_surface=false` → all routes 404 (registered conditionally, like the canvas routes).

- [ ] **Step 1: Write failing tests** — `Tests/Web_Server/test_mobile_api.py` using the integration file's `_FakeBaseServer` + `_logged_in_client` idiom (copy the helper, or import it via a small shared conftest if the file allows; otherwise duplicate locally with a comment). Cases: auth required (401 without cookie); CSRF required on send; conversations list shape; send → 202 → (polled) message appears in `?after=` history; unknown session 404; `mobile_surface=false` → 404. The service is injected as a fake (dataclass stub recording calls) — engine tests stay in Task 3.
- [ ] **Step 2: Verify failure** (404s), **Step 3: Implement** (handlers are thin: parse → service call → json_response; service constructed lazily in the mixin behind `self._mobile_service`, gated by the config flag, shut down in `on_shutdown`), **Step 4: Verify pass**, plus regression re-run of `Tests/Web_Server/test_web_auth_integration.py` (baseline 11 failures name-identical).
- [ ] **Step 5: Commit** — `git commit -m "feat(web-server): mobile REST API under /api/v1/mobile (ADR-234)"`

---

### Task 5: Mobile stream WebSocket

**Files:**
- Modify: `tldw_chatbook/Web_Server/mobile_api.py` (add `handle_mobile_stream`)
- Modify: `tldw_chatbook/Web_Server/serve.py` (route `GET /api/v1/mobile/stream`)
- Test: extend `Tests/Web_Server/test_mobile_api.py`

**Interfaces:**
- Consumes: `WebAuthManager.authenticate_request(facts, websocket=True)` — requires `Origin` match, GET upgrade, and BOTH subprotocols: `csrf.<session-csrf>` and `chatbook-v1` (`WEBSOCKET_PROTOCOL`); `MobileChatService.events()`.
- Produces: one WS per browser; server → client JSON frames = `MobileEvent.to_wire()`; client → server frames: `{"type":"ping"}` (reply `{"type":"pong"}`); close codes: `4401` auth expired (client redirects to `/auth/login`, preserving the composer draft in `sessionStorage`), `1011` engine error. Reconnect contract: client reconnects with backoff (1s, 2s, 5s, then 15s cap) and resyncs via `GET …/messages?after=<cursor>`; `visibilitychange`→visible triggers an immediate resync poll.

- [ ] **Step 1: Failing tests** — upgrade without subprotocols → `HTTPUnauthorized`; with valid session + subprotocols → 101; then a send via REST while the WS is open → the test receives `run_started`, ≥1 `delta`, `message_done`, `run_completed` frames in order (fake gateway, plain-chat path suffices here — agent WS behavior is covered in Task 3).
- [ ] **Step 2/3/4:** verify failure → implement (subscribe `svc.events()` per socket, `msg_json` send loop with 30 s idle ping; on `asyncio.CancelledError`/`ConnectionResetError` drain the subscriber) → verify pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(web-server): mobile stream WebSocket with CSRF subprotocol auth (ADR-234)"`

---

### Task 6: `/m` page, PWA, discovery links

**Files:**
- Create: `tldw_chatbook/Web_Server/static/mobile/index.html`, `app.css`, `app.js`, `manifest.webmanifest`, `icon-192.png`, `icon-512.png`, `apple-touch-icon.png`
- Create: `Helper_Scripts/generate_mobile_icons.py` (stdlib-only PNG writer)
- Modify: `tldw_chatbook/Web_Server/serve.py` (`handle_mobile_page` route `/m`; narrow-viewport "Phone view" link in the login page HTML and in `static/served_shell.html` toolbar; extend `_SERVED_SHELL_ASSETS` allowlist for `mobile/*` or add a dedicated `/static/mobile/{name}` route mirroring the existing asset handler)
- Test: `Tests/Web_Server/test_mobile_page.py`

**Interfaces:**
- Consumes: Task 4+5 API; `request["chatbook_csrf"]` for the meta tag (devices-page idiom); API base `/api/v1/mobile`.
- Produces: `/m` HTML (authed) with `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">`, `<meta name="chatbook-csrf" content=…>`, manifest link, `apple-mobile-web-app-capable`, theme-color, safe-area CSS (`env(safe-area-inset-*)`), and these views in `app.js` (vanilla, ~400 lines, no framework):
  - **Session list** — title, relative time, `has_folder_tools` badge; tap → transcript; "+" → new-chat (title prompt); pull-data via `GET /conversations` on load + on `visibilitychange`.
  - **Transcript** — messages rendered as bubbles (`role: user|assistant`, assistant text only — tool-call cards render as a compact "tool: <llm_name>" row from message metadata if present); composer (auto-grow `<textarea>`, Send disabled while a run is active, Stop button → cancel); WS live updates for the open session; **polling sync**: every 12 s and on `visibilitychange`, re-`GET …/messages?after=<cursor>` and append (spec §4.4 — this is the cross-surface completion sync).
  - **Approvals inbox** — badge count from WS `approval_requested` + `GET /approvals`; card per round (llm_name, arguments pretty-printed, options); tapping an option POSTs the decision.
  - **Header** — "Open terminal" link (`/`), sign-out link (`/auth/devices`).
  - All fetches: `credentials: "same-origin"`, CSRF header from meta (devices.js idiom); WS URL from `location`, subprotocols `["chatbook-v1", "csrf." + csrf]`; auth-expiry close (4401) → save draft to `sessionStorage`, redirect `/auth/login`.

Icons: `Helper_Scripts/generate_mobile_icons.py` writes the three PNGs (192/512/180 rounded-square, single accent color, "C" glyph) using only `struct`+`zlib` (raw RGBA scanlines → PNG chunks; no Pillow). Run once; commit the outputs; do not run it in CI.

- [ ] **Step 1: Failing tests** — `/m` requires auth (401 unauthenticated); authed GET → 200 with manifest link + csrf meta; `/static/mobile/manifest.webmanifest` served with `application/manifest+json`; each icon served with `image/png` and correct byte size class; login page and served shell contain the "Phone view" link (`grep`-style assertions on rendered HTML); `mobile_surface=false` → `/m` 404.
- [ ] **Step 2/3/4:** verify failure → implement (page handler mirrors `handle_devices_page`; asset route mirrors `handle_served_shell_asset` with `Cache-Control: public, max-age=3600` for icons/manifest, `no-store` for index.html) → verify pass.
- [ ] **Step 5: Manual live check** — run `tldw-serve --host 127.0.0.1 --port 8765 --pair` in the worktree venv, open `/m` at a phone viewport (browser-use skill, 390×844): session list renders, a plain chat round-trip works end-to-end against a stub gateway (temporary local edit is allowed for the probe and must be reverted — or point the service at the agent double via a `mobile_provider_gateway` test seam documented in the task file). Record findings in the task notes per `backlog/docs/lessons-live-verification.md`.
- [ ] **Step 6: Commit** — `git commit -m "feat(web-server): phone-first /m page with PWA manifest and discovery links (ADR-234)"`

---

### Task 7: Config wiring, README, close-out

**Files:**
- Modify: `tldw_chatbook/Web_Server/README.md`, `tldw_chatbook/Web_Server/serve.py` (if any wiring remains), `backlog/tasks/<TASK-ID> - ....md`
- Test: extend `Tests/Web_Server/test_mobile_api.py` (config parsing)

**Interfaces:**
- Consumes: everything above.
- Produces: `[web_server] mobile_surface` documented; README "Mobile companion surface" section (what /m does, v1 limits — approvals mobile-initiated only, completion-level sync, no service worker — config flag, PWA install note); task file closed (ACs checked, `## Implementation Notes` with approach/files/ADR-234/spike links, status Done via `backlog-py`).

- [ ] **Step 1: Failing test** — `get_cli_setting("web_server", "mobile_surface", default=True)` gate: server built with the flag false registers no `/m` and no `/api/v1/mobile/*` routes (assert via the app's router map in the existing fixture idiom).
- [ ] **Step 2/3:** implement any remaining wiring → green. **Step 4:** full targeted regression pass: `Tests/Web_Server/` new files + `test_web_auth_integration.py` (baseline 11) + `Tests/Canvas/test_web_auth.py` (80) + `test_session_store.py` (5) + `test_serve_cli_pair.py` (3) — no new failures beyond the documented baselines.
- [ ] **Step 5: Close out** — task file Done via `~/.local/bin/backlog-py task edit <TASK-ID> -s Done`; lessons check (add a `lessons-*` entry only if something generalizable surfaced).
- [ ] **Step 6: Commit** — `git commit -m "docs(web-server): mobile surface README + config + task close-out (ADR-234)"`

---

## Self-Review

**Spec coverage:** §4.1 surface/PWA/no-SW/discovery → Task 6; §4.2 REST table + WS auth note → Tasks 4-5; §4.3 Binding 1 + shared requirements (constants done in 1a; lazy DB open — the service constructs on first use behind the flag; concurrent-send limitation documented in ADR-234 §5) → Task 3; §4.4 reconnect/polling/draft-preservation → Tasks 5-6; §4.5 config flag → Tasks 4/7; §5 sequencing respected. Out of scope per spec: passkeys, notes/library reading, desktop relay (Phase 2).

**Placeholder scan:** the two explicitly-deferred lookups are anchored, not vague — `store.create_session`'s exact signature (read at implementation; line given) and the controller's Stop entrypoint (the TUI Stop button's call path; the implementer greps `chat_screen.py` for the stop action). No TBDs elsewhere; icon generator, WS frames, and page structure are fully specified.

**Type consistency:** `MobileEvent.to_wire()` frames (Task 3) are the WS body (Task 5); `svc.sessions`/`session_messages`/`send`/`cancel`/`list_approvals`/`decide_approval` names match across Tasks 3-6; the `csrf.`+`chatbook-v1` subprotocol pair appears identically in Tasks 5 and 6; `AgentModeGateway` (Task 2) is the Task 3 test dependency.
