# Research Workspace Grounded Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`; apply `superpowers:test-driven-development`
> before production changes and `superpowers:verification-before-completion`
> before commits. Do not delegate unless the user explicitly requests
> subagents. Apply `impeccable` immediately before UI implementation tasks.

**Goal:** Add persistent workspace Q&A with explicit general/grounded/auto
retrieval, citations, and processing disclosure, without importing Console's
agent/tool/approval runtime.

**Architecture:** A headless `ResearchChatService` captures a qualified
workspace, selected source/version snapshot, retrieval/generation settings,
and controller revision for each turn. Local mode reuses Chat persistence,
workspace `RagScope`, existing RAG pipeline functions, citation ownership, and
the provider gateway. Server mode calls the real server RAG and
`/api/v1/chat/completions` contracts with a server-owned workspace-scoped
conversation and `save_to_db=true`. Exact Local remote-egress consent is a
payload-free extension of the private overlay.

**Tech Stack:** Python 3.11+, Textual 8.x, existing Chat/RAG/citation services,
TLDW API client, Pydantic request subset, private overlay, pytest.

**Spec:**
`Docs/superpowers/specs/2026-08-23-research-workspace-design.md`

**Backlog:** `TASK-21509` (depends on `TASK-21507`, `TASK-21508`)

## Global constraints

- Research Chat has no ToolCatalog, MCP/ACP, approvals, autonomous loops,
  project instructions, or agent runtime imports.
- `general` never retrieves; `rag` requires selected ready sources; `auto`
  reports `Retrieved` or `Did not retrieve` for every response.
- Local retrieval is restricted by the captured `RagScope`. Server source
  bodies never pass through a client-selected provider.
- Local remote-egress consent is exact to workspace + provider + endpoint
  class + redaction policy + source-body mode. A route change invalidates it.
- Each answer retains source IDs/versions, citations, retrieval configuration,
  processing route, provider/model, and terminal generation state.
- Persist drafts per qualified workspace before switching; never retarget an
  in-flight turn.
- Unknown capability fails closed. No silent Local/Server fallback.
- No full-suite run/claim without explicit user approval.

## ADR check

ADR required: no new ADR

ADR path:
`backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already defines canonical chat owners, server-side processing,
local egress consent, no-tools scope, and immutable async context. Existing
server endpoints are being exposed by the client, not newly invented.

## Task 1: Pin turn, retrieval, provenance, and consent contracts

**Files:**

- Create: `tldw_chatbook/Research_Workspace/chat_models.py`
- Create: `tldw_chatbook/Research_Workspace/egress_consent.py`
- Extend: `tldw_chatbook/Research_Workspace/contracts.py`
- Extend: `tldw_chatbook/Research_Workspace/overlay_store.py`
- Test: `Tests/Research_Workspace/test_chat_models.py`
- Test: `Tests/Research_Workspace/test_egress_consent.py`

1. Add RED tests for invalid mode/settings, Grounded without ready sources,
   Auto retrieval disclosure, unqualified source IDs, context revision
   mismatch, consent fingerprint mismatch, bounded consent eviction, and
   payload/secret rejection.
2. Define frozen request/outcome contracts:

   ```python
   class ResearchChatMode(StrEnum):
       GENERAL = "general"
       RAG = "rag"
       AUTO = "auto"

   @dataclass(frozen=True, slots=True)
   class ResearchChatTurnRequest:
       workspace: QualifiedWorkspaceRef
       conversation_id: str
       user_text: str
       mode: ResearchChatMode
       source_snapshot: tuple[ResearchSourceVersionRef, ...]
       retrieval: ResearchRetrievalSettings
       processing_route: ProcessingRoute
       context_revision: int

   @dataclass(frozen=True, slots=True)
   class ResearchChatTurnOutcome:
       status: Literal["complete", "cancelled", "failed"]
       retrieved: bool
       source_snapshot: tuple[ResearchSourceVersionRef, ...]
       citations: tuple[ResearchCitationRef, ...]
       provider: str
       model: str
       processing_route_fingerprint: str
   ```

3. Bound top K, threshold, history, prompt, citation, and diagnostic fields.
   Use source version refs from the Sources phase rather than raw media IDs in
   controller/UI state.
4. Extend `ResearchWorkspacePort` with forward-typed
   `list_chat_sessions(ref, page)`, `create_chat_session(ref, request)`,
   `send_chat_turn(request, on_chunk, cancel_event)`, and
   `stop_chat_turn(ref, turn_id)` methods. Local and Server adapters implement
   only their own calls; `ResearchChatService` remains the shared behavior
   coordinator behind those methods.
5. Extend overlay schema v3 with bounded, payload-free consent fingerprints,
   one bounded unsent chat draft per qualified workspace, conversation
   presentation IDs, and payload-free append-stage receipts. Migrate v1-v2
   without inventing consent or drafts. Drafts are private device-only input,
   capped at 64 KiB each, and never canonical sent messages; append receipts
   retain at most 128 operations. No sent note/source/message body enters the
   overlay or consent record.
6. Implement `ResearchEgressConsentService.preflight/approve/is_approved`.
   Endpoint URL secrets/query parameters are reduced to an endpoint class and
   stable non-secret fingerprint before persistence.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_chat_models.py Tests/Research_Workspace/test_egress_consent.py Tests/Research_Workspace/test_overlay_store.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: define Research chat and egress contracts"
   ```

## Task 2: Implement canonical Local workspace chat

**Files:**

- Create: `tldw_chatbook/Research_Workspace/local_chat.py`
- Create: `tldw_chatbook/Research_Workspace/chat_service.py`
- Extend: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Reuse without copying logic: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`
- Test: `Tests/Research_Workspace/test_local_chat.py`
- Test: `Tests/Research_Workspace/test_chat_service.py`

1. Add RED tests for creating one workspace-scoped conversation, listing and
   selecting sessions, persisted user/assistant turns, FTS/vector/hybrid
   restriction, citations, cancellation, provider failure, egress block, and
   source-version provenance.
2. Create Local conversation through
   `ChatPersistenceService.create_conversation(scope_type="workspace",
   workspace_id=...)`. Let that service own its conversation membership; do
   not add a Research transcript table.
3. Build an `EffectiveScope` from the captured source snapshot and call the
   existing plain/semantic/hybrid pipeline functions by injected callable.
   `rag` refuses an empty effective scope. `auto` retrieves exactly when its
   deterministic readiness policy chooses to and stamps that decision.
4. Before provider dispatch, call the egress service when the processing route
   may leave the device. With approval, pass only retrieved excerpts or full
   bodies according to the approved source-body mode and redaction policy.
5. Resolve/stream through `ConsoleProviderGateway.resolve_for_send` and
   `stream_chat`, but pass no tools/tool schemas. Persist user and assistant
   messages via `ChatPersistenceService.create_message`; seal citations using
   the existing citation repository/owner coordinator and persist usage via the
   existing message-usage seam.
6. Cancellation retains partial text with cancelled state and never marks it
   complete. Provider failure keeps the draft/retry action and sanitized error.
7. Add a source import scan test asserting the new Research chat package has no
   imports from `Agents`, `Tools`, MCP/ACP, Console approval, or tool catalog.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_local_chat.py Tests/Research_Workspace/test_chat_service.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add local grounded Research chat"
   ```

## Task 3: Expose the real Server RAG and completion contracts

**Files:**

- Create: `tldw_chatbook/tldw_api/chat_completion_schemas.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/Chat/server_chat_conversation_service.py`
- Create: `tldw_chatbook/Research_Workspace/server_chat.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Test: `Tests/tldw_api/test_chat_completion_client.py`
- Test: `Tests/Chat/test_server_chat_conversation_service.py`
- Test: `Tests/Research_Workspace/test_server_chat.py`

1. Add RED client tests against the audited server contracts:

   - `POST /api/v1/rag/search` (through the existing `rag` namespace helper),
   - `POST /api/v1/chat/completions`, including SSE streaming with
     `save_to_db=false`,
   - non-streaming `save_to_db=true` + exact `conversation_id`,
   - scoped, idempotent `POST /api/v1/chats/{chat_id}/messages`,
   - `POST /api/v1/chat/messages/{message_id}/rag-context`, and
   - no `tools` / `tool_choice="none"`.

2. Add a bounded Pydantic subset for Chatbook's request fields: messages,
   model, stream, generation options, `save_to_db`, `conversation_id`,
   `api_provider`, and payload-free `research_context`; also add a bounded RAG
   context persistence subset for source refs, excerpts, citations, retrieval
   settings, answer, and generation state. Forbid extra fields so tools cannot
   leak into Research through an arbitrary dict.
3. Add `create_chat_completion`, `stream_chat_completion`, and
   `persist_chat_message_rag_context` to `TLDWAPIClient`, following existing
   timeout/SSE/error normalization. Extend existing `create_character_message`
   with the endpoint's `Idempotency-Key` argument. Do not implement a second
   HTTP client.
4. Create/select a server workspace conversation through
   `ChatConversationScopeService` with `scope_type="workspace"`, the captured
   `workspace_id`, and model assistant identity. Add thin scoped
   `create_message` and `persist_rag_context` wrappers to
   `ServerChatConversationService`; always pass scope fields on every
   list/get/create/update/delete/message call.
5. Server `general` streams completion without RAG. `rag`/retrieving `auto`
   calls Server RAG with the effective selected IDs in `include_media_ids` and
   then streams server completion with the returned bounded excerpts.
   Streaming always sends
   `save_to_db=false` because the audited server does not persist streamed
   content. Never call Local RAG or the Local provider gateway.
6. After stream completion or user cancellation, idempotently append the user
   and non-empty assistant/partial-assistant messages through the scoped
   message endpoint, preserving parent IDs. Then persist the bounded RAG
   context on the assistant message. Same-process retry resumes the first
   missing receipt and never duplicates a message. After restart, a receipt
   stopped between user and assistant append exposes `Regenerate interrupted
   answer` against the already-persisted user message; it never fabricates or
   locally stores the lost assistant body. Provider failure before canonical
   append retains the user input as the per-workspace draft.
7. The non-streaming fallback may use `save_to_db=true` with the exact
   conversation ID and must reconcile returned message IDs before showing a
   persistence receipt. Load messages/citations through existing context and
   citation APIs. If canonical persistence or citation context is absent,
   report the exact feature unavailable rather than keeping a browser/local
   transcript.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/tldw_api/test_chat_completion_client.py Tests/Chat/test_server_chat_conversation_service.py Tests/Research_Workspace/test_server_chat.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add server-owned Research chat"
   ```

## Task 4: Build the Grounded Chat pane and message actions

**Files:**

- Modify: `tldw_chatbook/UI/Research_Workspace_Modules/chat_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/chat_message.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/citation_detail.py`
- Modify: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Modify: `tldw_chatbook/css/features/_research_workspace.tcss`
- Test: `Tests/UI/test_research_chat_region.py`
- Test: `Tests/UI/test_research_chat_messages.py`
- Test: `Tests/UI/test_research_workspace_geometry.py`

1. Add mounted RED tests for conversation select/new/clear, mode selection,
   source chips, readiness blocker, route disclosure, retrieval options,
   transcript, composer, stream/Stop, retry, citations, and exact Auto outcome.
2. Mount controls once and patch visibility/disabled/reason states in place.
   The default view shows mode, selected sources, model, transcript, composer,
   and current route; advanced retrieval diagnostics are progressively shown.
3. Wire message copy/edit/regenerate/delete/undo/branch to canonical services.
   Save-to-Quick-Notes opens the existing Quick Notes editor with message and
   source provenance. Read aloud appears only when TTS capability is real.
4. Preserve draft, transcript scroll anchor, option state, and semantic focus
   across pane reflow. An active non-durable stream blocks workspace/authority
   switching with Stop recovery.
5. On a stale controller revision, retain owner-persisted completion but do not
   append/repaint it in the new visible workspace.
6. Rebuild CSS; prove composer, Stop/retry, route, blocked reason, and citations
   are painted/reachable at 160x40, 120x30, 100x30, 84x24, 80x24, 60x20.
7. Run:

   ```bash
   .venv/bin/python tldw_chatbook/css/build_css.py
   .venv/bin/python -m pytest -q Tests/UI/test_research_chat_region.py Tests/UI/test_research_chat_messages.py Tests/UI/test_research_workspace_geometry.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: add Research Grounded Chat pane"
   ```

## Task 5: Integration, live contract, and closeout

**Files:**

- Create: `Tests/integration/test_research_chat_round_trip.py`
- Modify: `Docs/User_Guide/research_workspace.md`
- Modify: `backlog/tasks/task-21509 - Add-persistent-grounded-Research-Workspace-Chat.md`

1. With temporary Local DBs, prove create/restart/reload, selected-source-only
   retrieval, citation persistence, Quick Note capture, stop, and no tool
   execution.
2. Against a configured real server (isolated `TLDW_CONFIG_PATH`), prove a
   workspace-scoped chat ID, selected-media RAG request, completion persistence,
   history reload, and citations. If unavailable, record the exact missing
   live evidence and do not mark the server round-trip AC complete.
3. Run all named plan tests, Ruff on changed Python files, CSS build/parity,
   import-boundary scan, and `git diff --check`. State full pytest was not run.
4. Update guide/task notes only from fresh evidence.
5. Commit:

   ```bash
   git commit -m "docs: complete Research Grounded Chat"
   ```

## Required inverse checks

1. Send a Grounded turn with no ready selected source; readiness test must fail.
2. Use Local provider gateway for Server data; no-cross-call test must fail.
3. Change provider/endpoint while reusing consent; fingerprint test must fail.
4. Omit `save_to_db` or workspace scope in Server mode; restart/reload test must
   fail.
5. Return Auto answer without Retrieved/Did not retrieve; outcome test fails.
6. Import ToolCatalog/Agents or pass tools; import/payload contract tests fail.
7. Let stale completion repaint after workspace switch; fencing test fails.

## Focused verification boundary

Run only the named Research_Workspace, Chat, tldw_api, UI, and one integration
test file; CSS build/parity if changed, Ruff on the changed Python inventory,
and `git diff --check`. No full-suite claim.
