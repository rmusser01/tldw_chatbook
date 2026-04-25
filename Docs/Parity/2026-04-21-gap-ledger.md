# Chatbook Server Parity Gap Ledger

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

## Status Update After Tranche 0

- `Cross-cutting Runtime Policy` is no longer an unlanded blocker. The foundational runtime-policy package, capability registry, hard-stop seams, representative UI preflight, raw-client boundary, and shared unsupported-capability report validator were landed and verified in [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md).
- The remaining runtime-policy work is breadth and adoption across more domains and screens, not absence of the authority model itself.
- The active parity focus should therefore shift to the next user-priority standalone and remote-interop rows rather than treating runtime policy as still missing.

## Critical Gaps

### Collections: Reading List / Read-it-later: Saved reading collection and read-later item flows
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local saved-state persistence, direct local URL reading-item creation backed by local media storage and URL extraction, local reading-highlight CRUD/export, local document annotation CRUD/sync, deterministic local document intelligence for outlines/figures/references/summary insights, local saved-search and note-link storage, local bulk status/tag/delete updates, local durable archive snapshots, local extractive summaries, local reading export, local reading TTS generation through Chatbook's TTS stack, server URL-save wrappers, server saved-search CRUD, server note-link wrappers, server bulk item updates, server archive snapshot creation, server reading summary generation, server reading export wrappers, server reading TTS audio wrappers, server save/remove compatibility mapping, the aggregate `All Media` server saved view, an authoritative scope-service capability helper for aggregate-only server saved browsing, invalid saved-context normalization/cleanup, server ingestion-source create for `archive_snapshot` and `git_repository`, and source-scoped unsupported-capability reporting are already landed.
- Gap: Per-media-type server saved views, chunk-level TTS playback adoption, and any sync/mirror semantics are still deferred for this user-priority standalone collection surface. These local/server gaps are now reported by `MediaReadingScopeService.list_unsupported_capabilities()`.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/items.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Screens/media_screen.py`; Verification: local direct URL reading-item creation, reading-highlight CRUD/export, document annotation CRUD/sync, deterministic local document intelligence, saved-state, saved-search, and note-link persistence is covered by `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, and `Tests/Media/test_media_reading_normalizers.py`; server save/remove compatibility, URL save, saved-search CRUD, note-link wrappers, bulk update, archive snapshot, summary generation, export, TTS audio generation, aggregate `All Media` saved-view behavior, saved-context capability/normalization, and known unsupported reporting are covered by `Tests/Media/test_media_reading_scope_service.py`, `Tests/Media/test_server_media_reading_service.py`, `Tests/tldw_api/test_media_reading_client.py`, and `Tests/UI/test_media_window_v2_parity.py`; server ingestion-source create is covered by `Tests/Media/test_server_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, and `Tests/UI/test_media_ingestion_source_panel.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This is now a partially landed vertical rather than an open blank-slate gap; the remaining work is the per-media-type saved-view matrix and any future sync contract.

### Watchlists: Watchlists, sources, jobs, runs, and alert rules
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local subscriptions are now mapped through a watchlists scope service, server source CRUD is wrapped through the shared API client for RSS, site, and forum sources, and local/server run lifecycle seams exist for list/detail/launch/observe. Local runs can execute through the local RSS/Atom/JSON/podcast and URL monitors, persist subscription items and run stats, and dispatch local alert notifications from completed run results. Alert-rule CRUD now has local persistence, policy-gated direct server API wrappers, source-normalized records, runtime-policy action gates, centralized group-edit rejection, and source-scoped unsupported-capability reporting.
- Gap: Watchlist group editing, dedicated watchlists management UX, and sync/mirror semantics remain missing. The backend seam now covers sources, executable local run lifecycle, server run lifecycle, alert rules, local alert notification delivery, and machine-readable reports for the deferred group-editing boundary.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`; Chatbook: `tldw_chatbook/Subscriptions/local_watchlists_service.py`, `tldw_chatbook/Subscriptions/server_watchlists_service.py`, `tldw_chatbook/Subscriptions/watchlist_scope_service.py`, `tldw_chatbook/Notifications/notification_dispatch_service.py`, `tldw_chatbook/tldw_api/client.py`; Verification: source CRUD, run lifecycle, local run execution, alert-rule CRUD, local alert notification dispatch, unsupported reporting, group-edit rejection, and policy registry coverage are covered by `Tests/Subscriptions/test_watchlist_scope_service.py`, `Tests/Subscriptions/test_local_watchlists_service.py`, `Tests/Subscriptions/test_server_watchlists_service.py`, `Tests/Subscriptions/test_notification_dispatch_service.py`, `Tests/tldw_api/test_watchlists_client.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This is the strongest local-name crosswalk in the matrix and directly matches the user's standalone monitoring priority.

### Writing Suite: Writing projects and manuscript hierarchy
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook now has app-wired local and policy-gated server writing-suite service seams for projects, manuscripts-as-server-parts, chapters, scenes, structure retrieval through the source scope service, source normalization, local direct manuscript-level scenes, local manual version snapshots/restores, local trash listing/restores, reorder/move helpers, server Markdown preservation through the Chatbook Markdown wrapper, and machine-readable unsupported-capability reporting. Server `unassigned_chapters` are retained as project-level bucket records rather than fake manuscripts. Server direct manuscript-level scenes, version-history, and trash-restore calls are represented as explicit unsupported operations because the current server manuscript contract has no verified endpoints for them.
- Gap: Dedicated Writing Suite UX adoption still needs completion. The core backend CRUD, local direct-scene, local manual-version, local trash, reorder/move, and unsupported reporting seams are now present, but the first serious standalone product surface is not complete.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`; Chatbook: `tldw_chatbook/Writing_Interop/local_writing_service.py`, `tldw_chatbook/Writing_Interop/server_writing_service.py`, `tldw_chatbook/Writing_Interop/writing_scope_service.py`, `tldw_chatbook/Writing_Interop/writing_markdown_adapter.py`, `tldw_chatbook/app.py`, `tldw_chatbook/config.py`; Verification: `Tests/Writing_Interop/test_local_writing_service.py`, `Tests/Writing_Interop/test_server_writing_service.py`, `Tests/Writing_Interop/test_writing_scope_service.py`, `Tests/Writing_Interop/test_writing_normalizers.py`, `Tests/tldw_api/test_writing_manuscripts_client.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 78. This remains a user-priority standalone row, but it is now a backend-foundation/adoption gap rather than an absent service model.

### Research Sessions / Runs: Deep research session lifecycle, streaming events, and bundle retrieval
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local research session/run persistence, event history, artifact storage, bundle retrieval, source-normalized session/run/event/artifact/bundle records, server run wrappers, a mode-aware scope service, and source-scoped unsupported-capability reporting are now present and app-wired. The server contract still does not expose research-run deletion, so Chatbook surfaces that as an explicit unsupported server operation after policy enforcement.
- Gap: Dedicated research UX/adoption remains missing, and server-side session semantics are still run-centric rather than a full draft-session CRUD surface. Local CRUD is now available at the service seam; server run launch/list/detail/update/observe/bundle/artifact operations are available through the server wrapper and scope service.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research_runs.py`, `../tldw_server/tldw_Server_API/app/core/Research/service.py`, `../tldw_server/tldw_Server_API/app/core/Research/streaming.py`; Chatbook: `tldw_chatbook/Research_Interop/local_research_service.py`, `tldw_chatbook/Research_Interop/server_research_service.py`, `tldw_chatbook/Research_Interop/research_scope_service.py`, `tldw_chatbook/app.py`; Verification: `Tests/Research/test_local_research_service.py`, `Tests/Research/test_research_scope_service.py`, `Tests/Research/test_server_research_service.py`, and `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 75. This is now a backend-foundation/adoption gap rather than an absent local-first data model. UI/UX work is deferred to the parallel UX effort.

### Local MCP Runtime: Local MCP server runtime, protocol handling, tools, prompts, and status
- Requirement class: Local parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local MCP modules and config exist, but integrated runtime, catalog, approvals, and governance UX are only partial.
- Gap: Chatbook does not yet present a serious local MCP runtime surface even though local operations were explicitly prioritized.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/server.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/protocol.py`; Chatbook: `tldw_chatbook/config.py`; Verification: unified MCP runtime exposes request, batch, status, module, resource, and tool execution paths.
- Recommended tranche: Tranche 2
- Notes: Priority 73. Local MCP should remain Chatbook-owned even if remote catalogs are imported later.

### Client Notifications: Client-local notification state and UI delivery; no direct server analog, with the remote counterpart handled in the server notifications/reminders/feed row
- Requirement class: Local parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has a local notification inbox DB, durable dispatch service, policy-gated local queue/settings service, app wiring, transient toast/notify delivery, centralized delivery-setting enforcement, and watchlist alert-rule producer adoption. Local queue list/update/observe and local settings list/update are represented as runtime-policy actions.
- Gap: Dedicated notification-center UX and delivery preference adoption across the remaining local producers still need completion. Server notification feeds and reminders remain a separate remote-owned surface, not a mirror of local notification state.
- Evidence: Server: adjacent remote counterparts in `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py` and `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`; Chatbook: `tldw_chatbook/Notifications/client_notifications_db.py`, `tldw_chatbook/Notifications/client_notifications_service.py`, `tldw_chatbook/Notifications/notification_dispatch_service.py`, `tldw_chatbook/Widgets/toast_notification.py`, `tldw_chatbook/app.py`; Verification: `Tests/Notifications/test_client_notifications_service.py`, `Tests/Subscriptions/test_client_notifications_db.py`, `Tests/Subscriptions/test_notification_dispatch_service.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 71. This is now a backend-foundation/adoption gap rather than an absent local contract.

## Foundational Work Landed

### Cross-cutting Runtime Policy: Auth, feature flags, rate limits, persona policy hooks, and MCP policy context
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: The runtime-policy foundation is now in place with one authoritative source-state model, action-level capability registry, hard-stop enforcement for approved seams, representative UI preflight, raw-client boundary cleanup, and a shared unsupported-capability report validator/collector that checks source-scoped gap reports against the registry.
- Remaining gap: Broader rollout is still needed across the rest of the product surface, but this is no longer a missing prerequisite.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/core/feature_flags.py`, `../tldw_server/tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces_rate_limit_policy.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/server.py`; Chatbook: `tldw_chatbook/runtime_policy/`, `tldw_chatbook/app.py`, `tldw_chatbook/tldw_api/client.py`; Verification: [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md) records the passing verification matrix and boundary guard coverage; `Tests/RuntimePolicy/test_unsupported_capabilities.py` verifies the shared report contract and current scope-service report outputs.
- Recommended tranche: Landed in Tranche 0; extend incrementally in Tranche 1+
- Notes: Priority 70. This remains leverage work, but it should no longer be treated as an open critical blocker.

## High-Value Partial Crosswalks

### Chat: Chat conversation sessions, history, and message flow
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local chat remains the primary standalone runtime, and Chatbook now has source-aware local/server conversation metadata seams. Local conversation CRUD, keyword-safe metadata updates, list/detail/tree retrieval, and soft delete are represented through `ChatConversationService`; server list/detail/update/tree plus conversation messages-with-context and citations are wrapped by a policy-gated `ServerChatConversationService`; `ChatConversationScopeService` routes source-specific operations, reports known unsupported local/server boundaries, and app startup wires the seam without changing the existing chat UI.
- Gap: Remote first-class conversation create/delete remain unsupported because the current server conversation contract exposes those actions only through chat launch/persist flows and not as direct conversation CRUD. Local RAG-context conversation adjunct persistence is not implemented in the local chat database seam. Dedicated chat UI adoption, remote message mutation parity, streaming/persist handoff alignment, and broader source-separated history UX still need finishing work, but these contract gaps are now machine-readable from the scope seam.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_loop.py`; Chatbook: `tldw_chatbook/Chat/chat_conversation_service.py`, `tldw_chatbook/Chat/server_chat_conversation_service.py`, `tldw_chatbook/Chat/chat_conversation_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Chat_Window_Enhanced_Refactored.py`, `tldw_chatbook/UI/Screens/chat_screen_state.py`; Verification: `Tests/Chat/test_chat_conversation_service.py`, `Tests/Chat/test_server_chat_conversation_service.py`, `Tests/Chat/test_chat_conversation_scope_service.py`, `Tests/tldw_api/test_chat_conversation_client.py`, `Tests/tldw_api/test_chat_conversation_schemas.py`, `Tests/ChaChaNotesDB/test_chat_conversation_parity.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 78. Chat remains a core credibility surface for Chatbook as a serious standalone client.

### Characters / Personas / CCP: Character catalog, persona profiles, chat sessions, and character messages
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: CCP is mature locally, and Chatbook now has source-aware character/persona scope routing plus local character-card search/detail/create/update/delete/restore, a local character/persona session metadata adapter for local chat-session create/list/detail/update/delete/restore/export, and source-aware local/server chat-dictionary routing for dictionary CRUD, entries, import/export, and process operations. Server adapters cover character search/detail/create/update/delete/restore, character exemplar search/detail/create/update/delete/selection-debug, persona profiles, persona exemplar CRUD/import/review, greetings, chat presets, character/persona chat-session CRUD/restore, server character-message list/detail/create/update/delete/search, per-chat settings, chat export, author-note info, lorebook diagnostics, server chat-dictionary/world-book administration, and machine-readable unsupported reporting for known local execution-surface gaps. Runtime policy now exposes action-level `character.sessions.*`, `character.messages.*`, and local/server `chat.dictionary*` codes beyond launch so these operations do not hide behind a generic CCP handoff permission.
- Gap: Streaming-persist handoff adoption, local persona profile/exemplar CRUD, local character exemplar CRUD, local greetings/presets/settings/lorebook diagnostics through the source-aware scope, local chat-dictionary activity/version history, and the eventual CCP UX adoption path are still incomplete. Local character/persona session metadata CRUD and dictionary CRUD/entry/import/export/process routing are now wrapped; remaining older local CCP execution paths are reported explicitly.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/persona.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`; Chatbook: `tldw_chatbook/Character_Chat/local_character_persona_service.py`, `tldw_chatbook/Character_Chat/character_persona_scope_service.py`, `tldw_chatbook/Character_Chat/local_chat_dictionary_service.py`, `tldw_chatbook/Character_Chat/chat_dictionary_scope_service.py`, `tldw_chatbook/Character_Chat/server_character_persona_service.py`, `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/Widgets/CCP_Widgets/`; Verification: `Tests/Character_Chat/test_local_character_persona_service.py`, `Tests/Character_Chat/test_character_persona_scope_service.py`, `Tests/Character_Chat/test_local_chat_dictionary_service.py`, `Tests/Character_Chat/test_chat_dictionary_scope_service.py`, `Tests/Character_Chat/test_server_chat_dictionary_service.py`, `Tests/tldw_api/test_character_persona_client.py`, `Tests/tldw_api/test_chat_dictionary_client.py`, `Tests/tldw_api/test_character_persona_schemas.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is an important existing seam to finish rather than a net-new product surface.

### Notes / Workspaces: Notes CRUD, workspace CRUD, and notes graph
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Dual-backend notes and workspace seams already exist, including scope services, server-aware notes surfaces, workspace CRUD/source/artifact helpers, local keyword persistence, policy-gated server notes-graph wrappers for graph fetch, neighbors, manual link create, and manual link delete, plus machine-readable unsupported reporting for local/workspace graph boundaries.
- Gap: Local/offline graph generation, sync/mirror graph semantics, cross-scope graph moves, and dedicated graph UX remain explicitly deferred. The notes graph scope service is server-backed only today and raises plus reports an honest unsupported boundary for local/workspace graph calls.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes_graph.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces.py`; Chatbook: `tldw_chatbook/Notes/Notes_Library.py`, `tldw_chatbook/Notes/notes_scope_service.py`, `tldw_chatbook/Notes/server_notes_workspace_service.py`, `tldw_chatbook/tldw_api/client.py`; Verification: `Tests/Notes/test_notes_scope_service.py`, `Tests/Notes/test_server_notes_workspace_service.py`, `Tests/tldw_api/test_notes_workspace_client.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is daily-use parity work with strong existing scaffolding.

### Media / Reading / Ingestion Sources: Reading lists, ingestion sources, ingestion jobs, reading progress, and media-side item flows
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Media and reading seams span local and server modes, including reading progress, read-it-later state, direct local URL reading-item creation, local reading-highlight CRUD/export, local document annotation CRUD/sync, deterministic local document intelligence, local saved-search and note-link storage, local bulk status/tag/delete updates, local durable archive snapshots, local extractive summaries, local reading export, local reading import execution/materialization for JSONL/JSON/CSV-style imports, local/server reading import job submission/status wrappers, server URL save, server saved-search CRUD, server note links, server reading bulk update/archive/summary/export/TTS wrappers, document-version helpers, server file-artifact create/detail/reference/delete/export/purge client wrappers, server OCR/VLM backend discovery and OCR POINTS preload wrappers, policy-gated direct server wrappers, server ingestion-source CRUD/sync/archive wrappers, local and server ingestion-source item reattach, server ingest-job controls, local ingestion-source/job execution for local-directory source item materialization, archive snapshot upload/sync materialization, git repository source materialization, URL article/file ingest, and local file ingest jobs, and machine-readable unsupported-capability reporting. Local ingestion sources can now be created, listed, read, patched, deleted, synced into materialized source-item records, used to materialize local directories, archive snapshots, and git repositories, execute local URL article and file-download ingest jobs, execute local file ingest jobs, execute local reading import jobs, and reattach conflict-detached notes-backed source items without requiring a server.
- Gap: Per-media-type server saved views and UI adoption remain incomplete. Server ingestion-source deletion is not exposed by tldw_server and is surfaced as an explicit unsupported boundary after policy enforcement and through the media scope unsupported report.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/ocr.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/vlm.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Screens/media_screen.py`, `tldw_chatbook/UI/Screens/media_ingest_screen.py`; Verification: `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, `Tests/Media/test_server_media_reading_service.py`, `Tests/Media/test_server_media_ingest_jobs_service.py`, `Tests/tldw_api/test_media_reading_client.py`, and `Tests/tldw_api/test_ocr_vlm_client.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 78. This row underpins the higher-ranked reading-list and research work.

### External Connectors: OAuth/token providers, linked accounts, remote source browsing, source import/sync, and job status
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has typed direct API-client wrappers for server connector provider discovery, OAuth authorization URL creation, OAuth callback completion, account list/delete, provider source browsing, source create/list/patch, source import trigger, source sync status, source sync trigger, and connector job status.
- Gap: Dedicated UX adoption, explicit offline-unavailable presentation, org connector policy admin, inbound webhook handling, credential lifecycle design, active-server switching cache invalidation, and local/server connector-source mirror semantics remain pending or out of scope. Local ingestion sources remain the offline/local mechanism and should not be conflated with server OAuth connector accounts.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/connectors.py`, `../tldw_server/tldw_Server_API/app/api/v1/schemas/connectors.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/connectors_schemas.py`; Verification: `Tests/tldw_api/test_connectors_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 35. This is client-facing server completeness for external sources, but it should remain below the local-first media/reading ingestion work and skip admin policy surfaces in this phase.

### Audio / Speech Services: TTS/STT health, speech generation, audio jobs, TTS history, and transcription/translation
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has typed direct API-client wrappers for server TTS health, STT health, provider/voice catalog discovery, binary speech generation, long-form speech jobs and artifacts, audio job submission/status/SSE progress, TTS history list/detail/favorite/delete, and file-based transcription/translation. Existing local media/TTS behavior remains separate and local-owned.
- Gap: Dedicated audio UX adoption, source-aware audio scope routing, websocket streaming, tokenizer endpoints, stored voice upload/cloning/encoding, audiobook creation/project/artifact APIs, and admin audio-job controls remain pending. Local/server audio history identity and artifact sync semantics are not defined.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/audio/audio_health.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/audio/audio_history.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/audio_schemas.py`; Verification: `Tests/tldw_api/test_audio_client.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 58. This is a cross-cutting media/chat utility surface, but it should stay source-separated until local audio artifact and history semantics are explicitly designed.

### LLM Provider / Model Catalog: LLM health, configured providers, model metadata, and available model IDs
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Observe
- Current state: Chatbook now has typed direct API-client wrappers for server LLM inference health, configured-provider list, provider detail, flattened model metadata with repeatable type/input/output modality filters, and the flat available-model ID catalog. Local provider configuration remains a separate Chatbook-owned concern.
- Gap: Dedicated UX adoption, source-aware provider catalog service routing, provider configuration mutation, server switching cache invalidation, MLX/llama.cpp admin process controls, and provider-setting sync/mirror semantics remain pending. Admin/provider-control endpoints are intentionally not wrapped in this discovery slice.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/llm_providers.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/llm_provider_schemas.py`; Verification: `Tests/tldw_api/test_llm_provider_client.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 57. This row is cross-cutting because chat, writing, audio, slides, research, and generation UX need an honest active-server model catalog without mutating local provider settings.

### Server Runtime / Config Discovery: Health, readiness, safe config, tokenizer/jobs config, and provider key status
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Observe
- Current state: Chatbook now has typed direct API-client wrappers for server health, liveness, readiness, metrics, security posture, non-sensitive docs-info capability flags, flashcards import limits, tokenizer config get/update, jobs config, provider key status, and explicit provider-key validation. This stays scoped to active-server discovery and safe client-facing config.
- Gap: Dedicated UX adoption, server switching cache invalidation, and admin config mutation remain pending. Chatbook local runtime settings remain separate and are not overwritten by server discovery.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/health.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/config_info.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/server_runtime_schemas.py`; Verification: `Tests/tldw_api/test_server_runtime_client.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 54. This row supports server switching and connected-mode capability gating without requiring the UI/UX layer to infer server capabilities from unrelated domain calls.

### Auth / Profile / Sessions: Login, refresh, logout, self sessions, registration, user profile, account security, BYOK, API keys, and storage
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Observe
- Current state: Chatbook now has typed direct API-client wrappers for OAuth2-compatible login, token refresh, logout with local bearer-token clearing, auth-scoped session list/revoke/revoke-all, registration, profile catalog fetch, self profile fetch/update, password reset/change, email verification resend/verify, magic-link request/verify, MFA setup/verify/disable/login, per-user API-key list/create/virtual-create/rotate/revoke, BYOK provider-key upsert/list/test/delete, OpenAI OAuth authorize/callback/status/refresh/disconnect/source-switch, self storage quota/recalculate, and non-admin generated-file storage/folder/usage/trash operations. The client now updates its bearer token on login/refresh/magic-link/MFA token responses without overloading the existing sharing token schema.
- Gap: Durable local credential storage, token auto-refresh policy, server switching cache invalidation, UX adoption, and admin/ops account surfaces remain pending or intentionally out of scope.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/auth.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/users.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/user_keys.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/storage.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/auth_user_schemas.py`, `tldw_chatbook/tldw_api/account_security_schemas.py`, `tldw_chatbook/tldw_api/user_keys_schemas.py`, `tldw_chatbook/tldw_api/storage_schemas.py`; Verification: `Tests/tldw_api/test_auth_user_client.py`, `Tests/tldw_api/test_account_storage_user_keys_client.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 56. This row is required for Chatbook-as-standalone-client access to multi-user servers, but server identity must stay remote-owned and separate from Chatbook's local single-user identity.

### User Governance / Consent: Consent preferences and self privilege-map introspection
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Observe
- Current state: Chatbook now has typed direct API-client wrappers for authenticated-user consent preference read/grant/withdraw and self privilege-map discovery. A user-detail privilege-map helper is also available for the server route that permits self or authorized administrative inspection, but org/team maps and snapshot administration remain outside this slice.
- Gap: Dedicated UX adoption, offline-unavailable presentation, org/team privilege maps, privilege snapshots/exports, resource-governor policy admin, and broader consent/governance sync semantics remain pending or intentionally out of scope. Chatbook should treat this as active-server policy state rather than local identity state.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/consent.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/privileges.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/user_governance_schemas.py`; Verification: `Tests/tldw_api/test_user_governance_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 31. This is needed for Chatbook-as-client transparency against a multi-user server, but it should not outrank local-first content and task surfaces.

### Server Skills: Server-managed SKILL.md skills, context payloads, and execution preview
- Requirement class: Remote parity required, local parity optional/deferred
- Client obligation: Full CRUD + Execute
- Current state: Chatbook now has typed direct API-client wrappers for server skill list, context payload retrieval, detail, create, update with `If-Match`, delete with `If-Match`, text import, multipart file import, binary zip export, execution preview, and built-in seed operations.
- Gap: Dedicated UX adoption, policy-aware invocation placement, local/offline unavailable presentation, and any local/server skill mirror or sync semantics remain pending. Chatbook should not auto-merge server SKILL.md records into local instruction assets until identity and safety rules are designed.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/skills.py`, `../tldw_server/tldw_Server_API/app/api/v1/schemas/skills_schemas.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/skills_schemas.py`; Verification: `Tests/tldw_api/test_skills_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 33. This is useful for server-client completeness and future agent tooling, but remains below the local-first product surfaces and does not require MCP SDK usage.

### Prompts / Chatbooks: Prompt library, prompt workflows, and chatbook import/export jobs
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Prompt CRUD now has local and policy-gated server adapters, server prompt update/delete client methods, source-normalized records, local prompt version list/restore from retained sync-log snapshots, server-backed prompt version list/restore wrappers, a policy-enforced prompt/chatbook scope service wired into app startup, and source-scoped unsupported-capability reporting. Chatbook import/export now has local and policy-gated server service adapters, local persistent chatbook record list/detail/create/update/delete, server continuation-export, export/import job list/detail/cancel/remove wrappers, completed-export binary download routing, and server chatbook payloads accept scope-layer dicts as well as typed request objects.
- Gap: Prompt UI adoption remains pending. Local chatbook archives and persistent record CRUD are available through the source-aware backend seam, while server list/detail/create/update/delete record-style chatbook CRUD remains explicitly unsupported unless the connected backend exposes those methods; the scope service enforces policy first, raises an honest unsupported boundary, and exposes that limitation through the unsupported-capability report.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chatbooks.py`; Chatbook: `tldw_chatbook/Prompt_Management/local_prompt_service.py`, `tldw_chatbook/Prompt_Management/server_prompt_service.py`, `tldw_chatbook/Prompt_Management/prompt_chatbook_scope_service.py`, `tldw_chatbook/Chatbooks/local_chatbook_service.py`, `tldw_chatbook/Chatbooks/server_chatbook_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/app.py`; Verification: `Tests/Prompt_Management/test_local_prompt_service.py`, `Tests/Prompt_Management/test_server_prompt_service.py`, `Tests/Prompt_Management/test_prompt_chatbook_scope_service.py`, `Tests/tldw_api/test_prompt_chatbook_client.py`, `Tests/Chatbooks/test_local_chatbook_service.py`, `Tests/Chatbooks/test_server_chatbook_service.py`, and `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 67. This remains important, but it does not outrank the standalone-first gaps above.

### Prompt Studio: Server prompt experimentation projects, prompts, test cases, evaluations, optimizations, and status
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Full CRUD + Observe
- Current state: Chatbook now has typed direct API-client wrappers for Prompt Studio projects, prompts, preview/convert/execute helpers, test cases including bulk/import/export/generate/run, evaluations, optimizations including simple/canonical create, cancel, strategies, history, iterations, strategy comparison, and queue status. The schemas intentionally tolerate the server's current mixed canonical and compatibility response shapes while preserving route-level typing for callers.
- Gap: Dedicated UX adoption, source-aware service routing, SSE/websocket realtime updates, CSV multipart upload import, background ping diagnostics, local Prompt Studio project mirrors, and sync/mirror semantics remain pending. Existing local prompt/evaluation data should stay separate until a deliberate identity bridge is designed.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_projects.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_test_cases.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_evaluations.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/prompt_studio_schemas.py`; Verification: `Tests/tldw_api/test_prompt_studio_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 43. This is a connected-server product surface rather than a standalone-first blocker, but the backend client contract is now present for future UX work.

### Study Core: Flashcards, quizzes, and study guides
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Study seams are already strong across local and server modes, including policy-gated direct server adapters for flashcards/decks/reviews, quizzes/questions/attempts, study-guide generation, and source-scoped unsupported reporting for the local workspace boundary.
- Gap: Full mutation parity and workspace-aware remote alignment still need cleanup. Local mode remains global-only for study decks/reviews and quizzes/attempts; workspace-scoped study is currently a server-mode boundary reported by `StudyScopeService` and `QuizScopeService`.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_documents.py`; Chatbook: `tldw_chatbook/Study_Interop/local_study_service.py`, `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/Study_Interop/server_quiz_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/Study_Interop/quiz_scope_service.py`, `tldw_chatbook/UI/Screens/study_screen.py`; Verification: `Tests/Study_Interop/test_server_study_service.py`, `Tests/Study_Interop/test_server_quiz_service.py`, `Tests/Study_Interop/test_study_scope_service.py`, and `Tests/Study_Interop/test_quiz_scope_service.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 63. Strong existing seams keep this below the most urgent standalone gaps.

### Collections: Outputs / Templates / Artifacts: Output artifacts, output templates, data tables, slides, and render/export jobs
- Requirement class: Remote parity required, local parity optional
- Client obligation: Full CRUD
- Current state: Chatbook now has a dedicated policy-gated `ServerOutputsService` over output-template CRUD, template preview/render launch, output artifact list/detail/create/update/delete, deleted-artifact listing, and purge helpers. An app-wired `OutputsScopeService` now provides the source-aware boundary, normalizes output template/artifact record IDs, fails honestly for local mode until optional local outputs parity is intentionally built, and reports unsupported capabilities for the missing local managed-output backend. Direct server data-table wrappers now cover generate, list/detail, export, content update, metadata patch, delete, regenerate, job status, and job cancel. Direct server slides wrappers now cover presentation CRUD/search, templates, visual styles, versions, prompt/chat/notes/media/RAG generation, render job submit/status/artifacts, binary export, and health.
- Gap: Dedicated outputs/data-table/slides UX adoption and any optional local output/template/data-table/presentation parity are still pending. Output-template render remains synchronous preview or artifact creation; slides have first-class render jobs, but they are only exposed through the backend API client for now.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/outputs.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/outputs_templates.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/data_tables.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/slides.py`; Chatbook: `tldw_chatbook/Outputs_Interop/server_outputs_service.py`, `tldw_chatbook/Outputs_Interop/outputs_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/outputs_schemas.py`, `tldw_chatbook/tldw_api/data_tables_schemas.py`, `tldw_chatbook/tldw_api/slides_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/Outputs/test_server_outputs_service.py`, `Tests/Outputs/test_outputs_scope_service.py`, `Tests/tldw_api/test_outputs_client.py`, `Tests/tldw_api/test_data_tables_client.py`, `Tests/tldw_api/test_slides_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 39. This is worth tracking, but the matrix keeps it below the local-first reading, watchlist, writing, and research gaps.

### Kanban Boards / Tasks: Boards, lists, cards, labels, comments, checklists, links, search, and workflow controls
- Requirement class: Remote parity required, local parity optional/deferred
- Client obligation: Full CRUD + Observe
- Current state: Chatbook now has typed direct API-client wrappers for the core Kanban board/list/card REST subset, including board/list/card create/list/detail/update/delete, archive/unarchive, restore, list/card reorder, card move/copy, nested board/card detail response shapes, and optimistic-lock update headers.
- Gap: Labels, comments, checklists, content links, card search/filter, activity feeds, import/export, bulk operations, workflow policy/state controls, source-aware service routing, dedicated UX adoption, and any optional local Kanban model remain pending. Workflow-specific routes are intentionally not implemented in this slice because broader workflows are currently deferred.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_boards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_lists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_cards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_labels.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_comments.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_checklists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_links.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_search.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/kanban/kanban_workflow.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/kanban_schemas.py`; Verification: `Tests/tldw_api/test_kanban_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This is a server-owned task-board surface for now; Chatbook should report local/offline unavailability until a separate local task-board model is intentionally approved.

### Evaluations: Evaluation CRUD, RAG evals, embedding A/B tests, and run history
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook already has dual-backend evaluation services, a source-aware scope service, normalized evaluation/dataset/run records, policy-gated dataset/evaluation CRUD, local dataset soft-delete, run list/detail/launch/observe/update, local persisted per-run dataset override/webhook URL launch config, server RAG-pipeline preset admin/cleanup wrappers, server embeddings A/B test admin wrappers, and source-scoped unsupported reporting for known local/server eval contract gaps.
- Gap: Evaluation UX adoption, deeper server result-artifact normalization, server target-catalog discovery, and local webhook callback delivery still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`; Chatbook: `tldw_chatbook/Evaluations_Interop/local_evaluations_service.py`, `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py`, `tldw_chatbook/Evaluations_Interop/evaluation_scope_service.py`, `tldw_chatbook/DB/Evals_DB.py`, `tldw_chatbook/UI/Screens/evals_screen.py`; Verification: `Tests/Evaluations_Interop/test_local_evaluations_service.py`, `Tests/Evaluations_Interop/test_server_evaluations_service.py`, `Tests/Evaluations_Interop/test_evaluation_scope_service.py`, `Tests/Evals/test_evals_db.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 58. This is meaningful parity work, but not ahead of the user-priority standalone surfaces.

### RAG / Embeddings / Chunking Admin: Chunking templates, chunking controls, embeddings admin, and reprocess helpers
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local and server admin seams are already present for chunking templates, local template apply/tag handling, template diagnostics, embedding collection create/list/detail/delete, local collection export, source-normalized records, policy-gated direct server RAG/admin adapters, local/server media reprocess launch through the RAG-admin scope, and source-scoped unsupported reporting for known server RAG-admin contract gaps.
- Gap: UX adoption and a direct server embedding collection export contract still need cleanup beyond chunking templates, collection admin, and reprocess launch.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chunking_templates.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/reprocess.py`; Chatbook: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`, `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`, `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py`, `tldw_chatbook/Widgets/chunking_template_editor.py`; Verification: `Tests/RAG_Admin/test_local_rag_admin_service.py`, `Tests/RAG_Admin/test_server_rag_admin_service.py`, `Tests/RAG_Admin/test_rag_admin_scope_service.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 62. This is good leverage work, but it still trails the top standalone-first rows.

## Remote-Only Client Obligations

### Study Packs: Study-pack generation jobs and pack materialization
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has policy-gated server study-pack job launch, job-status observe, pack retrieval, regenerate helpers, source-normalized `StudyScopeService` records, and unsupported-capability reporting for local/offline mode plus the missing server job-list contract.
- Gap: Dedicated study-pack UX/adoption, broader discovery affordances, and explicit offline-unavailable presentation remain pending. Local study-pack generation remains unimplemented by design unless a local generation plan is approved; server discovery remains job-status/detail oriented rather than list oriented.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/core/StudyPacks/jobs.py`, `../tldw_server/tldw_Server_API/app/core/StudyPacks/generation_service.py`, `../tldw_server/tldw_Server_API/app/api/v1/schemas/study_packs.py`; Chatbook: `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/flashcards_schemas.py`; Verification: `Tests/Study_Interop/test_server_study_service.py`, `Tests/Study_Interop/test_study_scope_service.py`, and `Tests/tldw_api/test_flashcards_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 38. This remains a remote-first, later client obligation with fallback to local study-core flows rather than a high-value partial crosswalk.

### Study Suggestions: Study-suggestion anchors, snapshots, actions, and refresh jobs
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has policy-gated server study-suggestion status, snapshot retrieval, refresh-job launch, action-trigger helpers, source-normalized `StudyScopeService` records, and unsupported-capability reporting for local/offline mode.
- Gap: Dedicated study-suggestion UX/adoption, anchor discovery presentation, and explicit offline-unavailable presentation remain pending. Local suggestion generation remains unimplemented by design unless a local generation plan is approved.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`, `../tldw_server/tldw_Server_API/app/core/StudySuggestions/jobs.py`; Chatbook: `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/study_suggestions_schemas.py`; Verification: `Tests/Study_Interop/test_server_study_service.py`, `Tests/Study_Interop/test_study_scope_service.py`, and `Tests/tldw_api/test_study_suggestions_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 38. This remains a remote-first, later client obligation with fallback to local study-core flows rather than a high-value partial crosswalk.

### Server Reminders / Notification Feeds: Server tasks, reminder CRUD, and notification-backed feed views
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook has server notification feed, unread-count, mark-read, dismiss, snooze/cancel-snooze, preferences, SSE observe, and reminder task CRUD wrappers through the shared API client and `ServerNotificationsService`. Feed mutations now use an explicit `notifications.feed.update.server` runtime-policy action, and `NotificationsScopeService` provides an app-wired remote-only seam with normalized server notification/reminder record IDs plus local/offline unavailable reporting.
- Gap: Dedicated remote reminder/feed UX and explicit offline-unavailable presentation are still missing. These remain server-owned surfaces and should not be merged into local notification authority.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py`; Chatbook: `tldw_chatbook/Notifications/server_notifications_service.py`, `tldw_chatbook/Notifications/notifications_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/notifications_reminders_schemas.py`, `tldw_chatbook/runtime_policy/registry.py`; Verification: `Tests/Notifications/test_server_notifications_service.py`, `Tests/Notifications/test_notifications_scope_service.py`, `Tests/tldw_api/test_notifications_reminders_client.py`, `Tests/tldw_api/test_notifications_reminders_schemas.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 45. This is now a remote-client backend/adoption gap rather than an absent client wrapper.

### Meetings: Meeting sessions, templates, artifacts, finalization, sharing, and event stream
- Requirement class: Remote-only acceptable
- Client obligation: Full CRUD + Observe
- Current state: Chatbook now has typed direct API-client wrappers for meeting health, session create/list/detail/status transition, template create/list/detail, artifact create/list, session finalization, Slack/webhook share dispatch, and session event SSE streaming.
- Gap: Dedicated meetings UX/adoption, local meeting-store parity, websocket live transcript ingestion, and any offline meeting workflow remain pending. The current Chatbook surface is a server client contract only.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/meetings.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/meetings_schemas.py`; Verification: `Tests/tldw_api/test_meetings_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 34. This is useful connected-client coverage but stays below local-first reading, writing, chat, research, and notification work.

### Workflows: General workflow definitions and run lifecycle
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: No workflow discovery, launch, or run-status surface exists in Chatbook.
- Gap: Chatbook cannot yet discover or observe remote workflow runs when connected.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workflows.py`; Chatbook: no dedicated workflow client in `tldw_chatbook/tldw_api/client.py` or the local UI; Verification: workflow definition, launch, and runtime routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 33. Remote-only acceptable rows should not outrank the standalone-first rows above.

### Scheduler Workflows: Scheduled orchestration and scheduling control plane
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook does not expose a scheduler-workflow control plane today, only adjacent local scheduling analogs.
- Gap: Remote scheduled orchestration remains undiscoverable and unmanageable from the client.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`; Chatbook: adjacent local scheduling analogs in `tldw_chatbook/UI/Screens/subscription_screen.py` and `tldw_chatbook/DB/Subscriptions_DB.py`; Verification: a separate scheduler workflow module is present server-side.
- Recommended tranche: Tranche 3
- Notes: Priority 30. This stays below the standalone-first rows because scheduler workflows are explicitly remote-only acceptable.

### Chat Workflows: Chat-specific orchestration and launch helpers
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook exposes ordinary chat, but not chat-workflow discovery, launch, or status.
- Gap: The remote chat-workflow contract is present server-side, but Chatbook has no dedicated client surface for it.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`; Chatbook: only general chat conversation methods in `tldw_chatbook/tldw_api/client.py`; Verification: scoped Chatbook sources expose general chat contracts, not a dedicated chat-workflow client.
- Recommended tranche: Tranche 3
- Notes: Priority 33. This is a real scored gap, but it remains appropriately below core local-first chat parity.

### Sharing: Share links, permissions, revocation, and share discovery
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has a dedicated policy-gated `ServerSharingService` over link create/list/revoke/inspect/verify/import, workspace permission sharing/list/update/revoke, shared-with-me discovery, shared workspace/media retrieval, clone, and shared-workspace chat, plus a `SharingScopeService` source boundary that normalizes server records and reports local/offline unavailability.
- Gap: Dedicated sharing UX/adoption remains pending. Sharing remains server-owned and remote-only; server share-link observation is still explicitly unsupported because the current server contract has no share-event stream.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/sharing.py`; Chatbook: `tldw_chatbook/Sharing_Interop/server_sharing_service.py`, `tldw_chatbook/Sharing_Interop/sharing_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/sharing_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/Sharing/test_server_sharing_service.py`, `Tests/Sharing/test_sharing_scope_service.py`, `Tests/tldw_api/test_sharing_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This remains a server-owned convenience surface rather than a local-parity target.

### Web Clipper: Browser clip save, status, and enrichment capture
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has a dedicated policy-gated `ServerWebClipperService` over clip save, clip status, and enrichment persistence, plus a `WebClipperScopeService` source boundary that normalizes clip/enrichment records and reports local/offline unavailability.
- Gap: Dedicated web-clipper UX/adoption and browser-extension handoff remain pending. Web clipper remains server-owned and remote-only; server clip listing and event observation are still explicitly unsupported because the current contract only exposes capture, enrichment persistence, and status by clip ID.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/web_clipper.py`; Chatbook: `tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py`, `tldw_chatbook/Web_Clipper_Interop/web_clipper_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/web_clipper_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/WebClipper/test_server_web_clipper_service.py`, `Tests/WebClipper/test_web_clipper_scope_service.py`, `Tests/tldw_api/test_web_clipper_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This should remain below the standalone-first reading and writing rows.

### Translation Utility: Remote text translation helper
- Requirement class: Remote-only acceptable
- Client obligation: Trigger
- Current state: Chatbook now has typed `TranslateRequest` / `TranslateResponse` schemas and a direct shared API-client wrapper for server text translation.
- Gap: Dedicated UX/adoption remains pending. There is no local translation backend target yet, so local/offline translation should stay out of parity claims until a Chatbook-owned local model path is intentionally designed.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/translate.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/translation_schemas.py`; Verification: `Tests/tldw_api/test_translation_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 24. This is a small remote utility seam, not a standalone-client product surface.

## Contract-Maturity Holds

### Research Search / Provider Surfaces: Legacy search entry points and third-party provider adapters
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has policy-gated local and server research-search services plus a `ResearchSearchScopeService` for source-scoped provider catalogs, launch routing, and unsupported-capability reporting. Local mode supports local websearch providers through the existing WebSearch_APIs runner plus direct arXiv and Semantic Scholar paper-search runners. Server mode supports supported-websearch discovery plus server websearch, arXiv, and Semantic Scholar launch helpers.
- Gap: Dedicated research-provider UX/adoption, provider configuration CRUD, provider observe/status events, and broader contract confidence remain pending. The server contract is usable for client wrappers, but still fragmented enough to keep this below the local-first research-session row.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research.py`, `../tldw_server/tldw_Server_API/app/core/Research/providers/web.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Arxiv.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Semantic_Scholar.py`; Chatbook: `tldw_chatbook/Research_Interop/local_research_search_service.py`, `tldw_chatbook/Research_Interop/server_research_search_service.py`, `tldw_chatbook/Research_Interop/research_search_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/research_search_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/Research/test_local_research_search_service.py`, `Tests/Research/test_research_search_scope_service.py`, `Tests/Research/test_server_research_search_service.py`, `Tests/tldw_api/test_research_search_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 31. Keep provider configuration and provider observe/status events on hold until the client-facing provider contract is clearer.

### Remote MCP Control Plane / Governance: MCP hub governance, catalog management, policy, approvals, and external-server control
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Current Chatbook MCP work is local-first and there is no dedicated remote governance client.
- Gap: The remote governance scope is explicit server-side but still admin-heavy and low-confidence as a client target.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_catalogs_manage.py`; Chatbook: `tldw_chatbook/config.py` with no dedicated governance surface; Verification: governance and catalog management routes exist, but the matrix confidence is low.
- Recommended tranche: Tranche 3
- Notes: Priority 27. The user explicitly deprioritized remote MCP governance relative to local MCP runtime.

## Deferred / Explicitly Out Of Scope

- No additional Task 5 deferrals were added beyond the lower-priority remote-only and contract-maturity rows already captured above.
