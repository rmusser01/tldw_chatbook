# Chatbook Server Parity Gap Ledger

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

## Status Update After Tranche 0

- `Cross-cutting Runtime Policy` is no longer an unlanded blocker. The foundational runtime-policy package, capability registry, hard-stop seams, representative UI preflight, and raw-client boundary were landed and verified in [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md).
- The remaining runtime-policy work is breadth and adoption across more domains and screens, not absence of the authority model itself.
- The active parity focus should therefore shift to the next user-priority standalone and remote-interop rows rather than treating runtime policy as still missing.

## Critical Gaps

### Collections: Reading List / Read-it-later: Saved reading collection and read-later item flows
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local saved-state persistence, local saved-search and note-link storage, local bulk status/tag/delete updates, local durable archive snapshots, local extractive summaries, local reading export, server URL-save wrappers, server saved-search CRUD, server note-link wrappers, server bulk item updates, server archive snapshot creation, server reading summary generation, server reading export wrappers, server reading TTS audio wrappers, server save/remove compatibility mapping, the aggregate `All Media` server saved view, and server ingestion-source create for `archive_snapshot` and `git_repository` are already landed.
- Gap: Local TTS parity, per-media-type server saved views, chunk-level TTS playback adoption, and any sync/mirror semantics are still deferred for this user-priority standalone collection surface.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/items.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Screens/media_screen.py`; Verification: local saved-state, saved-search, and note-link persistence is covered by `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, and `Tests/Media/test_media_reading_normalizers.py`; server save/remove compatibility, URL save, saved-search CRUD, note-link wrappers, bulk update, archive snapshot, summary generation, export, TTS audio generation, and the aggregate `All Media` saved view are covered by `Tests/Media/test_media_reading_scope_service.py`, `Tests/Media/test_server_media_reading_service.py`, `Tests/tldw_api/test_media_reading_client.py`, and `Tests/UI/test_media_window_v2_parity.py`; server ingestion-source create is covered by `Tests/Media/test_server_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, and `Tests/UI/test_media_ingestion_source_panel.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This is now a partially landed vertical rather than an open blank-slate gap; the remaining work is the per-media-type saved-view matrix and any future sync contract.

### Watchlists: Watchlists, sources, jobs, runs, and alert rules
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local subscriptions are now mapped through a watchlists scope service, server source CRUD is wrapped through the shared API client, and local/server run lifecycle seams exist for list/detail/launch/observe. Alert-rule CRUD now has local persistence, policy-gated direct server API wrappers, source-normalized records, runtime-policy action gates, and local completed-run alert evaluation that dispatches through Chatbook's durable notification path while honoring local notification settings.
- Gap: Watchlist group editing, dedicated watchlists management UX, and sync/mirror semantics remain missing. The backend seam now covers sources, run lifecycle, alert rules, and local alert notification delivery.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`; Chatbook: `tldw_chatbook/Subscriptions/local_watchlists_service.py`, `tldw_chatbook/Subscriptions/server_watchlists_service.py`, `tldw_chatbook/Subscriptions/watchlist_scope_service.py`, `tldw_chatbook/Notifications/notification_dispatch_service.py`, `tldw_chatbook/tldw_api/client.py`; Verification: source CRUD, run lifecycle, alert-rule CRUD, local alert notification dispatch, and policy registry coverage are covered by `Tests/Subscriptions/test_watchlist_scope_service.py`, `Tests/Subscriptions/test_local_watchlists_service.py`, `Tests/Subscriptions/test_server_watchlists_service.py`, `Tests/Subscriptions/test_notification_dispatch_service.py`, `Tests/tldw_api/test_watchlists_client.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
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
- Current state: The runtime-policy foundation is now in place with one authoritative source-state model, action-level capability registry, hard-stop enforcement for approved seams, representative UI preflight, and raw-client boundary cleanup.
- Remaining gap: Broader rollout is still needed across the rest of the product surface, but this is no longer a missing prerequisite.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/core/feature_flags.py`, `../tldw_server/tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces_rate_limit_policy.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/server.py`; Chatbook: `tldw_chatbook/runtime_policy/`, `tldw_chatbook/app.py`, `tldw_chatbook/tldw_api/client.py`; Verification: [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md) records the passing verification matrix and boundary guard coverage.
- Recommended tranche: Landed in Tranche 0; extend incrementally in Tranche 1+
- Notes: Priority 70. This remains leverage work, but it should no longer be treated as an open critical blocker.

## High-Value Partial Crosswalks

### Chat: Chat conversation sessions, history, and message flow
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local chat remains the primary standalone runtime, and Chatbook now has source-aware local/server conversation metadata seams. Local conversation CRUD, keyword-safe metadata updates, list/detail/tree retrieval, and soft delete are represented through `ChatConversationService`; server list/detail/update/tree plus conversation messages-with-context and citations are wrapped by a policy-gated `ServerChatConversationService`; `ChatConversationScopeService` routes source-specific operations and app startup wires the seam without changing the existing chat UI.
- Gap: Remote first-class conversation create/delete remain unsupported because the current server conversation contract exposes those actions only through chat launch/persist flows and not as direct conversation CRUD. Local RAG-context conversation adjunct persistence is not implemented in the local chat database seam. Dedicated chat UI adoption, remote message mutation parity, streaming/persist handoff alignment, and broader source-separated history UX still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_loop.py`; Chatbook: `tldw_chatbook/Chat/chat_conversation_service.py`, `tldw_chatbook/Chat/server_chat_conversation_service.py`, `tldw_chatbook/Chat/chat_conversation_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Chat_Window_Enhanced_Refactored.py`, `tldw_chatbook/UI/Screens/chat_screen_state.py`; Verification: `Tests/Chat/test_chat_conversation_service.py`, `Tests/Chat/test_server_chat_conversation_service.py`, `Tests/Chat/test_chat_conversation_scope_service.py`, `Tests/tldw_api/test_chat_conversation_client.py`, `Tests/tldw_api/test_chat_conversation_schemas.py`, `Tests/ChaChaNotesDB/test_chat_conversation_parity.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 78. Chat remains a core credibility surface for Chatbook as a serious standalone client.

### Characters / Personas / CCP: Character catalog, persona profiles, chat sessions, and character messages
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: CCP is mature locally, and Chatbook now has source-aware character/persona scope routing plus policy-gated direct server adapters for character listing, persona profiles, greetings, chat presets, character/persona chat-session CRUD/restore, per-chat settings, chat export, author-note info, lorebook diagnostics, and server chat-dictionary/world-book administration. Runtime policy now exposes action-level `character.sessions.*` and `chat.dictionary*` codes beyond launch so these operations do not hide behind a generic CCP handoff permission.
- Gap: Character-message mutation, streaming-persist handoff adoption, local character-session parity through the scope service, local/server dictionary-worldbook scope routing, and the eventual CCP UX adoption path are still incomplete. The local product can already create and manage character chats and local world books through existing local flows, but those older local paths are not yet wrapped behind the same source-aware scope services.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/persona.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`; Chatbook: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`, `tldw_chatbook/Character_Chat/server_character_persona_service.py`, `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/Widgets/CCP_Widgets/`; Verification: `Tests/Character_Chat/test_character_persona_scope_service.py`, `Tests/Character_Chat/test_server_chat_dictionary_service.py`, `Tests/tldw_api/test_character_persona_client.py`, `Tests/tldw_api/test_chat_dictionary_client.py`, `Tests/tldw_api/test_character_persona_schemas.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is an important existing seam to finish rather than a net-new product surface.

### Notes / Workspaces: Notes CRUD, workspace CRUD, and notes graph
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Dual-backend notes and workspace seams already exist, including scope services, server-aware notes surfaces, workspace CRUD/source/artifact helpers, local keyword persistence, and policy-gated server notes-graph wrappers for graph fetch, neighbors, manual link create, and manual link delete.
- Gap: Local/offline graph generation, sync/mirror graph semantics, cross-scope graph moves, and dedicated graph UX remain explicitly deferred. The notes graph scope service is server-backed only today and raises an honest unsupported boundary for local/workspace graph calls.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes_graph.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces.py`; Chatbook: `tldw_chatbook/Notes/Notes_Library.py`, `tldw_chatbook/Notes/notes_scope_service.py`, `tldw_chatbook/Notes/server_notes_workspace_service.py`, `tldw_chatbook/tldw_api/client.py`; Verification: `Tests/Notes/test_notes_scope_service.py`, `Tests/Notes/test_server_notes_workspace_service.py`, `Tests/tldw_api/test_notes_workspace_client.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is daily-use parity work with strong existing scaffolding.

### Media / Reading / Ingestion Sources: Reading lists, ingestion sources, ingestion jobs, reading progress, and media-side item flows
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Media and reading seams span local and server modes, including reading progress, read-it-later state, local saved-search and note-link storage, local bulk status/tag/delete updates, local durable archive snapshots, local extractive summaries, local reading export, local/server reading import job submission/status wrappers, server URL save, server saved-search CRUD, server note links, server reading bulk update/archive/summary/export/TTS wrappers, document-version helpers, policy-gated direct server wrappers, server ingestion-source CRUD/sync/archive wrappers, local and server ingestion-source item reattach, server ingest-job controls, and a local ingestion-source/job queue seam. Local ingestion sources can now be created, listed, read, patched, deleted, synced into queued jobs, used to queue local URL/file ingest jobs, queue local reading import jobs, and reattach conflict-detached notes-backed source items without requiring a server.
- Gap: Actual local ingestion/import execution, source item materialization, local TTS parity, per-media-type server saved views, and UI adoption remain incomplete. Server ingestion-source deletion is not exposed by tldw_server and is surfaced as an explicit unsupported boundary after policy enforcement.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Screens/media_screen.py`, `tldw_chatbook/UI/Screens/media_ingest_screen.py`; Verification: `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, `Tests/Media/test_server_media_reading_service.py`, `Tests/Media/test_server_media_ingest_jobs_service.py`, and `Tests/tldw_api/test_media_reading_client.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 78. This row underpins the higher-ranked reading-list and research work.

### Prompts / Chatbooks: Prompt library, prompt workflows, and chatbook import/export jobs
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Prompt CRUD now has local and policy-gated server adapters, server prompt update/delete client methods, source-normalized records, server-backed prompt version list/restore wrappers, a policy-enforced prompt/chatbook scope service wired into app startup, and source-scoped unsupported-capability reporting. Chatbook import/export now has local and policy-gated server service adapters, and server chatbook payloads accept scope-layer dicts as well as typed request objects.
- Gap: Prompt UI adoption remains pending. Local prompt history/version restore is not implemented, so prompt version controls are currently an explicit server-backed seam. Chatbook archive import/export is available, but list/detail/create/update/delete record-style chatbook CRUD remains explicitly unsupported unless a backend exposes those methods; the scope service enforces policy first, raises an honest unsupported boundary, and exposes that limitation through the unsupported-capability report.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chatbooks.py`; Chatbook: `tldw_chatbook/Prompt_Management/local_prompt_service.py`, `tldw_chatbook/Prompt_Management/server_prompt_service.py`, `tldw_chatbook/Prompt_Management/prompt_chatbook_scope_service.py`, `tldw_chatbook/Chatbooks/local_chatbook_service.py`, `tldw_chatbook/Chatbooks/server_chatbook_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/app.py`; Verification: `Tests/Prompt_Management/test_server_prompt_service.py`, `Tests/Prompt_Management/test_prompt_chatbook_scope_service.py`, `Tests/tldw_api/test_prompt_chatbook_client.py`, `Tests/Chatbooks/test_server_chatbook_service.py`, and `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 67. This remains important, but it does not outrank the standalone-first gaps above.

### Study Core: Flashcards, quizzes, and study guides
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Study seams are already strong across local and server modes, including policy-gated direct server adapters for flashcards/decks/reviews, quizzes/questions/attempts, and study-guide generation.
- Gap: Full mutation parity and workspace-aware remote alignment still need cleanup.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_documents.py`; Chatbook: `tldw_chatbook/Study_Interop/local_study_service.py`, `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/Study_Interop/server_quiz_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/Study_Interop/quiz_scope_service.py`, `tldw_chatbook/UI/Screens/study_screen.py`; Verification: `Tests/Study_Interop/test_server_study_service.py`, `Tests/Study_Interop/test_server_quiz_service.py`, `Tests/Study_Interop/test_study_scope_service.py`, and `Tests/Study_Interop/test_quiz_scope_service.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 63. Strong existing seams keep this below the most urgent standalone gaps.

### Collections: Outputs / Templates / Artifacts: Output artifacts, output templates, and render/export jobs
- Requirement class: Remote parity required, local parity optional
- Client obligation: Full CRUD
- Current state: Chatbook now has a dedicated policy-gated `ServerOutputsService` over output-template CRUD, template preview/render launch, output artifact list/detail/create/update/delete, deleted-artifact listing, and purge helpers. An app-wired `OutputsScopeService` now provides the source-aware boundary, normalizes output template/artifact record IDs, fails honestly for local mode until optional local outputs parity is intentionally built, and reports unsupported capabilities for the missing local managed-output backend plus server render-job list/detail/observe.
- Gap: Dedicated outputs UX/adoption and any optional local output/template parity are still pending. Server render currently means synchronous template preview or artifact creation; there is no first-class server render-job list/detail/observe contract to wire.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/outputs.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/outputs_templates.py`; Chatbook: `tldw_chatbook/Outputs_Interop/server_outputs_service.py`, `tldw_chatbook/Outputs_Interop/outputs_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/outputs_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/Outputs/test_server_outputs_service.py`, `Tests/Outputs/test_outputs_scope_service.py`, `Tests/tldw_api/test_outputs_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 39. This is worth tracking, but the matrix keeps it below the local-first reading, watchlist, writing, and research gaps.

### Evaluations: Evaluation CRUD, RAG evals, embedding A/B tests, and run history
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook already has dual-backend evaluation services, a source-aware scope service, normalized evaluation/dataset/run records, policy-gated dataset/evaluation CRUD, local dataset soft-delete, and run list/detail/launch/observe/update.
- Gap: Evaluation UX adoption, deeper server result-artifact normalization, and endpoint-specific mutation edges beyond the unified dataset/evaluation/run seams still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`; Chatbook: `tldw_chatbook/Evaluations_Interop/local_evaluations_service.py`, `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py`, `tldw_chatbook/Evaluations_Interop/evaluation_scope_service.py`, `tldw_chatbook/DB/Evals_DB.py`, `tldw_chatbook/UI/Screens/evals_screen.py`; Verification: `Tests/Evaluations_Interop/test_local_evaluations_service.py`, `Tests/Evaluations_Interop/test_server_evaluations_service.py`, `Tests/Evaluations_Interop/test_evaluation_scope_service.py`, `Tests/Evals/test_evals_db.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 58. This is meaningful parity work, but not ahead of the user-priority standalone surfaces.

### RAG / Embeddings / Chunking Admin: Chunking templates, chunking controls, embeddings admin, and reprocess helpers
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local and server admin seams are already present for chunking templates, template apply/diagnostics, embedding collection list/detail/delete, source-normalized records, policy-gated direct server RAG/admin adapters, and local/server media reprocess launch through the RAG-admin scope.
- Gap: UX adoption and any optional local/server edge parity beyond chunking templates, collection admin, and reprocess launch still need cleanup.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chunking_templates.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/reprocess.py`; Chatbook: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`, `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`, `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py`, `tldw_chatbook/Widgets/chunking_template_editor.py`; Verification: `Tests/RAG_Admin/test_local_rag_admin_service.py`, `Tests/RAG_Admin/test_server_rag_admin_service.py`, `Tests/RAG_Admin/test_rag_admin_scope_service.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 62. This is good leverage work, but it still trails the top standalone-first rows.

## Remote-Only Client Obligations

### Study Packs: Study-pack generation jobs and pack materialization
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has policy-gated server study-pack job launch, job-status observe, pack retrieval, and regenerate helpers through `ServerStudyService` and source-normalized `StudyScopeService` records.
- Gap: Dedicated study-pack UX/adoption, broader discovery affordances, and explicit offline-unavailable presentation remain pending. Local study-pack generation remains unimplemented by design unless a local generation plan is approved.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/core/StudyPacks/jobs.py`, `../tldw_server/tldw_Server_API/app/core/StudyPacks/generation_service.py`, `../tldw_server/tldw_Server_API/app/api/v1/schemas/study_packs.py`; Chatbook: `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/flashcards_schemas.py`; Verification: `Tests/Study_Interop/test_server_study_service.py`, `Tests/Study_Interop/test_study_scope_service.py`, and `Tests/tldw_api/test_flashcards_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 38. This remains a remote-first, later client obligation with fallback to local study-core flows rather than a high-value partial crosswalk.

### Study Suggestions: Study-suggestion anchors, snapshots, actions, and refresh jobs
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has policy-gated server study-suggestion status, snapshot retrieval, refresh-job launch, and action-trigger helpers through `ServerStudyService` and source-normalized `StudyScopeService` records.
- Gap: Dedicated study-suggestion UX/adoption, anchor discovery presentation, and explicit offline-unavailable presentation remain pending. Local suggestion generation remains unimplemented by design unless a local generation plan is approved.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`, `../tldw_server/tldw_Server_API/app/core/StudySuggestions/jobs.py`; Chatbook: `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/study_suggestions_schemas.py`; Verification: `Tests/Study_Interop/test_server_study_service.py`, `Tests/Study_Interop/test_study_scope_service.py`, and `Tests/tldw_api/test_study_suggestions_client.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 38. This remains a remote-first, later client obligation with fallback to local study-core flows rather than a high-value partial crosswalk.

### Server Reminders / Notification Feeds: Server tasks, reminder CRUD, and notification-backed feed views
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook has server notification feed, unread-count, mark-read, dismiss, snooze/cancel-snooze, preferences, SSE observe, and reminder task CRUD wrappers through the shared API client and `ServerNotificationsService`. Feed mutations now use an explicit `notifications.feed.update.server` runtime-policy action instead of being hidden behind feed list authority.
- Gap: Dedicated remote reminder/feed UX and explicit offline-unavailable presentation are still missing. These remain server-owned surfaces and should not be merged into local notification authority.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py`; Chatbook: `tldw_chatbook/Notifications/server_notifications_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/notifications_reminders_schemas.py`, `tldw_chatbook/runtime_policy/registry.py`; Verification: `Tests/Notifications/test_server_notifications_service.py`, `Tests/tldw_api/test_notifications_reminders_client.py`, `Tests/tldw_api/test_notifications_reminders_schemas.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 45. This is now a remote-client backend/adoption gap rather than an absent client wrapper.

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
- Current state: Chatbook now has a dedicated policy-gated `ServerSharingService` over link create/list/revoke/inspect/verify/import, workspace permission sharing/list/update/revoke, shared-with-me discovery, shared workspace/media retrieval, clone, and shared-workspace chat, plus a `SharingScopeService` source boundary that normalizes server records and rejects local mode as explicitly unavailable.
- Gap: Dedicated sharing UX/adoption remains pending. Sharing remains server-owned and remote-only, but the service seam now exposes the offline-unavailable state for future UI binding.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/sharing.py`; Chatbook: `tldw_chatbook/Sharing_Interop/server_sharing_service.py`, `tldw_chatbook/Sharing_Interop/sharing_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/sharing_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/Sharing/test_server_sharing_service.py`, `Tests/Sharing/test_sharing_scope_service.py`, `Tests/tldw_api/test_sharing_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This remains a server-owned convenience surface rather than a local-parity target.

### Web Clipper: Browser clip save, status, and enrichment capture
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has a dedicated policy-gated `ServerWebClipperService` over clip save, clip status, and enrichment persistence, plus a `WebClipperScopeService` source boundary that normalizes clip/enrichment records and rejects local mode as explicitly unavailable.
- Gap: Dedicated web-clipper UX/adoption and browser-extension handoff remain pending. Web clipper remains server-owned and remote-only, but the service seam now exposes the offline-unavailable state for future UI binding.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/web_clipper.py`; Chatbook: `tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py`, `tldw_chatbook/Web_Clipper_Interop/web_clipper_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/web_clipper_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/WebClipper/test_server_web_clipper_service.py`, `Tests/WebClipper/test_web_clipper_scope_service.py`, `Tests/tldw_api/test_web_clipper_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This should remain below the standalone-first reading and writing rows.

## Contract-Maturity Holds

### Research Search / Provider Surfaces: Legacy search entry points and third-party provider adapters
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has policy-gated local and server research-search services plus a `ResearchSearchScopeService` for source-scoped provider catalogs and launch routing. Local mode supports local websearch providers through the existing WebSearch_APIs runner. Server mode supports supported-websearch discovery plus server websearch, arXiv, and Semantic Scholar launch helpers.
- Gap: Dedicated research-provider UX/adoption, local paper-search parity, local provider configuration CRUD, and broader contract confidence remain pending. The server contract is usable for client wrappers, but still fragmented enough to keep this below the local-first research-session row.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research.py`, `../tldw_server/tldw_Server_API/app/core/Research/providers/web.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Arxiv.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Semantic_Scholar.py`; Chatbook: `tldw_chatbook/Research_Interop/local_research_search_service.py`, `tldw_chatbook/Research_Interop/server_research_search_service.py`, `tldw_chatbook/Research_Interop/research_search_scope_service.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/tldw_api/research_search_schemas.py`, `tldw_chatbook/app.py`; Verification: `Tests/Research/test_local_research_search_service.py`, `Tests/Research/test_research_search_scope_service.py`, `Tests/Research/test_server_research_search_service.py`, `Tests/tldw_api/test_research_search_client.py`, and app wiring assertions in `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 31. Keep provider configuration and paper-search local parity on hold until the client-facing provider contract is clearer.

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
