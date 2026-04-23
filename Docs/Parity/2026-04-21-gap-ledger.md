# Chatbook Server Parity Gap Ledger

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

## Status Update After Tranche 0

- `Cross-cutting Runtime Policy` is no longer an unlanded blocker. The foundational runtime-policy package, capability registry, hard-stop seams, representative UI preflight, and raw-client boundary were landed and verified in [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md).
- The remaining runtime-policy work is breadth and adoption across more domains and screens, not absence of the authority model itself.
- The active parity focus should therefore shift to the next user-priority standalone and remote-interop rows rather than treating runtime policy as still missing.
- `Watchlists` plus `Client Notifications` are now partially landed through the subscriptions-shell and first server control-plane vertical. The verification record lives in [watchlists-notifications-tranche-2.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Docs/Development/watchlists-notifications-tranche-2.md).

## Critical Gaps

### Collections: Reading List / Read-it-later: Saved reading collection and read-later item flows
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local saved-state persistence, server save/remove compatibility mapping, the aggregate `All Media` server saved view, authoritative capability enforcement, runtime normalization, search-panel status affordance, and server ingestion-source create for `archive_snapshot` and `git_repository` are now landed.
- Gap: The remaining gap is no longer a Chatbook-only follow-on. Per-media-type server saved views are blocked on a server list-contract extension, and any sync/mirror semantics remain deferred for later design.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/items.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/UI/MediaWindow_v2.py`, `tldw_chatbook/Widgets/Media/media_search_panel.py`; Verification: local saved-state persistence is covered by `Tests/Media/test_local_media_reading_service.py` and `Tests/Media/test_media_reading_normalizers.py`; authoritative capability behavior is covered by `Tests/Media/test_media_reading_scope_service.py`; aggregate `All Media` saved-view behavior, runtime normalization, and search-panel affordance are covered by `Tests/UI/test_media_window_v2_parity.py`; server ingestion-source create is covered by `Tests/Media/test_server_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, and `Tests/UI/test_media_ingestion_source_panel.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This vertical is landed for the current contract; any future follow-on depends on a server-side per-media-type saved-view extension and any separately approved sync contract.

### Watchlists: Watchlists, sources, jobs, runs, and alert rules
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: A source-aware subscriptions shell, local notifications inbox, server watchlist source CRUD/restore, server jobs, server runs, server alert-rule administration, and first-slice server reminders/feed controls are now landed. Local mode remains backed by subscriptions and local notifications, while server-only control-plane tabs show explicit local/offline guidance instead of creating fake local jobs, runs, reminders, or feed state.
- Gap: Watchlist groups remain read-only/deferred. Richer structured job and alert-rule editors, richer run outputs/logs/artifact or audio summaries, richer server reminder/feed UX, and any sync or mirror semantics remain future work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py`; Chatbook: `tldw_chatbook/Notifications/client_notifications_db.py`, `tldw_chatbook/Notifications/notification_dispatch_service.py`, `tldw_chatbook/Notifications/server_notifications_service.py`, `tldw_chatbook/Notifications/server_notifications_scope_service.py`, `tldw_chatbook/Subscriptions/local_watchlists_service.py`, `tldw_chatbook/Subscriptions/server_watchlists_service.py`, `tldw_chatbook/Subscriptions/watchlist_scope_service.py`, `tldw_chatbook/UI/SubscriptionWindow.py`; Verification: `Tests/tldw_api/test_watchlists_schemas.py`, `Tests/tldw_api/test_watchlists_client.py`, `Tests/tldw_api/test_server_notifications_client.py`, `Tests/Notifications/test_server_notifications_service.py`, `Tests/Subscriptions/test_client_notifications_db.py`, `Tests/Subscriptions/test_notification_dispatch_service.py`, `Tests/Subscriptions/test_server_watchlists_service.py`, `Tests/Subscriptions/test_watchlist_scope_service.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/UI/test_subscription_window_watchlists.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 81. Source CRUD, first server execution/control-plane slice, and first server reminders/feed slice are now landed; the remaining work is UX depth, group management, and sync design rather than missing basic server jobs/runs/alert-rule/reminder/feed routing.

### Writing Suite: Writing projects and manuscript hierarchy
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook now has partial structural parity for the Writing Suite. Local project/manuscript/chapter/scene CRUD is supported at the service/controller layer; local scene drafts are Markdown; local manuscript/chapter state is metadata and structure only; local versions and trash restore are supported. Server project/part/chapter/scene structural CRUD is supported through explicit server-mode service/controller seams, server parts are mapped as Chatbook manuscripts, server unassigned chapters are rendered in the explicit `Unassigned Chapters` bucket, and source-specific search/reorder/move seams are in place. The mounted Writing UI currently exposes source switching, project browse, outline/detail selection, project create, selected-entity save/delete, local version create/restore, and unsupported server reason state.
- Gap: Full mounted affordances for child create, search, reorder/move, and trash restore remain pending UX completion. Server direct manuscript-level scenes, server manual versions, server trash restore, and server scene reparenting remain blocked pending verified server endpoints. Generation, export, collaboration, sync/mirroring, and advanced prose IDE features remain future rows rather than structural-authoring parity requirements.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`; Chatbook: `tldw_chatbook/DB/Writing_DB.py`, `tldw_chatbook/Writing_Interop/local_writing_service.py`, `tldw_chatbook/Writing_Interop/server_writing_service.py`, `tldw_chatbook/Writing_Interop/writing_scope_service.py`, `tldw_chatbook/UI/Screens/writing_screen.py`, `tldw_chatbook/UI/Writing_Window.py`, `tldw_chatbook/Widgets/Writing/`; Verification: `Tests/tldw_api/test_writing_manuscripts_client.py`, `Tests/DB/test_writing_db.py`, `Tests/Writing_Interop/`, `Tests/UI/test_writing_screen.py`, and `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 78. Structural authoring is now landed for the current local/server contracts; remaining work is server-contract extension and future product breadth.

### Research Sessions / Runs: Deep research session lifecycle, streaming events, and bundle retrieval
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook now has a dedicated Research Sessions surface. Local mode stores research runs and artifacts in a standalone SQLite DB, exposes create/list/detail/control/bundle operations through a local service, and keeps local research state client-owned. Server mode uses typed research-runs client methods for create/list/detail/pause/resume/cancel, bundle/artifact retrieval, checkpoint patch-and-approve operations, and live SSE event streaming. The app wires these behind a source-aware scope service and a dedicated Research screen with a selected-run `Watch Events` action.
- Gap: Local mode does not run an autonomous research engine yet, and the mounted UI still needs richer artifact, checkpoint, bundle, and event-log controls beyond first-slice run browsing/control plus concise live event observation.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research_runs.py`, `../tldw_server/tldw_Server_API/app/core/Research/service.py`, `../tldw_server/tldw_Server_API/app/core/Research/streaming.py`; Chatbook: `tldw_chatbook/tldw_api/research_runs_schemas.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/DB/Research_DB.py`, `tldw_chatbook/Research_Interop/`, `tldw_chatbook/UI/Research_Window.py`, `tldw_chatbook/UI/Screens/research_screen.py`; Verification: `Tests/tldw_api/test_research_runs_client.py`, `Tests/Research_Interop/test_research_scope_service.py`, `Tests/UI/test_research_screen.py`, and `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 75. The missing-surface gap is closed for first-slice source-separated CRUD/control; the remaining work is execution depth and live observation.

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
- Current state: Chatbook now has a dedicated local notification store, dispatch pipeline, and inbox tab with read and dismiss behavior for watchlists and subscriptions actions.
- Gap: Broader notification producers plus richer local filtering or configuration remain separate from the landed client-local contract. Server reminder and notification-feed state is now surfaced separately and remains server-owned.
- Evidence: Server: adjacent remote counterparts in `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py` and `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`; Chatbook: `tldw_chatbook/Notifications/client_notifications_db.py`, `tldw_chatbook/Notifications/notification_dispatch_service.py`, `tldw_chatbook/UI/SubscriptionWindow.py`, `tldw_chatbook/config.py`; Verification: `Tests/Subscriptions/test_client_notifications_db.py`, `Tests/Subscriptions/test_notification_dispatch_service.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/UI/test_subscription_window_watchlists.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 71. This is now a partially landed supporting vertical; the remaining work is breadth and later server-adjacent surfacing, not the core local inbox contract.

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
- Current state: Local chat is already strong and mode-aware, while remote conversation interoperability remains thinner.
- Gap: Remote conversation contract alignment and explicit source separation still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_loop.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Chat_Window_Enhanced_Refactored.py`, `tldw_chatbook/UI/Screens/chat_screen_state.py`; Verification: dedicated chat routes exist and `chat_workflows.py` is intentionally scored separately.
- Recommended tranche: Tranche 1
- Notes: Priority 78. Chat remains a core credibility surface for Chatbook as a serious standalone client.

### Characters / Personas / CCP: Character catalog, persona profiles, chat sessions, and character messages
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: CCP is mature locally, with explicit server route families already identified.
- Gap: Shared identifiers and remote contract alignment are still incomplete across characters, personas, sessions, and messages.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/persona.py`; Chatbook: `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/Widgets/CCP_Widgets/`; Verification: character, persona, session, message, and memory route families are explicit.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is an important existing seam to finish rather than a net-new product surface.

### Notes / Workspaces: Notes CRUD, workspace CRUD, and notes graph
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Dual-backend notes and workspace seams already exist, including scope services and server-aware notes surfaces.
- Gap: Graph parity and full remote normalization still need completion.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes_graph.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces.py`; Chatbook: `tldw_chatbook/Notes/Notes_Library.py`, `tldw_chatbook/Notes/notes_scope_service.py`, `tldw_chatbook/Notes/server_notes_workspace_service.py`; Verification: notes, graph, and workspace endpoint families are separate and explicit.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is daily-use parity work with strong existing scaffolding.

### Media / Reading / Ingestion Sources: Reading lists, ingestion sources, ingestion jobs, reading progress, and media-side item flows
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Media and reading seams span local and server modes, including reading progress, local/server reading-highlight CRUD seams, first-class TUI highlight create/update/delete controls, ingestion-source management, typed server web-content ingest helpers, typed server ingest-job submit/status/list/cancel controls, typed server ingest-job SSE event parsing, a server job tab for submit plus last-batch refresh/cancel, selected-job cancel, selected-batch live watch, recent visible server-job live watch, known-batch lookup by ID, first-class server web-content ingest controls, and a server-only Web Clipper tab for clip save, known-clip status lookup, and enrichment persistence. The media viewer now loads and renders reading highlights for selected records and routes highlight mutations through the active source.
- Gap: Full contract parity is still incomplete for true historical batch discovery. The current server ingest-job list contract requires a known `batch_id`, and the current web-clipper contract does not expose a browse/history route. Chatbook can watch recent visible ingest jobs through the unscoped SSE snapshot and can inspect a known clip by ID, but durable historical browsing still needs server-side batch-discovery, recent-batches, or clip-history extensions.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading_highlights.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/web_clipper.py`; Chatbook: `tldw_chatbook/tldw_api/media_reading_schemas.py`, `tldw_chatbook/tldw_api/web_clipper_schemas.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/DB/Client_Media_DB_v2.py`, `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/WebClipper/server_web_clipper_service.py`, `tldw_chatbook/WebClipper/server_web_clipper_scope_service.py`, `tldw_chatbook/UI/MediaWindow_v2.py`, `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`, `tldw_chatbook/UI/Screens/media_screen.py`, `tldw_chatbook/UI/Screens/media_ingest_screen.py`; Verification: `Tests/tldw_api/test_media_reading_client.py`, `Tests/tldw_api/test_web_clipper_client.py`, `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_server_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, `Tests/WebClipper/test_server_web_clipper_service.py`, `Tests/UI/test_media_window_v2_parity.py`, and `Tests/UI/test_media_ingest_window_rebuilt.py` cover reading, reading-highlight, ingestion-source, ingest-job, web-clipper, and reading-progress routes.
- Recommended tranche: Tranche 1
- Notes: Priority 78. This row underpins the higher-ranked reading-list and research work.

### Prompts / Chatbooks: Prompt library, prompt workflows, and chatbook import/export jobs
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook now has typed server prompt CRUD client methods for paginated list, detail, create, update, usage, delete, versions, and restore; typed server chatbook job list/status/cancel/remove/download methods; a server chatbook service that preserves the existing plain-dict wizard contract and writes downloaded exports to local storage; live remote server job browsing plus cancel/download/remove controls in the export management window; a source-aware prompt scope service with stable normalized local/server prompt IDs; source-aware server prompt usage/version/restore service routing with explicit local-unavailable behavior for version history; app-level prompt scope wiring; existing CCP prompt list/load/create/update/delete routing through the active local/server runtime source; and mounted CCP controls for prompt usage recording, server version listing, and server version restore.
- Gap: The durable client/service seam, source-routed prompt CRUD path, live remote job browsing/actions, export download affordance, and mounted prompt usage/version controls are now exposed, but the TUI still needs chatbook cleanup/continuation affordances, prompt collections/workflows, and deeper import/export identity alignment.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chatbooks.py`; Chatbook: `tldw_chatbook/tldw_api/prompt_chatbook_schemas.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/Prompt_Management/prompt_scope_service.py`, `tldw_chatbook/Prompt_Management/prompt_normalizers.py`, `tldw_chatbook/Chatbooks/server_chatbook_service.py`, `tldw_chatbook/UI/CCP_Modules/ccp_prompt_handler.py`, `tldw_chatbook/UI/ChatbookExportManagementWindow.py`, `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/UI/Screens/chatbooks_screen.py`, `tldw_chatbook/Widgets/CCP_Widgets/ccp_prompt_editor_widget.py`; Verification: `Tests/tldw_api/test_prompt_chatbook_client.py`, `Tests/tldw_api/test_prompt_chatbook_schemas.py`, `Tests/Prompt_Management/test_server_prompt_adapter.py`, `Tests/Prompt_Management/test_prompt_scope_service.py`, `Tests/Chatbooks/test_server_chatbook_service.py`, `Tests/UI/test_ccp_prompt_handler_scope.py`, `Tests/UI/test_ccp_screen.py`, `Tests/UI/test_chatbooks_screen_server_actions.py`, and `Tests/UI/test_chatbook_management_server_jobs.py`.
- Recommended tranche: Tranche 1
- Notes: Priority 67. This remains important, but it does not outrank the standalone-first gaps above.

### Study Core: Flashcards, quizzes, and study guides
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Study seams are already strong across local and server modes, including quizzes and study-guide generation.
- Gap: Full mutation parity and workspace-aware remote alignment still need cleanup.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_documents.py`; Chatbook: `tldw_chatbook/Study_Interop/local_study_service.py`, `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/UI/Screens/study_screen.py`; Verification: flashcard, quiz, and study-guide contracts are explicit.
- Recommended tranche: Tranche 1
- Notes: Priority 63. Strong existing seams keep this below the most urgent standalone gaps.

### Collections: Outputs / Templates / Artifacts: Output artifacts, output templates, and render/export jobs
- Requirement class: Remote parity required, local parity optional
- Client obligation: Full CRUD
- Current state: Chatbook only has adjacent workspace-artifact surfacing today rather than a dedicated outputs/templates client surface.
- Gap: Dedicated outputs artifact, template, and render/export parity remains partial even though the server contract is explicit.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/outputs.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/outputs_templates.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/Notes/server_notes_workspace_service.py`, `tldw_chatbook/UI/Screens/notes_screen.py`, `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`; Verification: outputs and template modules expose list/detail/create/update/delete/preview routes.
- Recommended tranche: Tranche 3
- Notes: Priority 39. This is worth tracking, but the matrix keeps it below the local-first reading, watchlist, writing, and research gaps.

### Evaluations: Evaluation CRUD, RAG evals, embedding A/B tests, and run history
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook already has dual-backend evaluation services and screen scope support.
- Gap: Contract normalization and full mutation parity still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`; Chatbook: `tldw_chatbook/Evaluations_Interop/local_evaluations_service.py`, `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py`, `tldw_chatbook/UI/Screens/evals_screen.py`; Verification: the evaluations package exposes a unified control plane plus specialized routes.
- Recommended tranche: Tranche 1
- Notes: Priority 58. This is meaningful parity work, but not ahead of the user-priority standalone surfaces.

### RAG / Embeddings / Chunking Admin: Chunking templates, chunking controls, embeddings admin, and reprocess helpers
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local and server admin seams are already present for chunking, embeddings, and reprocess helpers.
- Gap: Full create/delete parity and source labeling still need cleanup.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chunking_templates.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/reprocess.py`; Chatbook: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`, `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`, `tldw_chatbook/Widgets/chunking_template_editor.py`; Verification: stable endpoint-backed embeddings and chunking surfaces exist.
- Recommended tranche: Tranche 1
- Notes: Priority 62. This is good leverage work, but it still trails the top standalone-first rows.

## Remote-Only Client Obligations

### Study Packs: Study-pack generation jobs and pack materialization
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook has no dedicated study-pack discovery, launch, or job-status surface today, and only adjacent local study-core helpers exist.
- Gap: The server study-pack contract is explicit, but Chatbook still lacks the remote discovery, launch, and observe layer for it.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/core/StudyPacks/jobs.py`, `../tldw_server/tldw_Server_API/app/core/StudyPacks/generation_service.py`, `../tldw_server/tldw_Server_API/app/api/v1/schemas/study_packs.py`; Chatbook: adjacent local study helpers in `tldw_chatbook/tldw_api/client.py` and `tldw_chatbook/Study_Interop/`; Verification: stable endpoint-backed study-pack routes exist in `flashcards.py` with job and pack materialization support.
- Recommended tranche: Tranche 3
- Notes: Priority 38. This remains a remote-first, later client obligation with fallback to local study-core flows rather than a high-value partial crosswalk.

### Study Suggestions: Study-suggestion anchors, snapshots, actions, and refresh jobs
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook has no dedicated study-suggestions surface today beyond adjacent next-review helpers.
- Gap: The server suggestion-anchor, snapshot, and action flows are not yet surfaced in Chatbook.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`, `../tldw_server/tldw_Server_API/app/core/StudySuggestions/jobs.py`; Chatbook: adjacent next-review helpers in `tldw_chatbook/tldw_api/client.py` and `tldw_chatbook/Study_Interop/study_scope_service.py`; Verification: dedicated endpoint family plus flashcard/quiz hooks confirm the server study-suggestions contract.
- Recommended tranche: Tranche 3
- Notes: Priority 38. This remains a remote-first, later client obligation with fallback to local study-core flows rather than a high-value partial crosswalk.

### Server Reminders / Notification Feeds: Server tasks, reminder CRUD, and notification-backed feed views
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has typed server reminder and notification-feed client methods, a remote-only server notification service, a policy-aware scope seam, app bootstrap wiring, and lightweight `Server Reminders` plus `Server Feed` tabs in the subscriptions/watchlists surface. Local mode shows explicit unavailable guidance and keeps the existing local Notifications inbox separate.
- Gap: Richer server reminder/feed presentation, filtering, preferences editing, long-running stream worker UX, and later sync or local notification ingestion remain future work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py`; Chatbook: `tldw_chatbook/tldw_api/server_notifications_schemas.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/Notifications/server_notifications_service.py`, `tldw_chatbook/Notifications/server_notifications_scope_service.py`, `tldw_chatbook/UI/SubscriptionWindow.py`; Verification: `Tests/tldw_api/test_server_notifications_client.py`, `Tests/Notifications/test_server_notifications_service.py`, `Tests/UI/test_subscription_window_watchlists.py`, and `Tests/RuntimePolicy/test_runtime_policy_core.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 45. The first remote-only interoperability slice is landed; keep this surface source-explicit and server-owned rather than mirroring it into Chatbook's local notification store by default.

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
- Current state: Chatbook now has typed non-admin Sharing client methods, a remote-only service, a policy-aware scope seam, app bootstrap wiring, and a lightweight `Sharing` panel in `Tools & Settings` for workspace share create/list/update/revoke, shared-with-me discovery, shared workspace clone/chat proxy operations, and share-token create/list/revoke plus public preview/verify/import. Local mode shows explicit unavailable guidance and keeps sharing server-owned.
- Gap: Richer shared-resource presentation, explicit shared workspace browsing beyond JSON status output, deeper share-link/public-access UX, and any future local import or sync semantics remain future work. Admin sharing config and audit stay intentionally excluded by the audit boundary.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/sharing.py`, `../tldw_server/tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`; Chatbook: `tldw_chatbook/tldw_api/sharing_schemas.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/Sharing/server_sharing_service.py`, `tldw_chatbook/Sharing/server_sharing_scope_service.py`, `tldw_chatbook/UI/Sharing_Panel.py`, `tldw_chatbook/UI/Tools_Settings_Window.py`; Verification: `Tests/tldw_api/test_sharing_client.py`, `Tests/Sharing/test_server_sharing_service.py`, `Tests/UI/test_tools_settings_window.py`, and `Tests/UI/test_screen_navigation.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. First remote-only support is landed; keep it server-owned and source-explicit unless a later local import/sync design is separately approved.

### Web Clipper: Browser clip save, status, and enrichment capture
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: Chatbook now has typed Web Clipper client methods, a remote-only service, a policy-aware scope seam, app bootstrap wiring, and a lightweight `Web Clipper` tab in the media ingest window for server clip save, known-clip status lookup, and enrichment persistence. Local mode shows explicit unavailable guidance and keeps Web Clipper server-owned.
- Gap: Browser-extension handoff UX, server clip browse/history, richer capture metadata helpers, and any local mirror/import semantics remain future work. The current server contract supports save, known-ID status, and enrichment persistence, not arbitrary clip discovery.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/web_clipper.py`; Chatbook: `tldw_chatbook/tldw_api/web_clipper_schemas.py`, `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/WebClipper/server_web_clipper_service.py`, `tldw_chatbook/WebClipper/server_web_clipper_scope_service.py`, `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`; Verification: `Tests/tldw_api/test_web_clipper_client.py`, `Tests/WebClipper/test_server_web_clipper_service.py`, and `Tests/UI/test_media_ingest_window_rebuilt.py`.
- Recommended tranche: Tranche 3
- Notes: Priority 29. First remote-only support is landed; keep it below standalone-first rows and source-explicit unless a future local import/sync design is approved.

## Contract-Maturity Holds

### Research Search / Provider Surfaces: Legacy search entry points and third-party provider adapters
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local search exists, but provider behavior is spread across legacy route files and multiple server core integrations.
- Gap: The server research-provider contract is still too low-confidence to drive high-priority client parity work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research.py`, `../tldw_server/tldw_Server_API/app/core/Research/providers/web.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Arxiv.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Semantic_Scholar.py`; Chatbook: `tldw_chatbook/UI/Screens/search_screen.py`, `tldw_chatbook/config.py`; Verification: provider logic is present but split across core files and marked low-confidence in the matrix.
- Recommended tranche: Tranche 3
- Notes: Priority 31. Keep this on hold until the client-facing provider contract is clearer.

### Remote MCP Control Plane / Governance: MCP hub governance, catalog management, policy, approvals, and external-server control
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook now has a dedicated `Unified MCP` destination in `Tools & Settings` with explicit local/server panes, configured server targets, destination-local source and scope restore, remote browse coverage for overview, inventory, catalogs, external servers, governance, and advanced admin, plus scoped remote actions for catalogs, external servers, credentials, governance, trust policy, capability mappings, governance-pack source/import/upgrade flows, path-scope CRUD, workspace-set CRUD and membership, shared-workspace CRUD, policy-assignment workspace membership control, permission-profile and policy-assignment credential-binding administration, slot-status views, and the top-level external-server secret setter.
- Gap: The route-family parity gap on the active `mcp_hub_management` surface is now effectively closed. Remaining work is UI polish, richer structured presentation, and follow-on parity cleanup; the only uncovered paths in the current server endpoint file are deprecated credential-status aliases that do not need first-class client support.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_catalogs_manage.py`; Chatbook: `tldw_chatbook/tldw_api/mcp_unified_client.py`, `tldw_chatbook/MCP/server_unified_service.py`, `tldw_chatbook/MCP/unified_control_plane_service.py`, `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`; Verification: focused and broader Unified MCP pytest coverage now verifies the landed browse/action seams.
- Recommended tranche: Tranche 3
- Notes: Priority 27. The user explicitly deprioritized remote MCP governance relative to local MCP runtime, so follow-on work here should stay source-explicit and focus on presentation and operator usability rather than reopening route-family coverage that is already landed.

## Deferred / Explicitly Out Of Scope

- No additional Task 5 deferrals were added beyond the lower-priority remote-only and contract-maturity rows already captured above.
