# Chatbook Server Parity Gap Ledger

## Critical Gaps

### Collections: Reading List / Read-it-later
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local saved-reading and read-it-later behavior already exists through the media reading surfaces, but the remote collection shape is not yet aligned.
- Gap: Remote collection parity and identifier normalization are still incomplete for a user-priority standalone collection surface.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/items.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/media_reading_scope_service.py`, `tldw_chatbook/UI/Screens/media_screen.py`; Verification: reading and item routes cover the saved-reading collection surface.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This is one of the clearest standalone-first parity rows and should stay source-separated until later sync work exists.

### Watchlists: Watchlists
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local subscriptions plus notification plumbing already provide a practical local precursor, but the server watchlist vocabulary is not mapped into Chatbook yet.
- Gap: Remote watchlist, source, run, and alert-rule alignment is missing despite strong local user value.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`; Chatbook: `tldw_chatbook/DB/Subscriptions_DB.py`, `tldw_chatbook/UI/Screens/subscription_screen.py`, `tldw_chatbook/Widgets/toast_notification.py`; Verification: stable watchlist and alert-rule routers exist server-side.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This is the strongest local-name crosswalk in the matrix and directly matches the user's standalone monitoring priority.

### Writing Suite: Writing Suite
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook only has adjacent writing helpers today and no dedicated project or manuscript hierarchy.
- Gap: The local-first writing product surface is absent even though the server contract is explicit and high value.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`; Chatbook: adjacent helpers in `tldw_chatbook/UI/ChatbookTemplatesWindow.py` and `tldw_chatbook/Widgets/TTS/chapter_editor_widget.py`; Verification: writing and manuscript modules expose project, chapter, and scene routes.
- Recommended tranche: Tranche 2
- Notes: Priority 78. This is a user-priority standalone gap, not cosmetic work.

### Research Sessions / Runs: Research Sessions / Runs
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: No dedicated Chatbook research session or run UX exists; only adjacent eval tooling is present.
- Gap: Chatbook lacks the local-first research session lifecycle the user explicitly wants, and the remote run contract is not surfaced either.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research_runs.py`, `../tldw_server/tldw_Server_API/app/core/Research/service.py`, `../tldw_server/tldw_Server_API/app/core/Research/streaming.py`; Chatbook: adjacent eval tooling in `tldw_chatbook/Evaluations_Interop/` and `tldw_chatbook/Widgets/Evals/`; Verification: stable endpoint-backed session/run lifecycle with streaming and bundle helpers exists.
- Recommended tranche: Tranche 2
- Notes: Priority 75. This is one of the most important missing standalone product surfaces in the audit.

### Local MCP Runtime: Local MCP Runtime
- Requirement class: Local parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local MCP modules and config exist, but integrated runtime, catalog, approvals, and governance UX are only partial.
- Gap: Chatbook does not yet present a serious local MCP runtime surface even though local operations were explicitly prioritized.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/server.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/protocol.py`; Chatbook: `tldw_chatbook/config.py`; Verification: unified MCP runtime exposes request, batch, status, module, resource, and tool execution paths.
- Recommended tranche: Tranche 2
- Notes: Priority 73. Local MCP should remain Chatbook-owned even if remote catalogs are imported later.

### Client Notifications: Client Notifications
- Requirement class: Local parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook can deliver local notifications and toast-like events, but it lacks a dedicated notification center or normalized local contract.
- Gap: A key standalone surface for watchlists, research, and local operations is still fragmented and under-specified.
- Evidence: Server: adjacent remote counterparts in `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py` and `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`; Chatbook: `tldw_chatbook/Widgets/toast_notification.py`, `tldw_chatbook/UI/Screens/subscription_screen.py`, `tldw_chatbook/config.py`; Verification: the client-local contract remains immature and separate from server feeds/reminders.
- Recommended tranche: Tranche 2
- Notes: Priority 71. Even with lower interop value, this stays high because it supports several top standalone rows.

### Cross-cutting Runtime Policy: Cross-cutting Runtime Policy
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Mode, auth, feature, and policy controls exist piecemeal across Chatbook, but not as one coherent source-labeled layer.
- Gap: The app still lacks the runtime-policy foundation needed to keep local/server authority explicit across the rest of the parity work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/core/feature_flags.py`, `../tldw_server/tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces_rate_limit_policy.py`, `../tldw_server/tldw_Server_API/app/core/MCP_unified/server.py`; Chatbook: `tldw_chatbook/config.py`, `tldw_chatbook/UI/Screens/chat_screen_state.py`, `tldw_chatbook/Widgets/enhanced_settings_sidebar.py`, `tldw_chatbook/tldw_api/client.py`; Verification: policy concerns are present, but not unified.
- Recommended tranche: Tranche 0
- Notes: Priority 70. This is leverage work for the rest of the matrix, not a cosmetic polish item.

## High-Value Partial Crosswalks

### Chat: Chat
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Local chat is already strong and mode-aware, while remote conversation interoperability remains thinner.
- Gap: Remote conversation contract alignment and explicit source separation still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_loop.py`; Chatbook: `tldw_chatbook/tldw_api/client.py`, `tldw_chatbook/UI/Chat_Window_Enhanced_Refactored.py`, `tldw_chatbook/UI/Screens/chat_screen_state.py`; Verification: dedicated chat routes exist and `chat_workflows.py` is intentionally scored separately.
- Recommended tranche: Tranche 1
- Notes: Priority 78. Chat remains a core credibility surface for Chatbook as a serious standalone client.

### Characters / Personas / CCP: Characters / Personas / CCP
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: CCP is mature locally, with explicit server route families already identified.
- Gap: Shared identifiers and remote contract alignment are still incomplete across characters, personas, sessions, and messages.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/persona.py`; Chatbook: `tldw_chatbook/UI/Screens/ccp_screen.py`, `tldw_chatbook/Widgets/CCP_Widgets/`; Verification: character, persona, session, message, and memory route families are explicit.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is an important existing seam to finish rather than a net-new product surface.

### Notes / Workspaces: Notes / Workspaces
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Dual-backend notes and workspace seams already exist, including scope services and server-aware notes surfaces.
- Gap: Graph parity and full remote normalization still need completion.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notes_graph.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces.py`; Chatbook: `tldw_chatbook/Notes/Notes_Library.py`, `tldw_chatbook/Notes/notes_scope_service.py`, `tldw_chatbook/Notes/server_notes_workspace_service.py`; Verification: notes, graph, and workspace endpoint families are separate and explicit.
- Recommended tranche: Tranche 1
- Notes: Priority 73. This is daily-use parity work with strong existing scaffolding.

### Media / Reading / Ingestion Sources: Media / Reading / Ingestion Sources
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Media and reading seams already span local and server modes, including reading progress and ingest-job paths.
- Gap: Full source, job, and contract parity is still incomplete.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/UI/Screens/media_screen.py`, `tldw_chatbook/UI/Screens/media_ingest_screen.py`; Verification: stable reading, ingestion-source, ingest-job, and reading-progress routes verify the split.
- Recommended tranche: Tranche 1
- Notes: Priority 78. This row underpins the higher-ranked reading-list and research work.

### Prompts / Chatbooks: Prompts / Chatbooks
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Create plus import/export paths exist, and there is already an explicit server chatbook service seam.
- Gap: Update, delete, and fully mode-separated parity are not finished.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chatbooks.py`; Chatbook: `tldw_chatbook/Chatbooks/chatbook_creator.py`, `tldw_chatbook/Chatbooks/chatbook_importer.py`, `tldw_chatbook/Chatbooks/server_chatbook_service.py`, `tldw_chatbook/UI/Screens/chatbooks_screen.py`; Verification: dedicated prompt and chatbook routers exist server-side.
- Recommended tranche: Tranche 1
- Notes: Priority 67. This remains important, but it does not outrank the standalone-first gaps above.

### Study Core: Study Core
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Study seams are already strong across local and server modes, including quizzes and study-guide generation.
- Gap: Full mutation parity and workspace-aware remote alignment still need cleanup.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/flashcards.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/quizzes.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chat_documents.py`; Chatbook: `tldw_chatbook/Study_Interop/local_study_service.py`, `tldw_chatbook/Study_Interop/server_study_service.py`, `tldw_chatbook/UI/Screens/study_screen.py`; Verification: flashcard, quiz, and study-guide contracts are explicit.
- Recommended tranche: Tranche 1
- Notes: Priority 63. Strong existing seams keep this below the most urgent standalone gaps.

### Evaluations: Evaluations
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook already has dual-backend evaluation services and screen scope support.
- Gap: Contract normalization and full mutation parity still need finishing work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`; Chatbook: `tldw_chatbook/Evaluations_Interop/local_evaluations_service.py`, `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py`, `tldw_chatbook/UI/Screens/evals_screen.py`; Verification: the evaluations package exposes a unified control plane plus specialized routes.
- Recommended tranche: Tranche 1
- Notes: Priority 58. This is meaningful parity work, but not ahead of the user-priority standalone surfaces.

### RAG / Embeddings / Chunking Admin: RAG / Embeddings / Chunking Admin
- Requirement class: Local parity required + Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local and server admin seams are already present for chunking, embeddings, and reprocess helpers.
- Gap: Full create/delete parity and source labeling still need cleanup.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chunking_templates.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/reprocess.py`; Chatbook: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`, `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`, `tldw_chatbook/Widgets/chunking_template_editor.py`; Verification: stable endpoint-backed embeddings and chunking surfaces exist.
- Recommended tranche: Tranche 1
- Notes: Priority 62. This is good leverage work, but it still trails the top standalone-first rows.

## Remote-Only Client Obligations

### Server Reminders / Notification Feeds: Server Reminders / Notification Feeds
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook does not expose a dedicated remote reminders or feed client yet.
- Gap: Discoverability and explicit offline fallback are still missing for server-owned reminders and notification feeds.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py`; Chatbook: nearest adjacent local plumbing in `tldw_chatbook/Widgets/toast_notification.py` and `tldw_chatbook/UI/Screens/subscription_screen.py`; Verification: reminder and notification feed routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 45. This matters for interoperability, but it is acceptable to keep remote-only.

### Workflows: Workflows
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: No workflow discovery, launch, or run-status surface exists in Chatbook.
- Gap: Chatbook cannot yet discover or observe remote workflow runs when connected.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/workflows.py`; Chatbook: no dedicated workflow client in `tldw_chatbook/tldw_api/client.py` or the local UI; Verification: workflow definition, launch, and runtime routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 33. Remote-only acceptable rows should not outrank the standalone-first rows above.

### Sharing: Sharing
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Chatbook has no dedicated sharing client today.
- Gap: Remote share discovery, permissions, and revocation are absent.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/sharing.py`; Chatbook: no dedicated sharing surface in `tldw_chatbook/tldw_api/client.py` or the local UI; Verification: share lifecycle routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This remains a server-owned convenience surface rather than a local-parity target.

### Web Clipper: Web Clipper
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: No Chatbook web-clipper client exists.
- Gap: Remote clip capture and status remain unimplemented in the client.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/web_clipper.py`; Chatbook: no dedicated clipper surface in `tldw_chatbook/tldw_api/client.py` or the local UI; Verification: save, status, and enrichment routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This should remain below the standalone-first reading and writing rows.

## Contract-Maturity Holds

### Research Search / Provider Surfaces: Research Search / Provider Surfaces
- Requirement class: Remote parity required, local parity assessed explicitly
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Local search exists, but provider behavior is spread across legacy route files and multiple server core integrations.
- Gap: The server research-provider contract is still too low-confidence to drive high-priority client parity work.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research.py`, `../tldw_server/tldw_Server_API/app/core/Research/providers/web.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Arxiv.py`, `../tldw_server/tldw_Server_API/app/core/Third_Party/Semantic_Scholar.py`; Chatbook: `tldw_chatbook/UI/Screens/search_screen.py`, `tldw_chatbook/config.py`; Verification: provider logic is present but split across core files and marked low-confidence in the matrix.
- Recommended tranche: Tranche 3
- Notes: Priority 31. Keep this on hold until the client-facing provider contract is clearer.

### Remote MCP Control Plane / Governance: Remote MCP Control Plane / Governance
- Requirement class: Remote parity required
- Client obligation: Discover / Configure / Trigger / Observe
- Current state: Current Chatbook MCP work is local-first and there is no dedicated remote governance client.
- Gap: The remote governance scope is explicit server-side but still admin-heavy and low-confidence as a client target.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/mcp_catalogs_manage.py`; Chatbook: `tldw_chatbook/config.py` with no dedicated governance surface; Verification: governance and catalog management routes exist, but the matrix confidence is low.
- Recommended tranche: Tranche 3
- Notes: Priority 27. The user explicitly deprioritized remote MCP governance relative to local MCP runtime.

## Deferred / Explicitly Out Of Scope

- No additional Task 5 deferrals were added beyond the lower-priority remote-only and contract-maturity rows already captured above.
