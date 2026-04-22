# Chatbook Server Parity Gap Ledger

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

## Status Update After Tranche 0

- `Cross-cutting Runtime Policy` is no longer an unlanded blocker. The foundational runtime-policy package, capability registry, hard-stop seams, representative UI preflight, and raw-client boundary were landed and verified in [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md).
- The remaining runtime-policy work is breadth and adoption across more domains and screens, not absence of the authority model itself.
- The active parity focus should therefore shift to the next user-priority standalone and remote-interop rows rather than treating runtime policy as still missing.
- `Watchlists` plus `Client Notifications` are now partially landed in the first-slice subscriptions-shell vertical. The verification record lives in [watchlists-notifications-tranche-2.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Docs/Development/watchlists-notifications-tranche-2.md).

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
- Current state: A source-aware subscriptions shell, local notifications inbox, and server watchlist source CRUD are now landed, with local mode still backed by subscriptions and server mode backed by live remote watchlist sources.
- Gap: Watchlist groups, jobs, runs, alert rules, restore UX, and any sync or mirror semantics are still deferred despite the strong local user value.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`; Chatbook: `tldw_chatbook/Notifications/client_notifications_db.py`, `tldw_chatbook/Notifications/notification_dispatch_service.py`, `tldw_chatbook/Subscriptions/local_watchlists_service.py`, `tldw_chatbook/Subscriptions/server_watchlists_service.py`, `tldw_chatbook/Subscriptions/watchlist_scope_service.py`, `tldw_chatbook/UI/SubscriptionWindow.py`; Verification: `Tests/tldw_api/test_watchlists_schemas.py`, `Tests/tldw_api/test_watchlists_client.py`, `Tests/Subscriptions/test_client_notifications_db.py`, `Tests/Subscriptions/test_notification_dispatch_service.py`, `Tests/Subscriptions/test_server_watchlists_service.py`, `Tests/Subscriptions/test_watchlist_scope_service.py`, `Tests/UI/test_screen_navigation.py`, and `Tests/UI/test_subscription_window_watchlists.py`.
- Recommended tranche: Tranche 2
- Notes: Priority 81. This is now a partially landed vertical; the remaining work is the broader watchlists execution and control-plane surface rather than first-slice source CRUD.

### Writing Suite: Writing projects and manuscript hierarchy
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Chatbook only has adjacent writing helpers today and no dedicated project or manuscript hierarchy.
- Gap: The local-first writing product surface is absent even though the server contract is explicit and high value.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`; Chatbook: adjacent helpers in `tldw_chatbook/UI/ChatbookTemplatesWindow.py` and `tldw_chatbook/Widgets/TTS/chapter_editor_widget.py`; Verification: writing and manuscript modules expose project, chapter, and scene routes.
- Recommended tranche: Tranche 2
- Notes: Priority 78. This is a user-priority standalone gap, not cosmetic work.

### Research Sessions / Runs: Deep research session lifecycle, streaming events, and bundle retrieval
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: No dedicated Chatbook research session or run UX exists; only adjacent eval tooling is present.
- Gap: Chatbook lacks the local-first research session lifecycle the user explicitly wants, and the remote run contract is not surfaced either.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/research_runs.py`, `../tldw_server/tldw_Server_API/app/core/Research/service.py`, `../tldw_server/tldw_Server_API/app/core/Research/streaming.py`; Chatbook: adjacent eval tooling in `tldw_chatbook/Evaluations_Interop/` and `tldw_chatbook/Widgets/Evals/`; Verification: stable endpoint-backed session/run lifecycle with streaming and bundle helpers exists.
- Recommended tranche: Tranche 2
- Notes: Priority 75. This is one of the most important missing standalone product surfaces in the audit.

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
- Gap: Broader notification producers, richer filtering or configuration, and any server reminder or notification-feed surface remain separate from the landed client-local contract.
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
- Current state: Media and reading seams already span local and server modes, including reading progress and ingest-job paths.
- Gap: Full source, job, and contract parity is still incomplete.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reading.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`; Chatbook: `tldw_chatbook/Media/local_media_reading_service.py`, `tldw_chatbook/Media/server_media_reading_service.py`, `tldw_chatbook/UI/Screens/media_screen.py`, `tldw_chatbook/UI/Screens/media_ingest_screen.py`; Verification: stable reading, ingestion-source, ingest-job, and reading-progress routes verify the split.
- Recommended tranche: Tranche 1
- Notes: Priority 78. This row underpins the higher-ranked reading-list and research work.

### Prompts / Chatbooks: Prompt library, prompt workflows, and chatbook import/export jobs
- Requirement class: Local parity required + Remote parity required
- Client obligation: Full CRUD
- Current state: Create plus import/export paths exist, and there is already an explicit server chatbook service seam.
- Gap: Update, delete, and fully mode-separated parity are not finished.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/prompts.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/chatbooks.py`; Chatbook: `tldw_chatbook/Chatbooks/chatbook_creator.py`, `tldw_chatbook/Chatbooks/chatbook_importer.py`, `tldw_chatbook/Chatbooks/server_chatbook_service.py`, `tldw_chatbook/UI/Screens/chatbooks_screen.py`; Verification: dedicated prompt and chatbook routers exist server-side.
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
- Current state: Chatbook does not expose a dedicated remote reminders or feed client yet.
- Gap: Discoverability and explicit offline fallback are still missing for server-owned reminders and notification feeds.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/reminders.py`, `../tldw_server/tldw_Server_API/app/api/v1/endpoints/notifications.py`; Chatbook: nearest adjacent local plumbing in `tldw_chatbook/Widgets/toast_notification.py` and `tldw_chatbook/UI/Screens/subscription_screen.py`; Verification: reminder and notification feed routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 45. This matters for interoperability, but it is acceptable to keep remote-only.

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
- Current state: Chatbook has no dedicated sharing client today.
- Gap: Remote share discovery, permissions, and revocation are absent.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/sharing.py`; Chatbook: no dedicated sharing surface in `tldw_chatbook/tldw_api/client.py` or the local UI; Verification: share lifecycle routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This remains a server-owned convenience surface rather than a local-parity target.

### Web Clipper: Browser clip save, status, and enrichment capture
- Requirement class: Remote-only acceptable
- Client obligation: Discover / Trigger / Observe
- Current state: No Chatbook web-clipper client exists.
- Gap: Remote clip capture and status remain unimplemented in the client.
- Evidence: Server: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/web_clipper.py`; Chatbook: no dedicated clipper surface in `tldw_chatbook/tldw_api/client.py` or the local UI; Verification: save, status, and enrichment routes are explicit.
- Recommended tranche: Tranche 3
- Notes: Priority 29. This should remain below the standalone-first reading and writing rows.

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
