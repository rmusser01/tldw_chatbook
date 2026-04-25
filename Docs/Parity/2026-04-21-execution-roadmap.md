# Chatbook Server Execution Roadmap

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

Tranche ordering follows the capability matrix, gap ledger, and target-state design. The sequence is intentionally standalone-first: lock the runtime-policy and capability-map prerequisites first, strengthen domains that already have partial dual-backend seams next, then add the highest-value missing local and remote surfaces, and only after that spend effort on remote-only convenience layers.

## Status Update

`Tranche 0: Runtime Policy And Capability Map` is now landed and verified. See [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md) for the authoritative verification record.

That changes the roadmap in one important way: runtime policy remains a cross-cutting extension surface, but it is no longer the next unresolved blocker. Active execution should now move into the domains that either build directly on the new authority layer or deliver the highest-value standalone-client parity you prioritized.

Additional backend foundations have since landed for media/read-it-later, collections feeds, watchlists/client notifications, writing suite, research sessions/runs, remote outputs, sharing, web clipper, study packs/suggestions, server notification/reminder wrappers, broad server Claims control/query wrappers, audio REST, LLM provider/model catalog discovery, server runtime/config discovery, auth/profile/account-security/user-storage access, user governance/consent, external connectors, chat grammars, explicit feedback, server Skills, Prompt Studio REST/SSE, and Kanban board/list/card plus non-workflow subresource REST. Media now includes direct local URL reading-item creation, local reading-highlight CRUD/export, local document annotation CRUD/sync, deterministic local document intelligence, local ingestion-source/job execution for local directories, archive snapshots, git repositories, URL article/file jobs, local file jobs, reading imports, and TTS plus machine-readable unsupported reporting for per-media-type server saved views and server ingestion-source deletion; watchlists now include source CRUD, local/server run list/detail/launch/observe seams, alert-rule CRUD, local completed-run alert evaluation, notification delivery through the central Chatbook notification dispatcher, centralized group-edit hard stops, and source-scoped unsupported reporting. Collections feeds, sharing, web clipper, server reminders/notification feeds, broad server Claims controls, audio REST, LLM provider/model catalog discovery, server runtime/config discovery, auth/profile/account-security/user-storage access, user governance/consent, external connectors, chat grammars, explicit feedback, server Skills, Prompt Studio REST/SSE, and Kanban REST now have direct client or remote-owned seams with explicit local/server authority boundaries. The remaining work in those rows is now primarily adoption/UI, contract edge cleanup, remaining producer adoption, or explicitly deferred sync/mirroring behavior rather than absence of a service seam.

## Immediate Post-Tranche-0 Order

1. `Media / Reading / Ingestion Sources` plus `Collections: Reading List / Read-it-later`
   This is the cleanest next crosswalk because Chatbook already has landed local saved-state persistence, direct local URL reading-item creation, local reading-highlight CRUD/export, local document annotation CRUD/sync, deterministic local document intelligence, local saved-search/note-link storage, local bulk status/tag/delete updates, local durable archive snapshots, local extractive summaries, local TTS, local export/import execution and import-job tracking, server URL save, server saved-search CRUD, server note links, server reading bulk update/archive/summary/import-job/export/TTS wrappers, server save/remove compatibility, the aggregate `All Media` saved view with authoritative capability/normalization handling, server ingestion-source create for `archive_snapshot` and `git_repository`, local/server ingestion-source item reattach, local queued ingestion-source/job persistence, local-directory source item materialization, local archive snapshot upload/sync materialization, local git repository source materialization, local URL article/file ingest execution, local file ingest execution, and scope-level unsupported reporting; per-media-type server saved views and sync/mirror semantics remain deferred.
2. `Watchlists` plus `Client Notifications`
   Chatbook already has a real local subscriptions stack, source CRUD/run-lifecycle/alert-rule service seams, executable local run handling, local completed-run alert evaluation, durable local notification queue/settings services, toast-like notification plumbing, central group-edit hard stops, and unsupported reporting, so the remaining high-value bridge is adoption into the eventual watchlists management UX, source-type executor hardening, plus remaining notification producer coverage.
3. `Local MCP Runtime`
   The runtime-policy layer is now in place, and Chatbook already has local MCP modules; the next gap is turning that into a first-class local runtime, approvals, status, and governance surface.
4. `Writing Suite`
   This remains a high-value standalone row with a clear server contract. The backend seam is now app-wired and covers project/manuscript/chapter/scene CRUD, source-scoped structure retrieval, local direct manuscript-level scenes, local manual version snapshots/restores, local trash listing/restores, reorder/move helpers, Markdown-preserving server scene writes, and machine-readable unsupported-capability reporting for known server contract gaps. The next useful work is eventual UX adoption.
5. `Research Sessions / Runs`
   Research is a user-priority local-first domain. The backend seam now covers local session/run persistence, event/artifact/bundle retrieval, server run launch/list/detail/update/observe/bundle/artifact wrappers, source-normalized records, and unsupported reporting for server run deletion; remaining work is UX adoption and any future server session semantics beyond run-centric execution.
6. `Workflows`, `Scheduler Workflows`, and `Chat Workflows`
   These remain remote-only acceptable, but once the local-first rows above are in place, Chatbook should add discover/trigger/observe support for connected-server workflow surfaces.
7. `Server Reminders / Notification Feeds`, `Claims Notifications / Alerts`, `Sharing`, and `Web Clipper`
   Backend wrappers now exist for these remote surfaces. They should still follow the higher-value standalone rows because the remaining work is UX adoption, offline-unavailable presentation, and producer/browser-extension handoff rather than service construction.

## Tranche 0: Runtime Policy And Capability Map

This tranche was the prerequisite layer for the rest of the roadmap. It locked the operating rules that keep Chatbook standalone-first while still allowing clean server interoperability: source-labeled local versus server mode, explicit write authority, offline fallback behavior, tenancy-aware server usage, and evidence-backed capability mapping. The capability matrix and gap ledger already established the baseline; this tranche turned that runtime-policy contract into an actual gate for later execution rather than letting downstream verticals invent their own source or authority rules.

Domains in this tranche:

- `Cross-cutting Runtime Policy`

Execution intent for this tranche:

- Keep the capability matrix and gap ledger as the source of truth for what is dual-backend, local-first, remote-only, or contract-maturity constrained.
- Normalize source labels, authority rules, and offline-unavailable states before domain work starts to avoid parity regressions later.
- Use this tranche to define the shared runtime seams that Tranche 1 and Tranche 2 work will extend rather than re-decide.
- Status: landed. Continue extending it as later tranches add new domains rather than reopening the authority model.

## Tranche 1: Strengthen Existing Dual-Backend Domains

This tranche should focus on the domains where Chatbook already has meaningful local/server seams, partial client adapters, or source-aware UI surfaces. These rows are the fastest path to credible parity because they mostly require contract cleanup, identifier normalization, mode separation hardening, and finishing missing CRUD or observe paths instead of inventing entirely new product surfaces. This is the right place to consolidate the evidence-building work from the matrix into durable client contracts.

Domains in this tranche:

- `Chat`
- `Characters / Personas / CCP`
- `Notes / Workspaces`
- `Media / Reading / Ingestion Sources`
- `Prompts / Chatbooks`
- `Study Core`
- `Evaluations`
- `RAG / Embeddings / Chunking Admin`

Execution intent for this tranche:

- Finish the strongest existing local/server seams before building fresh remote-only surfaces.
- Current implementation note: source-aware backend foundations are now landed for chat conversation metadata/history, character/persona chat-session administration, chat dictionaries, notes/workspaces including server notes-graph operations, media/reading/ingestion, watchlists, study/evaluations, RAG/admin, research sessions/runs, research search/provider launch, and prompts/chatbooks. Chat now has local/server conversation list/detail/update/tree routing, server messages-with-context/citations wrappers, local create/delete, explicit server unsupported boundaries for missing first-class conversation create/delete endpoints, explicit local unsupported boundaries for RAG-context adjuncts, and source-scoped unsupported reporting for those chat contract limits. Chat grammars now has direct server wrappers for saved GBNF grammar create/list/detail/update/delete while local grammar-library storage and chat-launch adoption remain follow-on. Explicit feedback now has direct wrappers for chat/RAG feedback submit/list/update/delete while local persistence, analytics UX, and source-aware adoption remain follow-on. Character/persona now has local character-card search/detail/create/update/delete/restore plus local character/persona session metadata CRUD/restore/export through a source-aware adapter, source-aware local/server chat-dictionary CRUD/entry/import/export/process routing, server wrappers and scope routing for character search/detail/create/update/delete/restore, character exemplar search/detail/create/update/delete/selection-debug, character/persona chat-session CRUD/restore, server character-message list/detail/create/update/delete/search, server persona exemplar CRUD/import/review, per-chat settings, chat export, author-note info, lorebook diagnostics, server chat-dictionary/world-book administration, and source-scoped unsupported reporting for local persona-profile/exemplar/execution-surface and dictionary history/version gaps. Notes/workspaces now has policy-gated server graph fetch/neighbor/manual-link wrappers and source-scoped unsupported reporting for local/workspace graph calls while local graph generation remains explicitly deferred. Prompts/chatbooks now includes local/server prompt version list/restore, prompt CRUD, local persistent chatbook record CRUD, chatbook archive job seams including server continuation export, export/import job list/detail/cancel/remove, completed-export download routing, and source-scoped unsupported reporting for server persistent chatbook record CRUD. Media ingestion now includes direct local URL reading-item creation, local reading-highlight CRUD/export, local document annotation CRUD/sync, deterministic local document intelligence, local source/job execution for local directories, archive snapshots, git repositories, URL article/file jobs, local file jobs, reading imports, local TTS, local saved-search/note-link/bulk-update/archive/summary/export/import-job/source-item reattach storage, server file-artifact create/detail/reference/delete/export/purge wrappers, server OCR/VLM backend discovery and OCR POINTS preload wrappers, and policy-gated server URL save, saved-search CRUD, note-link, reading bulk update, archive snapshot, summary, import-job, export, TTS, and ingestion-source item reattach wrappers. Collections feeds now has direct server wrappers for feed create/list/detail/update/delete with returned health and schedule metadata while source-aware routing and local subscription mirror semantics remain follow-on. Audio REST now includes direct server wrappers for TTS/STT health, provider/voice discovery, speech generation/jobs/artifacts, audio job submit/status/SSE, TTS history, transcription/translation, tokenizer encode/decode, custom voice management, and audiobook REST while websocket and admin audio-job endpoints remain follow-on. LLM provider/model catalog discovery now includes direct server wrappers for inference health, configured providers, provider detail, flattened model metadata with modality filters, and available model IDs while server provider process control and provider configuration mutation remain deferred. Server runtime/config discovery now includes direct server wrappers for health/live/ready/metrics/security, docs-info capabilities, flashcards import limits, tokenizer get/update, jobs config, provider key status, and provider-key validation while admin config mutation remains follow-on. Auth/profile/account access now includes direct server wrappers for login, refresh, logout, auth-scoped sessions, registration, profile catalog, self profile fetch/update, password reset/change, email verification, magic-link auth, MFA setup/verify/disable/login, per-user API keys, BYOK provider keys, OpenAI OAuth source control, self storage quota, and non-admin generated-file storage/folder/usage/trash operations while durable credential storage, token refresh policy, and server switching cache invalidation remain follow-on. User governance now includes direct server wrappers for authenticated-user consent read/grant/withdraw and self privilege-map discovery while org/team maps, snapshots, exports, and resource-governor policy admin remain out of scope. External connectors now includes direct wrappers for provider discovery/OAuth URL/callback, accounts, source browse/create/list/patch, import/sync trigger/status, and connector job status while org policy admin, inbound webhooks, and local mirror semantics remain follow-on or out of scope. Server Skills now includes direct wrappers for skill list/context/detail/create/update/delete/text-import/file-import/export/execute/seed operations while local mirror semantics and invocation UX remain follow-on. Outputs/artifacts now include direct server data-table wrappers for generate/list/detail/export/update/delete/regenerate/job operations plus direct slides/presentation wrappers for CRUD/search, templates/styles, versions, generation, render jobs, artifacts, export, and health. Kanban REST now includes direct server wrappers for board/list/card CRUD, labels, comments, checklists, content links, activity feeds, search/filter, import/export, enhanced copy, bulk card/link operations, nested detail response shapes, and optimistic-lock update headers while workflow controls remain deferred with broader workflows. Meetings now has direct server wrappers for session/template/artifact/finalize/share operations and SSE event observation, with websocket live-ingest still deferred. Study now reports the local workspace-scope boundary for deck/review and quiz/attempt flows while keeping server workspace filtering and creation source-separated. Evaluations now include local persisted per-run dataset override/webhook URL launch config, policy-gated server RAG-pipeline preset admin/cleanup wrappers, and server embeddings A/B test admin wrappers while still reporting local webhook delivery plus missing server target catalog and sample-level result detail through the evaluation scope seam. RAG/admin now includes local template apply/tag handling, server collection create, and local collection export while still reporting missing server collection export through the RAG-admin scope seam. Research search now has local web/paper search and server web/paper search behind a source-scoped provider catalog, and direct server adapters for these active parity rows now participate in runtime-policy hard stops. A small remote translation utility wrapper is also available through typed API-client schemas, but it remains a remote utility rather than a local-first product surface. Remaining tranche work is primarily UI adoption, broader endpoint coverage, and honest unsupported-boundary reduction rather than first-pass service construction.
- Flashcards API-client update: Chatbook now has typed wrappers for the broader server flashcards REST surface, including assets, bulk mutations, tags, scheduling reset, structured imports, JSON/APKG import, review sessions, assistant, generation, export, templates, analytics, and workspace query passthrough. Source-aware service adoption and UX remain separate follow-on work.
- Prefer source-separated crosswalks and normalization work over redesigning mature local flows.
- Use these domains to prove the runtime-policy model in everyday product surfaces before expanding to more ambitious missing capabilities.

## Tranche 2: Add Missing High-Value Local / Remote Surfaces

This tranche adds the highest-value local-first domains that are still thin or absent in Chatbook, but only after Tranche 0 and Tranche 1 have established stable policy and reusable dual-backend patterns. The priority here is to crosswalk existing local surfaces where possible instead of treating every gap as a blank-sheet replacement. That bias matters most for the user-priority standalone rows called out in the audit: read-it-later, watchlists, writing, research, local notifications, and local MCP runtime.

Domains in this tranche:

- `Collections: Reading List / Read-it-later`
- `Watchlists`
- `Writing Suite`
- `Research Sessions / Runs`
- `Audio / Speech Services`
- `LLM Provider / Model Catalog`
- `Server Runtime / Config Discovery`
- `Auth / Profile / Sessions`
- `Client Notifications`
- `Local MCP Runtime`

Execution intent for this tranche:

- Crosswalk `Watchlists` onto existing local subscriptions and notification plumbing before inventing a separate remote-first model. Treat source CRUD, run list/detail/launch/observe, executable local run handling, alert-rule CRUD, completed-run alert notification dispatch, and the local notification queue/settings backend seam as landed foundations; focus next on UI adoption, group editing policy, source-type executor hardening, and remaining producer adoption.
- Build `Collections: Reading List / Read-it-later` on top of the existing media and reading seams rather than treating it as a disconnected collection system. Treat local saved-state persistence, local saved-search/note-link/bulk-update/archive/summary storage, server URL save, server saved-search CRUD, server note links, server save/remove compatibility, the aggregate `All Media` saved view, server ingestion-source create, and local queued ingestion-source/job persistence as landed foundations; keep per-media-type server saved views and any sync/mirror contract out of scope for this tranche.
- Treat the landed `Writing Suite` backend seam as the base for a serious standalone surface. Local direct manuscript-level scenes, manual versions, trash restore/listing, and reorder/move helpers are now represented in the backend seam; server direct manuscript-level scenes, version history, and trash restore remain honest unsupported boundaries with source-scoped reporting until the server contract exposes them.
- Keep `Audio / Speech Services` source-separated: use the landed REST client wrappers for connected-server TTS/STT discovery, generation, jobs, history, transcription, tokenizer encode/decode, custom voice management, and audiobook operations while leaving websocket speech/chat and admin job surfaces as separate contract slices.
- Keep `LLM Provider / Model Catalog` source-separated: observe active-server provider/model availability for connected workflows without rewriting Chatbook local provider configuration; defer provider process control, server-side provider config mutation, and sync/mirror semantics.
- Keep `Server Runtime / Config Discovery` active-server scoped: use it for capability gating, status, and safe config discovery without treating server health/config as local state.
- Keep `Auth / Profile / Sessions` remote-owned: use server identity only for connected mode and do not mirror server accounts into Chatbook's local single-user identity.
- Keep `Client Notifications` and `Local MCP Runtime` Chatbook-owned so offline capability remains credible even as remote interop improves.

## Tranche 3: Remote-Only Surfaces And Convenience Layers

This tranche is intentionally last. These rows matter for connected-server completeness, but they should not outrank the runtime-policy foundation or the core local/remote parity domains above. The dominant pattern here is remote discovery, trigger, configuration, and status UX with explicit offline-unavailable behavior. Some rows are straightforward remote-only convenience layers, while others should stay gated behind further contract-maturity confirmation.

Domains in this tranche:

- `Study Packs`
- `Study Suggestions`
- `Collections: Feed Subscriptions`
- `Collections: Outputs / Templates / Artifacts`
- `Prompt Studio`
- `Kanban Boards / Tasks`
- `Chat Grammars`
- `Explicit Feedback`
- `User Governance / Consent`
- `External Connectors`
- `Server Skills`
- `Research Search / Provider Surfaces`
- `Server Reminders / Notification Feeds`
- `Claims Notifications / Alerts`
- `Workflows`
- `Scheduler Workflows`
- `Chat Workflows`
- `Remote MCP Control Plane / Governance`
- `Sharing`
- `Web Clipper`

Execution intent for this tranche:

- Keep remote-only workflows and remote MCP governance behind the core standalone-first roadmap rather than letting them pull priority forward.
- Treat `Sharing`, `Web Clipper`, `Study Packs`, `Study Suggestions`, `Collections: Feed Subscriptions`, `Collections: Outputs / Templates / Artifacts`, `Prompt Studio`, `Kanban Boards / Tasks`, `Chat Grammars`, `Explicit Feedback`, `User Governance / Consent`, `External Connectors`, `Server Skills`, `Research Search / Provider Surfaces`, and `Claims Notifications / Alerts` as backend-seam-present rows whose next work is adoption, offline-unavailable presentation, realtime/stream edge cleanup where relevant, and contract edge cleanup rather than first-pass client construction. For output-template renders, Prompt Studio websocket/ping diagnostics, Kanban workflow controls, collections-feed local subscription mirror rules, chat-grammar launch/mirror rules, feedback source/mirror rules, user-governance org/team/admin privilege surfaces, connector org policy/webhook/mirror rules, Skills local mirror rules, Research Search, and Study Packs, keep contract gaps explicit through scope-service unsupported reports until the server exposes or Chatbook adopts those contract slices cleanly.
- Reuse the source-labeling, offline fallback, and discover/trigger/observe patterns proven in earlier tranches instead of creating special-case remote UI rules.

## Follow-On Vertical Plans

- `Collections: Reading List / Read-it-later parity`: Turn the tranche guidance into a focused plan for the standalone-first reading collection surface, using the existing media and reading seams as the base for local and remote collection alignment. Keep the plan scoped to the landed aggregate saved view and compatibility mapping unless a separate sync/mirror design is approved.
- `Watchlists / subscriptions alignment`: Turn the tranche guidance into a focused plan that maps local subscriptions, the landed source/run/alert-rule seams, and notification delivery onto the server watchlist vocabulary without collapsing local ownership.
- `Local MCP runtime parity`: Define the local-first runtime, approvals, catalog, prompts, tools, resources, and status surface before any later remote governance work.
- `Remote MCP control plane`: Define the later remote governance surface separately from local MCP runtime so catalog and approval policy do not subsume standalone Chatbook control.
- `Writing suite parity`: Plan the local-first project, manuscript, chapter, and scene hierarchy together with the server contract and source-separated UI behavior.
- `Research sessions parity`: Plan the local and remote session lifecycle, run execution, streaming status, and bundle retrieval surface as a standalone-first research vertical.
- `Remote workflows surface`: Split general workflows, scheduler workflows, and chat workflows into a dedicated remote-only plan that covers discovery, scheduler configuration and scheduling control-plane behavior, launch, run status, and observation after the core parity rows land.
- `Study packs / study suggestions adoption`: Build on the existing server wrappers and scope-service routing once the UX layer is ready; keep local generation out of scope unless a separate local execution plan is approved.
