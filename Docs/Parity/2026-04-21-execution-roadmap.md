# Chatbook Server Execution Roadmap

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

Tranche ordering follows the capability matrix, gap ledger, and target-state design. The sequence is intentionally standalone-first: lock the runtime-policy and capability-map prerequisites first, strengthen domains that already have partial dual-backend seams next, then add the highest-value missing local and remote surfaces, and only after that spend effort on remote-only convenience layers.

## Status Update

`Tranche 0: Runtime Policy And Capability Map` is now landed and verified. See [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md) for the authoritative verification record.

That changes the roadmap in one important way: runtime policy remains a cross-cutting extension surface, but it is no longer the next unresolved blocker. Active execution should now move into the domains that either build directly on the new authority layer or deliver the highest-value standalone-client parity you prioritized.

Additional backend foundations have since landed for media/read-it-later, watchlists/client notifications, writing suite, research sessions/runs, remote outputs, sharing, web clipper, study packs/suggestions, and server notification/reminder wrappers. Media now includes a local ingestion-source/job queue seam plus machine-readable unsupported reporting for local execution/TTS/import materialization, per-media-type server saved views, and server ingestion-source deletion; watchlists now include source CRUD, local/server run list/detail/launch/observe seams, alert-rule CRUD, local completed-run alert evaluation, notification delivery through the central Chatbook notification dispatcher, centralized group-edit hard stops, and source-scoped unsupported reporting. Sharing, web clipper, and server reminders/notification feeds now have remote-owned source seams with explicit local-unavailable behavior and machine-readable unsupported reports for known server contract gaps. The remaining work in those rows is now primarily adoption/UI, actual local execution, contract edge cleanup, remaining producer adoption, or explicitly deferred sync/mirroring behavior rather than absence of a service seam.

## Immediate Post-Tranche-0 Order

1. `Media / Reading / Ingestion Sources` plus `Collections: Reading List / Read-it-later`
   This is the cleanest next crosswalk because Chatbook already has landed local saved-state persistence, local saved-search/note-link storage, local bulk status/tag/delete updates, local durable archive snapshots, local extractive summaries, local TTS, local export/import execution and import-job tracking, server URL save, server saved-search CRUD, server note links, server reading bulk update/archive/summary/import-job/export/TTS wrappers, server save/remove compatibility, the aggregate `All Media` saved view, server ingestion-source create for `archive_snapshot` and `git_repository`, local/server ingestion-source item reattach, local queued ingestion-source/job persistence, local-directory source item materialization, local URL article ingest execution, local file ingest execution, and scope-level unsupported reporting; URL file/download ingest execution, non-directory source execution, per-media-type server saved views, and sync/mirror semantics remain deferred.
2. `Watchlists` plus `Client Notifications`
   Chatbook already has a real local subscriptions stack, source CRUD/run-lifecycle/alert-rule service seams, local completed-run alert evaluation, durable local notification queue/settings services, toast-like notification plumbing, central group-edit hard stops, and unsupported reporting, so the remaining high-value bridge is actual local run execution, adoption into the eventual watchlists management UX, plus remaining producer coverage.
3. `Local MCP Runtime`
   The runtime-policy layer is now in place, and Chatbook already has local MCP modules; the next gap is turning that into a first-class local runtime, approvals, status, and governance surface.
4. `Writing Suite`
   This remains a high-value standalone row with a clear server contract. The backend seam is now app-wired and covers project/manuscript/chapter/scene CRUD, source-scoped structure retrieval, local direct manuscript-level scenes, local manual version snapshots/restores, local trash listing/restores, reorder/move helpers, Markdown-preserving server scene writes, and machine-readable unsupported-capability reporting for known server contract gaps. The next useful work is eventual UX adoption.
5. `Research Sessions / Runs`
   Research is a user-priority local-first domain. The backend seam now covers local session/run persistence, event/artifact/bundle retrieval, server run launch/list/detail/update/observe/bundle/artifact wrappers, source-normalized records, and unsupported reporting for server run deletion; remaining work is UX adoption and any future server session semantics beyond run-centric execution.
6. `Workflows`, `Scheduler Workflows`, and `Chat Workflows`
   These remain remote-only acceptable, but once the local-first rows above are in place, Chatbook should add discover/trigger/observe support for connected-server workflow surfaces.
7. `Server Reminders / Notification Feeds`, `Sharing`, and `Web Clipper`
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
- Current implementation note: source-aware backend foundations are now landed for chat conversation metadata/history, character/persona chat-session administration, notes/workspaces including server notes-graph operations, media/reading/ingestion, watchlists, study/evaluations, RAG/admin, research sessions/runs, research search/provider launch, and prompts/chatbooks. Chat now has local/server conversation list/detail/update/tree routing, server messages-with-context/citations wrappers, local create/delete, explicit server unsupported boundaries for missing first-class conversation create/delete endpoints, explicit local unsupported boundaries for RAG-context adjuncts, and source-scoped unsupported reporting for those chat contract limits. Character/persona now has server wrappers and scope routing for character/persona chat-session CRUD/restore, per-chat settings, chat export, author-note info, lorebook diagnostics, server chat-dictionary/world-book administration, and source-scoped unsupported reporting for local legacy-scope and server character-message mutation gaps in addition to persona profiles, greetings, and presets. Notes/workspaces now has policy-gated server graph fetch/neighbor/manual-link wrappers and source-scoped unsupported reporting for local/workspace graph calls while local graph generation remains explicitly deferred. Prompts/chatbooks now includes policy-gated server prompt version list/restore, prompt CRUD, chatbook archive job seams, and source-scoped unsupported reporting for local prompt versions and persistent chatbook record CRUD. Media ingestion now includes local queued source/job persistence plus local saved-search/note-link/bulk-update/archive/summary/export/import-job/source-item reattach storage and policy-gated server URL save, saved-search CRUD, note-link, reading bulk update, archive snapshot, summary, import-job, export, TTS, and ingestion-source item reattach wrappers. Study now reports the local workspace-scope boundary for deck/review and quiz/attempt flows while keeping server workspace filtering and creation source-separated. Evaluations now report local run override/webhook limits plus missing server target catalog, sample-level result detail, RAG-pipeline admin wrappers, and embedding A-B test wrappers through the evaluation scope seam. RAG/admin now reports local template apply/tag gaps, local collection export, and missing server collection create/export contracts through the RAG-admin scope seam. Research search now has local websearch and server web/paper search behind a source-scoped provider catalog, and direct server adapters for these active parity rows now participate in runtime-policy hard stops. Remaining tranche work is primarily UI adoption, broader endpoint coverage, and honest unsupported-boundary reduction rather than first-pass service construction.
- Prefer source-separated crosswalks and normalization work over redesigning mature local flows.
- Use these domains to prove the runtime-policy model in everyday product surfaces before expanding to more ambitious missing capabilities.

## Tranche 2: Add Missing High-Value Local / Remote Surfaces

This tranche adds the highest-value local-first domains that are still thin or absent in Chatbook, but only after Tranche 0 and Tranche 1 have established stable policy and reusable dual-backend patterns. The priority here is to crosswalk existing local surfaces where possible instead of treating every gap as a blank-sheet replacement. That bias matters most for the user-priority standalone rows called out in the audit: read-it-later, watchlists, writing, research, local notifications, and local MCP runtime.

Domains in this tranche:

- `Collections: Reading List / Read-it-later`
- `Watchlists`
- `Writing Suite`
- `Research Sessions / Runs`
- `Client Notifications`
- `Local MCP Runtime`

Execution intent for this tranche:

- Crosswalk `Watchlists` onto existing local subscriptions and notification plumbing before inventing a separate remote-first model. Treat source CRUD, run list/detail/launch/observe, alert-rule CRUD, completed-run alert notification dispatch, and the local notification queue/settings backend seam as landed foundations; focus next on UI adoption, group editing policy, remaining producer adoption, and actual local execution.
- Build `Collections: Reading List / Read-it-later` on top of the existing media and reading seams rather than treating it as a disconnected collection system. Treat local saved-state persistence, local saved-search/note-link/bulk-update/archive/summary storage, server URL save, server saved-search CRUD, server note links, server save/remove compatibility, the aggregate `All Media` saved view, server ingestion-source create, and local queued ingestion-source/job persistence as landed foundations; keep per-media-type server saved views and any sync/mirror contract out of scope for this tranche.
- Treat the landed `Writing Suite` backend seam as the base for a serious standalone surface. Local direct manuscript-level scenes, manual versions, trash restore/listing, and reorder/move helpers are now represented in the backend seam; server direct manuscript-level scenes, version history, and trash restore remain honest unsupported boundaries with source-scoped reporting until the server contract exposes them.
- Keep `Client Notifications` and `Local MCP Runtime` Chatbook-owned so offline capability remains credible even as remote interop improves.

## Tranche 3: Remote-Only Surfaces And Convenience Layers

This tranche is intentionally last. These rows matter for connected-server completeness, but they should not outrank the runtime-policy foundation or the core local/remote parity domains above. The dominant pattern here is remote discovery, trigger, configuration, and status UX with explicit offline-unavailable behavior. Some rows are straightforward remote-only convenience layers, while others should stay gated behind further contract-maturity confirmation.

Domains in this tranche:

- `Study Packs`
- `Study Suggestions`
- `Collections: Outputs / Templates / Artifacts`
- `Research Search / Provider Surfaces`
- `Server Reminders / Notification Feeds`
- `Workflows`
- `Scheduler Workflows`
- `Chat Workflows`
- `Remote MCP Control Plane / Governance`
- `Sharing`
- `Web Clipper`

Execution intent for this tranche:

- Keep remote-only workflows and remote MCP governance behind the core standalone-first roadmap rather than letting them pull priority forward.
- Treat `Sharing`, `Web Clipper`, `Study Packs`, `Study Suggestions`, `Collections: Outputs / Templates / Artifacts`, and `Research Search / Provider Surfaces` as backend-seam-present rows whose next work is adoption, offline-unavailable presentation, and edge cleanup rather than first-pass client construction. For Outputs, Research Search, and Study Packs, keep contract gaps explicit through scope-service unsupported reports until the server exposes first-class asynchronous render-job, provider configure/observe, and study-pack job-list contracts.
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
