# Chatbook Server Execution Roadmap

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

Tranche ordering follows the capability matrix, gap ledger, and target-state design. The sequence is intentionally standalone-first: lock the runtime-policy and capability-map prerequisites first, strengthen domains that already have partial dual-backend seams next, then add the highest-value missing local and remote surfaces, and only after that spend effort on remote-only convenience layers.

## Status Update

`Tranche 0: Runtime Policy And Capability Map` is now landed and verified. See [runtime-policy-tranche-0.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Docs/Development/runtime-policy-tranche-0.md) for the authoritative verification record.

That changes the roadmap in one important way: runtime policy remains a cross-cutting extension surface, but it is no longer the next unresolved blocker. Active execution should now move into the domains that either build directly on the new authority layer or deliver the highest-value standalone-client parity you prioritized.

`Watchlists` plus `Client Notifications` are also now partially landed for source CRUD and the first server control-plane slice. See [watchlists-notifications-tranche-2.md](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Docs/Development/watchlists-notifications-tranche-2.md) for the verification record and the precise deferred scope.

At that point, the next unresolved Tranche 2 focus moved to `Local MCP Runtime`, with `Writing Suite` and `Research Sessions / Runs` following after that.

`Local MCP Runtime` now has a ninth real control-plane increment. Chatbook’s `Unified MCP` destination exposes source-explicit local overview, inventory, external-profile management, local governance browse with save/delete plus exact-capability preview, and a local `Advanced` section with direct runtime status plus direct request/batch helpers over the in-process runtime delegate. Local runtime operations now resolve to concrete local action ids for effective-governance preview, deny rules can block runtime calls, batch helpers return per-item governance annotations, local `ask` governance persists approval requests with list/approve/deny/filter/delete actions plus runtime-effective reuse of approved or denied decisions, recent runtime activity persists across service recreation with blocked/success outcomes and governance metadata, protocol diagnostics expose protocol version, transport, supported methods, manifest counts, and direct implementation coverage, and runtime health exposes ready/degraded state, initialization timestamp, uptime, manifest load counts, component cache state, and issues. The remaining Local MCP work is richer lifecycle controls, telemetry retention/filtering depth, deeper approval policy semantics, and broader operator UX rather than the absence of a local shell.

`Collections: Reading List / Read-it-later` is now also landed for the current server contract. The authoritative capability seam, runtime normalization, and search-panel affordance now make the boundary explicit: local saved browsing remains per-media-type, while server saved browsing is aggregate-only in `All Media`. Any future per-media-type server saved view remains blocked on a server contract extension, so the next larger non-MCP row should now move to `Writing Suite`.

`Research Sessions / Runs` now has mounted detail depth in addition to its source-separated surface. Chatbook has typed research-runs client methods, local run/artifact persistence, local/server scope routing, typed server SSE event streaming, a selected-run `Watch Events` action, mounted bundle and artifact loading, checkpoint approve controls, a persistent event log in the Research screen, and app navigation wiring. Remaining Research work is no longer missing mounted observe/control seams; it is a real local autonomous research execution engine plus further UX polish.

`Prompts / Chatbooks` now has its first contract seam plus source-routed prompt CRUD in the existing CCP prompt editor. Chatbook has typed server prompt CRUD methods, typed server chatbook job list/status/cancel/remove/download methods, a plain-dict-preserving server chatbook service, live remote server job browsing plus cancel/download/remove controls in the export management window, a normalized local/server prompt scope service, source-aware server prompt usage/version/restore routing, app-level prompt scope wiring, CCP prompt list/load/create/update/delete routing through the active runtime source, and mounted CCP controls for prompt usage, server version listing, and server version restore. Remaining work is mostly prompt collections/workflows, chatbook cleanup/continuation affordances, and deeper import/export identity alignment.

`Web Clipper` now has a first remote-only server slice. Chatbook includes typed save/status/enrichment client methods, a remote-only service, policy-aware scope routing, app bootstrap wiring, and a lightweight media-ingest tab for clip save, known-clip status lookup, and enrichment persistence. Remaining work is browser-extension handoff UX, server clip browse/history if the server exposes it, richer capture helpers, and any future local mirror/import design.

`Sharing` now has a first remote-only server slice. Chatbook includes typed non-admin sharing client methods, a remote-only service, policy-aware scope routing, app bootstrap wiring, and a lightweight `Sharing` panel in `Tools & Settings` for workspace share permissions, shared-with-me discovery, clone/chat proxy actions, and share-token operations. Remaining work is richer shared-resource rendering, deeper public-share UX, and any future local import/sync design.

`Collections: Outputs / Templates / Artifacts` now has a first dedicated server slice. Chatbook includes typed output-template and output-artifact client methods, a remote-only server outputs service, a policy-aware scope seam, app bootstrap wiring, and a lightweight `Outputs` panel in `Tools & Settings` for template CRUD/preview plus artifact browse/get/create/update/delete in explicit server mode. Remaining work is richer download/export ergonomics, deeper render-job observation/history, and any later optional local parity or sync design.

`Chat` now has backend-only source-aware conversation and chat-loop seams. Chatbook wires local conversation metadata/history/delete, server conversation list/detail/update/tree operations, and server chat-loop start/event/approve/reject/cancel operations behind source-scoped policy actions, preserving local workspace scope filtering and avoiding any chat UI rewrite. Remaining Chat work is mounted remote launch/message handoff, remote delete pending a server conversation-delete contract, attachment/export identity, and mounted UX for explicit local/server conversation browsing.

## Immediate Post-Tranche-0 Order

1. `Local MCP Runtime`
   Next active Tranche 2 focus. The runtime-policy layer is now in place, and Chatbook already has local MCP modules; the next gap is turning that into a first-class local runtime, approvals, status, and governance surface.
2. `Writing Suite`
   This is now the next larger non-MCP row. The read-it-later tranche is landed for the current server contract, and Chatbook still lacks a serious standalone project and manuscript hierarchy.
3. `Research Sessions / Runs`
   Source-separated CRUD/control, live server event streaming, and mounted bundle/artifact/checkpoint/event-log depth are landed. Continue this row only when adding a real local research execution engine or further UX polish.
4. `Media / Reading / Ingestion Sources`
   The approved `Read-it-later` follow-up is landed for the current server contract, and Chatbook now has typed server ingest-job submit/status/list/cancel/event-stream seams, typed server web-content ingest helpers with first-class server ingest controls, local/server reading-highlight CRUD seams with first-class viewer authoring controls, server reading saved-search CRUD, server reading-item note-link CRUD, server reading import submit/list/detail, server reading archive creation, server reading export bytes, server reading summarize responses, server reading TTS bytes, ingestion-source item reattach, a server job tab with last-batch refresh/cancel, selected-job cancel, selected-batch live watch, recent visible server-job live watch, known-batch lookup by ID, and a remote-only Web Clipper tab for save/status/enrichment. Remaining work is true historical batch discovery, which still needs a server-side batch-discovery or recent-batches contract, server clip browse/history if desired, digest scheduling when workflows are back in scope, plus any future server contract change for per-media-type saved views.
5. `Prompts / Chatbooks`
   First contract seam, active-source prompt CRUD routing in CCP, live remote job browsing/actions, export download, and mounted prompt usage/version controls are landed. Continue this row when exposing prompt collections/workflows or chatbook cleanup/continuation controls.
6. `Watchlists` plus `Client Notifications`
   Partially landed. Chatbook now has a source-aware subscriptions shell, remote watchlist source CRUD/restore, server jobs/runs/alert-rule administration, server reminders/feed controls, server-only control-plane tabs with explicit local guidance, and a persisted local notifications inbox. Remaining work is groups, richer structured control-plane UX, richer run output/log/artifact handling, and richer reminder/feed presentation.
7. `Workflows`, `Scheduler Workflows`, and `Chat Workflows`
   These remain remote-only acceptable, but once the local-first rows above are in place, Chatbook should add discover/trigger/observe support for connected-server workflow surfaces.
8. `Server Reminders / Notification Feeds`, `Sharing`, and `Web Clipper`
   Server reminders/feed, Sharing, and Web Clipper now have first remote-only slices. Continue them only for richer UX, deeper shared-resource rendering, browser handoff, stream ergonomics, or any server contract extensions they depend on.

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

- Crosswalk `Watchlists` onto existing local subscriptions and notification plumbing before inventing a separate remote-first model.
- Treat the first-slice watchlists source CRUD, restore, jobs, runs, alert rules, and client-notifications inbox as landed foundations; keep watchlist groups, richer structured job/rule editors, richer run output/log/artifact UX, and server reminder or feed surfaces as later follow-on work.
- Treat `Collections: Reading List / Read-it-later` as landed for the current server contract. Keep any future per-media-type server saved view follow-on explicitly blocked on a server list-contract extension, and keep any sync/mirror contract out of scope until separately approved.
- Treat `Writing Suite` and first-slice `Research Sessions / Runs` as landed foundations; remaining work should deepen UX and execution semantics rather than re-litigating source separation.
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

- Keep remote-only workflows, sharing, web clipper, and remote MCP governance behind the core standalone-first roadmap rather than letting them pull priority forward.
- Treat first-slice server reminders/feed, Sharing, and Web Clipper as landed remote-only foundations; continue them only for richer UX, stream ergonomics, deeper shared-resource rendering, clip browsing/history, browser handoff, or explicitly approved local ingestion/sync semantics.
- Treat `Study Packs` and `Study Suggestions` as having first-slice remote client/service/scope seams; continue them only for mounted UX, richer job/snapshot presentation, or explicitly approved local materialization/import semantics. Keep `Research Search / Provider Surfaces` contract-maturity-sensitive until clearer evidence exists.
- Reuse the source-labeling, offline fallback, and discover/trigger/observe patterns proven in earlier tranches instead of creating special-case remote UI rules.

## Follow-On Vertical Plans

- `Collections: Reading List / Read-it-later parity`: Landed for the current server contract. Reopen this row only if the server adds per-media-type saved browsing support or a separate sync/mirror design is approved.
- `Watchlists / subscriptions alignment`: Source CRUD, restore, server jobs/runs, alert rules, and notification delivery now map onto the server watchlist vocabulary without collapsing local ownership. Continue only for groups, richer job/run/rule UX, server reminder/feed, or sync design.
- `Server reminders / notification feeds`: First-slice typed client/service/scope support and lightweight subscriptions-window tabs are landed. Continue only for richer feed/reminder UX, preferences editing, stream worker ergonomics, or later explicit local-notification ingestion.
- `Sharing`: First-slice typed client/service/scope support and a lightweight Tools & Settings panel are landed for workspace share permissions, shared-with-me discovery, clone/chat proxy actions, and share-token operations. Continue only for richer shared-resource browsing, deeper public-share UX, or later explicit local import/sync design.
- `Web Clipper`: First-slice typed client/service/scope support and a lightweight media-ingest tab are landed for server save, known-ID status lookup, and enrichment persistence. Continue only for browser-extension handoff UX, server clip browse/history, richer capture helpers, or later explicit local import/sync design.
- `Local MCP runtime parity`: A ninth local control-plane slice is landed with source-explicit local overview, inventory, external profiles, governance-rule management plus preview, a local advanced/runtime surface for direct status plus request/batch helpers, effective-governance resolution/enforcement for local runtime operations, persisted local approval requests with list/approve/deny/filter/delete control-plane actions, persisted recent runtime activity observation with blocked/success outcomes, protocol diagnostics for method and implementation coverage, and runtime health for ready/degraded state, initialization timestamp, uptime, manifest load counts, component cache state, and issues. Continue this row with richer lifecycle controls, telemetry retention/filtering depth, deeper approval policy semantics, and later operator UX rather than reopening the basic local shell.
- `Remote MCP control plane`: The current remote Unified MCP hub-management route surface is now effectively covered inside `Tools & Settings`, with explicit local/server panes, configured server targets, remote browse sections, governance actions, governance-pack source/import/upgrade flows, expanded advanced admin browse/action coverage, assignment workspace membership control, credential-binding administration, slot-status views, and the top-level external-server secret setter. The next follow-on should focus on UI polish, richer structured presentation, and any future server-side MCP surface expansion without collapsing the source-separated local MCP runtime model.
- `Writing suite parity`: Plan the local-first project, manuscript, chapter, and scene hierarchy together with the server contract and source-separated UI behavior.
- `Research sessions parity`: Local/server run CRUD, control, selected-run live server event observation, and mounted bundle/artifact/checkpoint/event-log depth are landed. Follow-on work should focus on a local autonomous research execution engine and broader UX polish.
- `Remote workflows surface`: Split general workflows, scheduler workflows, and chat workflows into a dedicated remote-only plan that covers discovery, scheduler configuration and scheduling control-plane behavior, launch, run status, and observation after the core parity rows land.
- `Study packs / study suggestions once contract maturity is confirmed`: Hold the vertical plan until the server-side contract is stable enough to justify a focused client surface.
