# Chatbook Server Target-State Design

Audit date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

## Operating Rules

- `Separated Local/Server primary UI`: For any domain with both local and server capability, Chatbook treats `Local` and `Server` as separate operating modes. The primary screens, lists, detail views, and actions must stay source-labeled and source-scoped rather than mixing records by default.
- `Remote-only discoverability`: Domains that are intentionally server-owned must still appear in Chatbook when a server is configured. The client must be able to discover, inspect, launch, or observe them from server mode even when no local equivalent exists.
- `Offline fallback behavior`: Loss of server connectivity must not break local-first domains. Local domains continue to function against local storage and local runtime services; remote-only domains show an explicit unavailable-offline state instead of implying cached authority.
- `Local write authority before sync`: Until a separate sync design exists, local mode writes only to local state and server mode writes only to server state. Chatbook does not silently dual-write, cross-promote, or treat server data as the source of truth for local edits.
- `Future mirror relevance without sync design`: Some dual-backend domains are likely future mirror candidates, but this document only records whether later mirroring might matter. It does not define identity mapping, conflict resolution, import rules, or mixed-view behavior.

## Core Dual-Backend Domains

This section covers domains that operate in explicit local and server modes, including source-separated domains where local and remote surfaces both exist now even if full local parity is not required for every subdomain.

### Chat
Authored now: local conversations are authored in local mode and server conversations are authored in server mode. Viewed now: conversation lists and detail views are selected from the active source, not merged. Local: Chatbook must preserve full local conversation CRUD and chat launch without server dependency. Remote: Chatbook must support server conversation CRUD and launch when connected. Offline: local chat remains available, while server chat becomes unavailable without inventing cached server write authority. Mixed view: deferred.

### Characters / Personas / CCP
Authored now: characters, personas, sessions, and related messages are authored separately per source. Viewed now: users browse either the local catalog or the server catalog based on mode. Local: Chatbook must keep full local character and persona management plus launch flows. Remote: Chatbook must support connected-server character, persona, session, and message operations without collapsing them into the local store. Offline: local CCP flows continue; server CCP flows are unavailable. Mixed view: deferred.

### Notes / Workspaces
Authored now: local notes and workspaces are authored locally, while server notes and workspaces are authored on the server. Viewed now: note lists, workspace boundaries, and graph views come from the active source only. Local: Chatbook must retain full local note, workspace, and graph behavior. Remote: Chatbook must support server note, workspace, and graph-aware CRUD in server mode. Offline: local notes remain fully usable; server notes become read/write unavailable. Mixed view: deferred.

### Media / Reading / Ingestion Sources
Authored now: local media, reading progress, and ingestion sources are authored locally unless the user is in server mode. Viewed now: media libraries and reading state are shown from one source at a time. Local: Chatbook must keep local media CRUD, reading progress, ingestion sources, and local ingest execution. Remote: Chatbook must support server reading, item, source, and ingest-job flows when connected. Offline: local media and reading continue; server-backed jobs and server libraries are unavailable. Mixed view: deferred.

### Prompts / Chatbooks
Authored now: prompts and chatbooks are authored locally in local mode and on the server in server mode. Viewed now: prompt libraries, chatbook packages, and import/export jobs are source-specific. Local: Chatbook must preserve standalone prompt authoring plus local chatbook import and export. Remote: Chatbook must support server prompt lifecycle and server chatbook jobs when connected. Offline: local authoring and packaging continue; server prompt and chatbook operations are unavailable. Mixed view: deferred.

### Study Core
Authored now: decks, cards, quizzes, and study-guide artifacts are authored against the selected source. Viewed now: study libraries and results are shown from local or server mode, not a combined catalog. Local: Chatbook must keep standalone local flashcard, quiz, and study-guide workflows. Remote: Chatbook must support server study CRUD and execution flows when connected. Offline: local study remains available; server study operations are unavailable. Mixed view: deferred.

### Evaluations
Authored now: datasets, runs, and results are authored locally in local mode or against the server in server mode. Viewed now: evaluation history and detail screens stay source-scoped. Local: Chatbook must preserve standalone local evaluation datasets, runs, and results. Remote: Chatbook must support server evaluation CRUD and run history when connected. Offline: local evaluations continue; server evaluation views become unavailable for mutation and refresh. Mixed view: deferred.

### RAG / Embeddings / Chunking Admin
Authored now: chunking templates, embedding settings, and reprocess actions are authored against the active backend. Viewed now: admin controls and status are shown from the selected source only. Local: Chatbook must retain local retrieval-admin control over chunking, embeddings, and reprocess helpers. Remote: Chatbook must support server admin controls when connected. Offline: local admin remains available; server admin functions are unavailable. Mixed view: deferred.

## Local-First Domains With Remote Interop

### Collections: Reading List / Read-it-later
Authored now: saved-reading state is authored locally first, with a separate server collection when connected in server mode. Viewed now: the reading collection is shown from the chosen source, not as a merged inbox. Local: Chatbook must provide full local saved-reading and read-it-later CRUD as a standalone collection surface. Remote: Chatbook must support server reading-list flows when connected. Offline: local read-it-later remains fully usable. Mixed view: deferred.

### Watchlists
Authored now: local watchlist-equivalent state is authored through local subscriptions and notifications, while server watchlists are authored only in server mode. Viewed now: users inspect either local monitoring state or server watchlists based on mode. Local: Chatbook must provide practical local watchlist parity through local subscriptions, rule management, and local notification delivery. Remote: Chatbook must support server watchlists, sources, runs, and alert rules when connected. Offline: local monitoring continues, but server watchlists and alert rules are unavailable. Mixed view: deferred.

### Writing Suite
Authored now: writing projects and manuscript hierarchy are authored locally in local mode, and server manuscripts are authored only in server mode. Viewed now: project and manuscript views remain source-scoped. Local: Chatbook now implements the v1 structural-authoring service/controller target with local projects, manuscripts, unassigned chapters, chapters, scenes, Markdown scene drafts, manual versions, soft-delete, trash restore, reorder, move, and source-specific search. Remote: Chatbook supports the current server manuscript contract for project, part-as-manuscript, chapter, scene, structure, search, reorder, and soft-delete operations without treating server manuscripts as local records. Mounted UI status: source switch, project browse, outline/detail selection, project create, selected save/delete, local version create/restore, and unsupported server reason state are present; child create, search, reorder/move, and trash restore controls remain pending the follow-on UX pass. Offline: local writing stays available without degradation. Mixed view: deferred. Future rows: generation, export/publishing, collaboration, sync/mirroring, server direct manuscript-level scenes, server version history, server trash restore, and server scene reparenting.

### Research Sessions / Runs
Authored now: research sessions and runs are authored locally in local mode, and server research sessions and runs are authored only in server mode. Viewed now: session lists, run status, and bundles are viewed from one source at a time. Local: Chatbook must add a standalone local research-session lifecycle with local status and result handling and keep local research state locally owned. Remote: Chatbook must support server research sessions, streaming, and bundle retrieval when connected without collapsing them into the local research store. Offline: local research remains available; server research runs are unavailable. Mixed view: deferred.

### Client Notifications
Authored now: notification state for local operations is authored and owned by Chatbook locally. Viewed now: notification delivery and status are viewed from Chatbook's local notification surface rather than a server queue. Local: Chatbook must keep local notification generation, queueing, and delivery for local watchlists, research, and other local operations. Remote: Chatbook may surface server-originated events separately, but it must not replace local notification ownership with the server feed model. Offline: local notifications continue because they are client-owned. Mixed view: deferred.

### Local MCP Runtime
Authored now: MCP runtime configuration, approvals, prompts, resources, and execution state are authored locally inside Chatbook. Viewed now: the local MCP catalog and runtime status are viewed as a Chatbook-owned local surface. Local: Chatbook must provide local MCP runtime execution, local approvals, and local governance for offline-capable operations. Remote: Chatbook may later import or reference remote catalogs, but remote governance remains a different surface and must not subsume local control. Offline: local MCP continues to operate without server dependency. Mixed view: deferred.

## Remote-First / Conditional Domains

This section covers domains where the current parity target is primarily remote, while any local-adjacent behavior remains optional, separate, or too low-confidence to treat as a settled local-first domain.

### Collections: Outputs / Templates / Artifacts
Authored now: managed outputs, templates, and render/export jobs are authored on the server in server mode, while adjacent local workspace artifacts remain separately local where they exist. Viewed now: server-managed outputs and local-adjacent artifacts stay source-scoped rather than appearing as one merged artifact catalog. Local: Chatbook may continue to expose adjacent local workspace artifacts, but full local managed-outputs parity is optional and not the current target state. Remote: Chatbook must support server outputs, templates, and render jobs when connected. Offline: server-managed outputs are unavailable, while adjacent local artifacts can remain visible. Mixed view: deferred.

### Research Search / Provider Surfaces
Authored now: the server-side research-provider contract is authored on the server in server mode, while local search settings and provider tools remain separate local-adjacent behavior. Viewed now: local search/provider tools and server provider-backed research actions are shown from the selected source rather than treated as one unified provider surface. Local: Chatbook may keep existing local search and provider tools separate until a clearer parity target exists; local parity is not yet a settled commitment here. Remote: Chatbook must discover, configure, trigger, and observe server provider surfaces when connected as that contract stabilizes. Offline: local search can continue, but server provider surfaces are unavailable. Mixed view: deferred. Confidence note: the server provider contract remains lower-confidence than the core dual-backend and strong local-first domains.

## Remote-Only Domains

This section covers server-owned domains with no current local authoring target.

### Workflows
Authored now: general workflow definitions and runs are authored on the server only. Viewed now: workflow lists, launch controls, and run status are viewed from server mode only. Local: Chatbook does not implement a local workflow engine for this audit. Remote: Chatbook must discover, launch, and observe general workflows when connected. Offline: workflows show explicit unavailable-offline UI. Mixed view: deferred because there is no local workflow source.

### Scheduler Workflows
Authored now: scheduler-workflow definitions, schedules, and orchestration control-plane state are authored on the server only. Viewed now: scheduler workflow lists, schedule configuration, launch controls, and run status are viewed from server mode only. Local: Chatbook does not implement a local scheduler-workflow engine or local scheduling control plane for this audit. Remote: Chatbook must discover, configure, launch, and observe scheduler workflows when connected, preserving their additional scheduling and control-plane scope. Offline: scheduler workflows show explicit unavailable-offline UI. Mixed view: deferred because there is no local scheduler-workflow source.

### Chat Workflows
Authored now: chat-workflow definitions and orchestration state are authored on the server only. Viewed now: chat-workflow discovery, launch controls, and status are viewed from server mode only. Local: Chatbook keeps ordinary local chat separate and does not implement a local chat-workflow engine for this audit. Remote: Chatbook must discover, launch, and observe server chat workflows when connected. Offline: chat workflows show explicit unavailable-offline UI. Mixed view: deferred because there is no local chat-workflow source.

### Server Reminders / Notification Feeds
Authored now: reminders, server tasks, and notification-feed state are authored on the server only. Viewed now: these feeds are viewed from server mode and must stay distinct from Chatbook's local notifications. Local: Chatbook does not treat local notifications as a local reminder mirror. Remote: Chatbook must discover, configure, trigger, and observe server reminders and notification feeds when connected. Offline: the reminder/feed surface is explicitly unavailable. Mixed view: deferred.

### Study Packs
Authored now: study-pack jobs and generated packs are authored on the server only. Viewed now: pack status and generated pack artifacts are viewed from server mode. Local: Chatbook falls back to existing local study-core deck, flashcard, and review flows rather than inventing local copies of this server-first feature. Remote: Chatbook must discover, launch, and observe study-pack surfaces when connected. Offline: study packs are unavailable, with fallback to local study-core workflows only. Mixed view: deferred.

### Study Suggestions
Authored now: suggestion anchors, snapshots, actions, and refresh jobs are authored on the server only. Viewed now: suggestion state and suggestion actions are viewed from server mode. Local: Chatbook falls back to existing local study-core review and next-review flows rather than inventing local copies of this server-first feature. Remote: Chatbook must discover, launch, and observe study-suggestion surfaces when connected. Offline: study suggestions are unavailable, with fallback to local study-core workflows only. Mixed view: deferred.

### Sharing
Authored now: share links, permissions, and revocation state are authored on the server only. Viewed now: sharing is discovered and managed from server mode only. Local: Chatbook does not define a local sharing system for this audit. Remote: Chatbook must support share discovery, creation, inspection, and revocation when connected. Offline: sharing is unavailable. Mixed view: deferred.

### Web Clipper
Authored now: clip save requests, status, and enrichment state are authored on the server only. Viewed now: clip capture and status are viewed from server mode only. Local: Chatbook does not implement a local clipper surface for this audit. Remote: Chatbook must trigger remote clip capture and observe clip status when connected. Offline: web clipper actions are unavailable. Mixed view: deferred.

### Remote MCP Control Plane / Governance
Authored now: MCP hub policy, catalogs, approvals, and external-server governance are authored on the server only. Viewed now: remote governance surfaces are viewed from server mode only and remain separate from local MCP runtime ownership. Local: Chatbook does not mirror remote governance into the local MCP store. Remote: Chatbook must discover, configure, approve, and observe remote MCP governance when connected. Offline: remote governance is unavailable, while local MCP runtime remains usable. Mixed view: deferred.

## Cross-Cutting Runtime Policy

Authored now: local mode selection, local config, and source labels are authored in Chatbook, while auth, sessions, feature flags, and rate-limit policy are authored on the server. Viewed now: users should always know whether they are acting locally or against a server, and server policy should only appear when server mode is active. Local: Chatbook must preserve a usable standalone runtime with explicit source labels, local config, local notifications, and local MCP/runtime policy that do not require server auth. Remote: Chatbook must honor server auth, feature, session, tenancy, and policy constraints when connected to a multi-user server. Offline: Chatbook remains operational for local domains and clearly marks server policy as inactive because no server authority is reachable. Mixed view: deferred until a future mirror design defines identity, conflict, and presentation rules across local and server sources.
