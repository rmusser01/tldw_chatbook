# Chatbook Required Server API Contract

Date: 2026-04-29

Status: design contract

Related repositories:

- `tldw_chatbook`
- `tldw_server2`

Related Chatbook docs:

- `Docs/Parity/domain-edge-contracts/chat.md`
- `Docs/Parity/domain-edge-contracts/media-reading.md`
- `Docs/Parity/domain-edge-contracts/notes-workspaces.md`
- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`

## Goal

Define the server API surface Chatbook needs in order to close the accepted non-sync domain deferrals while remaining a standalone local-first client.

This contract does not require server parity for billing, admin-only ops, or broad UI/UX work. It also does not move local-only Chatbook responsibilities into the server.

## Baseline Finding

The server already exposes broad route coverage for most domains:

- Chat conversation list/detail/update/tree/share and chat-loop run/event/control routes exist.
- Reading, saved searches, reading item updates, reading imports, and reading TTS routes exist.
- Notes, notes graph, workspaces, workspace notes, workspace sources, and workspace artifacts routes exist.
- Watchlists have source/group/job/run/item/output/template routes.
- Workflows and workflow scheduler routes exist.
- Research run, event stream, and bundle routes exist.
- Writing/manuscript structural routes exist.
- RAG, media embeddings, evaluations, flashcards, quizzes, audio jobs, audio streaming, voice assistant, notifications, web clipper, sharing, and Unified MCP routes exist.

The remaining gap is therefore not whole-domain route creation. The gap is a smaller set of missing first-class operations plus stable contracts that Chatbook can use for action gating, identity handling, and source-honest unsupported behavior.

## API Contract Principles

1. Server APIs must be scoped to the authenticated user and expose only the org/team/workspace/user resources that principal can manage.
2. Every Chatbook-visible server action must be represented in an authenticated capability response before the UI or services enable it.
3. Unsupported operations must be explicit. Chatbook must not infer support from route existence alone.
4. Server event streams must provide stable event IDs, stream names, cursors, replay windows, and dedupe keys.
5. Server IDs must remain server IDs. Chatbook may normalize them for display, but must not write server-owned records into local tables as authoritative records unless a later sync/import contract explicitly says so.
6. Moves must preserve identity, audit history, permissions, version/ETag semantics, and rollback behavior. Chatbook must not emulate moves with copy/delete.
7. Remote-only domains stay active-server-owned. Local parity domains remain Chatbook-local unless this contract explicitly requires a server API.

## Priority 0: Capability Catalog

Expose one authenticated capability catalog.

Recommended endpoint:

- `GET /api/v1/capabilities`

Minimum response fields:

- `server_id` or stable deployment/profile identifier.
- `contract_version`.
- `generated_at`.
- `principal`: user, tenant, org/team memberships if safe to expose.
- `operations[]`.

Each operation entry must include:

- `operation_id`: stable action ID, for example `chat.conversation.create.server`.
- `domain`: `chat`, `media`, `notes`, `workspaces`, `workflows`, `mcp`, etc.
- `source`: `server`.
- `scope_types`: supported scopes, for example `global`, `workspace`, `org`, `team`.
- `route`: canonical route if supported.
- `methods`: allowed HTTP methods.
- `required_permissions`: permission strings or policy IDs.
- `supported`: boolean.
- `unsupported_reason`: `server_contract_missing`, `scope_not_supported`, `permission_denied`, `feature_disabled`, `contract_deferred`, or null.
- `stability`: `stable`, `preview`, or `internal`.
- `resource_constraints`: optional limits, quotas, or feature flags.

Rules:

- This endpoint is the server-side source of truth for Chatbook action availability.
- Chatbook's existing active-server capability service remains the local projection/cache for the selected server. It must consume this endpoint and active-server context; it must not become a second independent authority with separately invented support rules.
- It may be backed by existing route metadata, permission catalogs, and feature flags, but must present one stable Chatbook-facing shape.
- It must not expose admin/billing/ops-only operations unless the authenticated principal can manage them and Chatbook has a destination for them.
- Route presence without a capability entry is not enough for Chatbook to enable an action.
- The response must be scoped by active authenticated principal and should include concrete manageable scope instances, not only scope type names.

Scope instance entries must include:

- `scope_type`: `user`, `workspace`, `team`, `org`, or another stable server scope.
- `scope_id`: concrete server ID for the scope.
- `display_name`: safe label for Chatbook panes.
- `parent_scope`: optional parent reference for team/org/workspace nesting.
- `operations`: operation IDs available in that concrete scope.
- `default_selected`: optional hint for initial Chatbook selection.

Security and redaction rules:

- `permission_denied` operations may be represented as unavailable only when product UX requires discoverability. Otherwise they should be omitted.
- Internal route names, policy IDs, and permission strings must be redacted unless the authenticated principal is allowed to inspect them.
- Billing, ops, and unrelated admin capabilities must not appear solely because the principal has broad server admin access; Chatbook must also have a destination and product need for the operation.
- Feature-disabled operations should expose stable reason codes without leaking deployment internals.

## Priority 1: Missing First-Class APIs

### Chat Conversations

Current gap: server conversation list/detail/update/tree/share exist, but first-class create/delete are not exposed as a stable Chatbook contract outside launch/persist flows.

Expose:

- `POST /api/v1/chat/conversations`
- `DELETE /api/v1/chat/conversations/{conversation_id}`

Create request requirements:

- `title`.
- `scope_type`: `global` or `workspace`.
- `workspace_id` when `scope_type=workspace`.
- Optional assistant identity fields: character, persona, agent, or model source.
- Optional metadata fields already returned by conversation list/detail.
- Optional initial messages only if server can persist them atomically.

Delete requirements:

- Must support soft-delete or tombstone semantics consistent with existing list filters.
- Must require scope validation.
- Must not allow workspace conversations to be deleted through unscoped global calls.
- Must return conflict/version errors when applicable.

Capability entries:

- `chat.conversation.create.server`
- `chat.conversation.delete.server`

### Chat Loop Persistence Attachment

Current gap: server chat-loop routes expose run start/events/approve/reject/cancel, but Chatbook needs a stable rule for how a run becomes, or attaches to, a server conversation.

Extend existing chat-loop start/run contract with:

- Optional `conversation_id` for attach.
- Optional `create_conversation` payload.
- Optional `idempotency_key` for attach/create/persist retries.
- Response field `conversation_ref` when a persisted server conversation exists.
- Event field `conversation_ref` when persistence happens asynchronously.

Minimum `conversation_ref` fields:

- `conversation_id`.
- `scope_type`.
- `workspace_id`.
- `state`.
- `version` or `etag` if available.

Rules:

- A `run_id` is not a conversation ID.
- Chatbook must not create a local conversation from server loop events.
- If the server cannot persist or attach a run, the capability catalog must report that explicitly.
- Retrying attach/create/persist with the same `idempotency_key` must return the same `conversation_ref` or a stable terminal error.
- Attach must be atomic with respect to the target conversation version/ETag when one is supplied.

Capability entries:

- `chat.loop.start.server`
- `chat.loop.attach_conversation.server`
- `chat.loop.persist_conversation.server`

### Ingestion Sources Delete

Current gap: server ingestion sources expose create/list/get/patch/sync/archive/reattach, but no first-class delete route.

Expose:

- `DELETE /api/v1/ingestion-sources/{source_id}`

Delete request/response requirements:

- Must support soft-delete or tombstone semantics.
- Must define whether managed source items are detached, deleted, preserved, or archived.
- Must refuse deletion during active sync unless `force=true` is explicitly supported.
- Must return enough data for Chatbook to update local presentation state without guessing.
- Repeated delete calls must be idempotent or return a stable terminal `not_found_or_deleted` style response that Chatbook can treat as complete.

Capability entry:

- `media.ingestion_sources.delete.server`

## Priority 2: Contract Stabilization For Existing APIs

These domains appear to have existing server route coverage. The required work is to stabilize schemas, events, artifact references, and capability entries so Chatbook can enable them safely.

### Notifications

Existing server routes cover notification list, unread count, mark-read, dismiss, snooze, preferences, and SSE streaming.

Stabilize:

- Stream name.
- `event_id` monotonicity and uniqueness.
- Cursor resume rules.
- Replay window and retention behavior.
- Dedupe key fields.
- Behavior after active server switch.

Capability entries:

- `notifications.list.server`
- `notifications.stream.server`
- `notifications.mark_read.server`
- `notifications.dismiss.server`
- `notifications.snooze.server`
- `notifications.preferences.update.server`

### Workflows And Scheduler

Existing server routes cover workflow definition CRUD/versioning, run lifecycle, events, WebSocket, templates/options, artifacts/investigation, and workflow schedules.

Stabilize:

- Definition schema version and validation errors.
- Run status lifecycle.
- Event cursor and replay semantics.
- Artifact list/detail/download shape.
- Cancel behavior and terminal-state idempotency.
- Schedule ownership and admin-owner filter behavior.

Capability entries:

- `workflows.definition.create.server`
- `workflows.definition.update.server`
- `workflows.definition.delete.server`
- `workflows.run.start.server`
- `workflows.run.cancel.server`
- `workflows.run.events.server`
- `workflows.run.artifacts.server`
- `workflows.schedule.create.server`
- `workflows.schedule.update.server`
- `workflows.schedule.delete.server`

### Watchlists

Existing server routes cover sources, groups, jobs, runs, stream, items, outputs, templates, and import/export.

Chatbook currently treats local watchlists as local parity and remote watchlists as active-server-owned. If group editing remains deferred in Chatbook, the server only needs capability entries, not new routes.

Stabilize:

- Source/job/run identity fields.
- Run stream event IDs.
- Output artifact metadata/download shape.
- Group read/edit capabilities as separate actions.

Capability entries:

- `watchlists.sources.*.server`
- `watchlists.groups.read.server`
- `watchlists.groups.write.server`
- `watchlists.jobs.*.server`
- `watchlists.runs.*.server`
- `watchlists.outputs.*.server`

### Research

Existing server routes cover research run create/list/get, event stream, pause/resume/cancel, bundle, and artifacts.

Stabilize:

- Run lifecycle.
- Event cursor semantics.
- Bundle schema version.
- Artifact list/detail/download contract.
- Permission behavior for multi-user servers.

Capability entries:

- `research.runs.*.server`
- `research.events.stream.server`
- `research.bundle.read.server`
- `research.artifacts.*.server`

### Writing Suite

Existing server routes cover writing projects, manuscripts, structural entities, versions, and analysis-style helpers.

Stabilize:

- Project/manuscript/chapter/scene identity model.
- Server unassigned chapter handling.
- Draft semantics: scene body content vs manuscript/chapter metadata and ordered membership.
- Version creation vs autosave/update semantics.
- Export/download artifact shape.

Capability entries:

- `writing.projects.*.server`
- `writing.manuscripts.*.server`
- `writing.chapters.*.server`
- `writing.scenes.*.server`
- `writing.versions.*.server`
- `writing.exports.*.server`

### Study And Evaluations

Existing server routes cover flashcards, quizzes, study suggestions, evaluation runs/history, and related artifacts.

Stabilize:

- Target catalog discovery if Chatbook should choose server evaluation targets.
- Result artifact list/detail/download shape.
- Study pack job lifecycle.
- Source object references for generated cards/quizzes.

Recommended endpoints if not already canonical:

- `GET /api/v1/evaluations/targets`
- `GET /api/v1/evaluations/history/{run_id}/artifacts`
- `GET /api/v1/evaluations/history/{run_id}/artifacts/{artifact_id}/download`

Capability entries:

- `evaluations.targets.list.server`
- `evaluations.runs.*.server`
- `evaluations.artifacts.*.server`
- `study.flashcards.*.server`
- `study.quizzes.*.server`
- `study.suggestions.*.server`

### RAG And Embeddings

Existing server routes cover unified RAG and media embeddings.

Stabilize:

- Collection identity and export contract.
- Export job lifecycle and download route.
- Per-media embedding admin operations if Chatbook should expose them.
- Model/provider catalog for embedding operations.

Recommended endpoints if not already canonical:

- `POST /api/v1/rag/collections/{collection_id}/exports`
- `GET /api/v1/rag/collections/{collection_id}/exports/{export_id}`
- `GET /api/v1/rag/collections/{collection_id}/exports/{export_id}/download`

Capability entries:

- `rag.collections.export.server`
- `rag.collections.search.server`
- `rag.embeddings.media.rebuild.server`
- `rag.embeddings.media.delete.server`
- `rag.embedding_models.list.server`

### Audio And Voice

Existing server routes cover audio jobs, TTS, STT, streaming, voice assistant, and artifacts.

Stabilize:

- WebSocket/session discovery and auth requirements.
- Speech job history and artifact metadata.
- Job event stream cursor/dedupe rules.
- Voice assistant session lifecycle.
- Artifact retention and deletion.

Capability entries:

- `audio.jobs.*.server`
- `audio.tts.*.server`
- `audio.stt.*.server`
- `audio.voice.sessions.*.server`
- `audio.artifacts.*.server`

### Unified MCP

Existing server routes cover Unified MCP, hub management, catalogs, and catalog management.

Stabilize all authenticated scopes Chatbook is allowed to manage:

- User-local server-side MCP settings.
- Workspace/team/org catalogs.
- External server registry.
- Tool inventory and tool metadata.
- Approval queues.
- Governance policies.
- Audit/history.
- Scope switching metadata.

Capability entries:

- `mcp.registry.*.server`
- `mcp.catalogs.*.server`
- `mcp.tools.*.server`
- `mcp.approvals.*.server`
- `mcp.governance.*.server`
- `mcp.audit.*.server`

## Priority 3: Decision-Gated Domain Deferral Closure APIs

These APIs close known domain deferrals. They are not generic nice-to-haves. Each row must be explicitly included or explicitly deferred before implementation so Chatbook does not keep ambiguous placeholders.

### Per-Media-Type Read-It-Later Server Browsing

Current Chatbook contract treats server read-it-later as aggregate-only.

Closure API, if included:

- `media_type` filter for saved/read-it-later context.
- Explicit accepted media type values.
- Total counts by media type.

Capability entry:

- `collections.reading_list.per_media_type.server`

### Chunk-Level Reading TTS

Current Chatbook contract defers chunk-level TTS for both local and server.

Closure APIs, if included:

- `GET /api/v1/reading/items/{item_id}/chunks`
- `POST /api/v1/reading/items/{item_id}/chunks/{chunk_id}/tts`
- `GET /api/v1/reading/items/{item_id}/chunks/{chunk_id}/tts/{job_id}`
- `GET /api/v1/reading/items/{item_id}/chunks/{chunk_id}/tts/{job_id}/audio`
- `DELETE /api/v1/reading/items/{item_id}/chunks/{chunk_id}/tts/{artifact_id}`

Required semantics:

- Chunk identity.
- Content hash binding.
- Artifact retention.
- Progress/resume.
- Voice/provider overrides.
- Deletion behavior.

Capability entry:

- `media.reading.tts.chunk_level.server`

### Workspace Notes Graph

Current Chatbook contract limits notes graph operations to server user-space notes.

Closure APIs, if included:

- `GET /api/v1/workspaces/{workspace_id}/notes/graph`
- `GET /api/v1/workspaces/{workspace_id}/notes/{note_id}/neighbors`
- `POST /api/v1/workspaces/{workspace_id}/notes/links`
- `DELETE /api/v1/workspaces/{workspace_id}/notes/links/{link_id}`

Capability entry:

- `notes.graph.workspace`

### Cross-Scope Note And Workspace Moves

Current Chatbook contract defers cross-scope moves.

Closure APIs, if included:

- `POST /api/v1/notes/{note_id}/move-to-workspace`
- `POST /api/v1/workspaces/{workspace_id}/notes/{note_id}/move-to-user-space`
- `POST /api/v1/workspaces/{source_workspace_id}/notes/{note_id}/move-to-workspace/{target_workspace_id}`

Required semantics:

- Identity continuity.
- Audit history.
- Permission checks on source and destination.
- Version/ETag handling.
- Conflict behavior.
- Rollback or failure atomicity.
- Idempotency keys for retried moves.
- Stable source and destination references in the response.

Capability entries:

- `notes.cross_scope_move.server_to_workspace`
- `notes.cross_scope_move.workspace_to_server`
- `notes.cross_scope_move.workspace_to_workspace`

Local-to-server and server-to-local are not pure server moves. They require Chatbook import/export semantics and should remain separate from server workspace move APIs.

Decision recording rule:

- If a closure API is included, it must move from this section into a concrete implementation lane with route, schema, permission, and tests.
- If deferred, the capability catalog must include the corresponding unsupported operation with `contract_deferred` or `server_contract_missing`.
- Chatbook must continue hard-stopping deferred operations before dispatch.

## Chatbook-Local Only

These do not require new server APIs:

- Local notifications and local notification preferences.
- Local MCP governance backing.
- Local watchlists and read-it-later parity.
- Local writing-suite storage and drafting.
- Local research sessions.
- Local chat execution through existing Chatbook chat flow.
- Local/offline notes graph if later adopted.
- Local speech artifacts if implemented as local-only history.

## Implementation Sequencing

Recommended parallel lanes:

1. Capability catalog: build first because every other lane uses its operation IDs.
2. Missing first-class APIs: chat conversation create/delete, chat loop attach/persist, ingestion source delete.
3. Stabilization contracts: notifications/events, workflows/artifacts, research/artifacts, watchlists/runs, audio/job events.
4. Optional deferrals: per-media-type read-it-later, chunk TTS, workspace graph, cross-scope moves.
5. Chatbook service adapters: consume capability catalog and remove hardcoded unsupported assumptions only after the server operation is present.

## Testing Requirements

Server-side tests:

- Capability catalog filters by authenticated principal.
- Every Chatbook operation ID has either a supported route or an explicit unsupported reason.
- Missing first-class APIs enforce ownership, scope, and permissions.
- Event streams resume by cursor and dedupe duplicate events.
- Delete/move APIs are idempotent or return stable terminal errors.

Chatbook-side tests:

- Capability catalog drives action availability.
- Chatbook does not dispatch unsupported operations.
- Active server switching invalidates capability and event cursor caches.
- Server IDs remain source-qualified in shared service seams.
- Chatbook-local operations continue to work without a server.

## Open Decisions

1. Whether to close optional deferrals now or keep them explicit in capabilities.
2. Whether capability catalog should be one endpoint or split into `GET /api/v1/capabilities` plus domain-specific detail endpoints.
3. Whether chat-loop persistence should create conversations at run start or attach/persist only after finalization.
4. Whether cross-scope note moves are worth implementing before sync/write replay exists.
