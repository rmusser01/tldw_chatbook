# Remaining Server Parity Roadmap Design

Date: 2026-04-28

Status: Approved for planning

Related docs:

- `Docs/Parity/2026-04-21-capability-matrix.md`
- `Docs/Parity/2026-04-21-gap-ledger.md`
- `Docs/Parity/2026-04-21-execution-roadmap.md`
- `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`
- `Docs/superpowers/specs/2026-04-21-runtime-policy-capability-registry-tranche-0-design.md`

## Purpose

Chatbook should remain a standalone local application while also becoming a reliable client for `tldw_server`. Recent parity work has closed most backend route and service-adapter gaps outside intentionally deferred workflow surfaces. The remaining work is now less about adding raw API wrappers and more about making connected-server behavior reliable, source-honest, testable, and ready for a separate UI/UX rewrite.

This roadmap prioritizes connected-client reliability first. Active-server identity, credential lifecycle, capability snapshots, event observation, server switching, and sync-safe source authority are prerequisites for safe interoperability. Domain-specific parity work should build on those foundations instead of continuing to invent per-screen or per-domain source behavior.

## Goals

- Keep Chatbook fully usable without a server.
- Make server mode explicit, reliable, and bounded by the authenticated user's server permissions.
- Preserve local/server/workspace authority rules at the action level.
- Prepare for future sync and mirror behavior without silently syncing data before identity, conflict, and cursor rules exist.
- Provide backend/service contracts and acceptance criteria that the parallel UI/UX effort can consume.
- Keep the existing runtime-policy registry as the central source of action capability truth.
- Keep unsupported capabilities machine-readable and visible to consumers.

## Non-Goals

- Do not implement workflows, scheduler workflows, or chat workflows in this roadmap pass.
- Do not cover billing, admin, or ops-only surfaces.
- Do not use the MCP SDK.
- Do not redesign the current UI. The UI/UX layer is owned by a parallel effort.
- Do not treat current UI test failures as blockers unless the test validates service wiring rather than broken UI behavior.
- Do not silently mirror remote-only server resources into local authoritative state.
- Do not merge local notification state with server reminder/feed authority.

## Recommended Approach

Use a reliability-first roadmap:

1. Lock active-server/session behavior.
2. Normalize realtime and notification observation.
3. Design the sync/mirror substrate.
4. Close remaining non-UI domain edges.
5. Produce UI handoff contracts for the separate UX rewrite.

This approach is preferred over continuing domain verticals first because many remaining domain gaps depend on shared source context, server switching, capability discovery, events, or sync identity. Building those foundations first reduces rework and prevents future source-leak regressions.

## Architecture

### Server Connection Foundation

This layer owns active-server state and must become the only way domain services acquire server context. It must consolidate and extend existing seams rather than create a second active-server authority.

Existing seams to preserve:

- `runtime_policy.bootstrap.RuntimePolicyContext` and `RuntimeSourceState` remain the app-authoritative source for `active_source`, `active_server_id`, reachability, and auth probe state.
- `runtime_policy.server_capabilities.ActiveServerCapabilityService` remains the capability snapshot seam and should be hardened rather than replaced.
- `MCP.server_target_store.ConfiguredServerTargetStore` remains the existing persisted server-target/profile registry and should be generalized or wrapped for non-MCP server profile use instead of duplicated.
- `Auth_Account_Interop.AuthAccountScopeService` remains the policy-gated server auth/account operation seam.
- Legacy `tldw_api` config remains a compatibility input until migrated.

Authority rule:

- New connection/auth work must extend or wrap the seams above. It must not introduce a parallel selected-server registry, a second persisted active-server state file, a second capability snapshot authority, or domain-owned credential caches.
- Any implementation plan that needs server identity, credentials, client construction, capability snapshots, or server switching invalidation must name the existing seam it extends before code work starts.

Responsibilities:

- Active server profile selection.
- Durable credential storage.
- Token refresh and logout.
- Reachability checks.
- Capability snapshots.
- Server unavailable fallback.
- Server switching invalidation.
- Credential-bound client creation.

Domain services should not read raw app config or cache server credentials directly as the target state. They should ask the connection foundation for an active server context and fail with a typed unavailable/unauthorized result when the server context is invalid.

Migration must be staged. The first implementation should introduce a compatibility client-provider facade that can still build clients from the legacy `tldw_api` config while exposing the new active-server context API. Domain services should then migrate to the provider in audited batches. Do not attempt a big-bang rewrite of every `_require_client()` implementation in the connection/auth tranche.

Migration guardrails:

- Maintain a direct and indirect migration audit rather than treating all existing config-based factories as immediate failures.
- Every migration batch must update the audit with migrated services, remaining compatibility factories, intentional bootstrap seams, and UI/event helper call sites.
- Audit checks should compare against the current baseline and fail only on newly introduced legacy builders or newly unlisted compatibility factories.

Credential security requirements:

- Do not persist bearer, API, refresh, OAuth, or BYOK secrets in plaintext JSON profile files.
- Store secrets per server profile and per credential purpose.
- Keep server target/profile metadata separate from secret material.
- Redact secrets from logs, exceptions, debug dumps, exports, and unsupported-capability reports.
- Refresh tokens must have explicit lifecycle handling and must be cleared on logout, credential removal, and server-profile deletion.
- Server switching must never reuse a credential from another server profile.
- Existing config tokens should be treated as legacy inputs; migration should either import them into the credential store and scrub them where safe, or leave them as read-only compatibility credentials until the user explicitly migrates.
- Tests need fake or in-memory credential stores so no real OS keychain or local secret backend is required in CI.

Credential implementation constraints:

- Profile switching must re-resolve credentials by active server profile ID and must not reuse a token object cached under a different profile.
- Logout, credential removal, and server-profile deletion must clear refresh credentials and invalidate any credential-bound client cache entries.
- Test backends must be able to assert that no secret material is written to target-store JSON, exported settings, log messages, exception strings, or cache-key representations.

### Source Authority Foundation

This layer keeps local, server, and workspace authority explicit per action.

Existing runtime-policy registry behavior should remain central:

- Every action has a source-aware capability ID.
- Invalid source/action combinations hard-stop before adapter dispatch.
- Scope services expose `list_unsupported_capabilities()` or an equivalent machine-readable report.
- Local writes stay local unless a future sync policy explicitly queues them.
- Server writes stay server-owned unless a future cache/mirror policy explicitly records them locally.
- Workspace-scoped records do not leak into general user-space collections.

### Realtime And Notification Foundation

This layer normalizes event and notification inputs without merging source authority.

Responsibilities:

- Observe server SSE streams where existing server contracts expose them.
- Reserve a transport seam for later WebSocket flows.
- Normalize observed events into source-scoped event records.
- Route selected local events into Chatbook's local notification inbox.
- Route selected server events into connected-server notification presentation without making local notification state authoritative for server reminders/feeds.
- Preserve offline behavior when the active server is unavailable.

Local notification state remains Chatbook-owned. Server reminders, server feeds, server claims notifications, and other server-owned event records remain active-server owned.

Normalized event records must include enough identity for reconnect and dedupe:

- Source authority: `local` or `server`.
- Active server profile ID for server events.
- Stream name and stream instance ID.
- Event ID or server cursor when provided.
- Fallback dedupe key derived from source, stream, event kind, entity ID, timestamp, and payload hash.
- Emitted timestamp when provided and received timestamp when observed locally.
- Transport type: local producer, SSE, WebSocket, polling, or manual refresh.
- Payload kind and normalized entity reference.
- Delivery state for notification presentation, separate from server-owned read/dismiss state.

Reconnect behavior must be explicit. Observers should resume from the last acknowledged cursor where the server supports it, dedupe repeated events, bound retained event history, and cancel streams immediately when the active server changes or credentials are cleared.

Event store rules:

- Cursor keys must include source authority, active server profile ID for server streams, stream name, and stream instance ID.
- Cursor advancement happens only after local processing acknowledges the event.
- Dedupe keys must be stable across reconnects and must fall back to source, stream, event kind, entity ID, timestamp, and payload hash when the server does not provide a stable event ID.
- Event and dedupe retention must be bounded by count, age, or both.
- Unsupported, expired, or rejected cursors must produce a typed reset/requery result instead of silently replaying across streams or server profiles.

### Sync/Mirror Foundation

Sync should be explicit, domain-gated, and designed before broad adoption.

Minimum substrate:

- Local entity ID.
- Remote entity ID.
- Source scope.
- Active server profile ID.
- Sync eligibility flag.
- Last observed remote cursor/version/hash/timestamp.
- Last local dirty timestamp.
- Local outbox queue.
- Remote pull cursor.
- Conflict strategy.
- Last sync result and error state.

Sync must be opt-in per domain. No domain should get write-enabled sync until it passes an identity-readiness checklist. Early sync work should start as dry-run/read-only mirror reporting, not outbox replay.

Identity-readiness checklist:

- Stable local entity IDs.
- Stable remote entity IDs.
- Explicit local-to-remote identity map.
- Defined scope mapping for global, user, server, and workspace records.
- Clear create/update/delete parity, or explicit unsupported-operation handling.
- Version, timestamp, hash, or ETag strategy for change detection.
- Conflict strategy for concurrent local and remote edits.
- Safe server-switch behavior for queued and pulled state.
- Redaction policy for synced metadata.

Potential early candidates can include selected notes, media metadata, or chat metadata only after they satisfy the checklist. Candidate status is not approval to write sync. The first implementation for any candidate domain must be dry-run/read-only mirror reporting with an explicit no-mutation test boundary. Chat metadata is not eligible for write-enabled sync until server create/delete and streaming/persist identity limitations are resolved or explicitly modeled as read-only/server-owned edges. Workspace data should remain isolated and require explicit workspace-aware sync rules.

## Data Flow Rules

### Local Mode

- Writes go to Chatbook local services and local DBs.
- Reads come from local services.
- Local producers may emit local notification records.
- Sync outbox writes happen only for domains explicitly marked sync-eligible.
- No active server is required.

### Server Mode

- Writes go to the active server.
- Reads come from the active server or an explicitly marked remote cache.
- Server events may be observed and shown locally, but server-owned resources remain server-owned.
- Local cache/mirror writes require an explicit approved cache or sync policy.
- If the active server becomes unavailable, server actions fail with source-aware unavailable results rather than silently falling back to local writes.

### Workspace Scope

- Workspace-scoped records stay isolated from general notes/media/chat collections.
- Workspace-scoped notes do not appear in general notes lists.
- Cross-scope moves remain deferred.
- Future sync must preserve workspace boundaries and remote workspace identity.

### Server Switching

Switching the active server must invalidate:

- Server-scoped caches.
- Capability snapshots.
- Active remote selections.
- Credential-bound clients.
- Pending server event streams.
- Remote sync cursors for the previous server context.
- Any UI view-model state derived from the previous server.

Local state must remain intact and independent from server switching.

## Workstreams

### 1. Connection And Auth

This should be the next implementation target.

Deliverables:

- Consolidation plan for `RuntimePolicyContext`, `ActiveServerCapabilityService`, `ConfiguredServerTargetStore`, and `AuthAccountScopeService`.
- Server profile registry suitable for one active server now and multiple switchable servers later, built by extending or wrapping `ConfiguredServerTargetStore` instead of adding an unrelated registry.
- Secure durable credential/token storage with a testable in-memory backend.
- Token refresh policy.
- Logout and credential clearing.
- Server profile switching with cache invalidation.
- Active-server capability snapshot hardening through `ActiveServerCapabilityService`.
- Compatibility client-provider facade for services that still build clients from legacy `tldw_api` config.
- Audit list of server-backed domain services and their migration status to the provider facade.
- Consistent unavailable/unauthorized errors.
- Tests covering token refresh, logout, unreachable server behavior, and server switching invalidation.

Acceptance criteria:

- Existing runtime-policy active-server state remains authoritative.
- No second active-server registry or capability snapshot authority is introduced.
- Newly migrated domain services obtain server context from the provider facade.
- Unmigrated domain services are listed and continue to work through compatibility mode.
- Secrets are not written to plaintext JSON profile files.
- Logout, server-profile removal, and credential removal clear the right per-server secrets.
- Switching servers cannot reuse stale capability snapshots or stale clients.
- Local mode works without credential state.
- Server mode cannot silently degrade into local writes.

### 2. Realtime And Notifications

Deliverables:

- Shared server event-observation service for SSE streams.
- Transport abstraction that can later support WebSockets.
- Source-scoped normalized event records with stream IDs, event IDs or cursors, dedupe keys, timestamps, and payload hashes.
- Local notification routing from local producers.
- Optional server-event-to-notification presentation without merging server authority into local inbox state.
- Consistent retry/backoff, cursor resume, retention, dedupe, and cancellation behavior.

Acceptance criteria:

- Server event observation stops when active server changes.
- Reconnects do not duplicate already-handled events.
- Streams resume from supported cursors where the server contract allows it.
- Local notifications work offline.
- Server notification/reminder state remains server-owned.
- Notification producers do not bypass category/global delivery settings.

### 3. Sync And Mirror Foundation

Deliverables:

- Entity identity map.
- Per-server sync profile state.
- Local outbox.
- Remote pull cursor store.
- Conflict strategy enum and default conflict policy.
- Per-domain sync eligibility registry.
- Sync dry-run/reporting API.
- Identity-readiness checklist enforcement before any write-enabled sync.
- Read-only mirror/dry-run mode before outbox replay for a domain.
- Source-aware unsupported reports for unsyncable domains.

Acceptance criteria:

- No domain can enter sync accidentally.
- No domain can use write-enabled sync until it passes the identity-readiness checklist.
- Chat metadata remains read-only/server-owned for sync until server create/delete and persist identity limitations are explicitly modeled.
- Workspace-scoped records preserve workspace boundaries.
- Server switching cannot replay queued operations against the wrong server.
- Conflict results are explicit and inspectable.

### 4. Domain Edge Closure

This work should focus on non-UI backend and service gaps that remain after the broad route-coverage sprint.

Priority edges:

- Chat: local chat-loop execution decision, streaming/persist handoff alignment, source-separated history contracts, local equivalents for selected server-only adjuncts if approved.
- Media/Reading: chunk-level TTS playback adoption, per-media-type server saved views only if the server contract supports them, stronger ingest-job event/status normalization.
- Notes/Workspaces: local/offline graph generation decision, graph sync semantics, workspace-aware sync design, cross-scope moves still deferred.
- Writing Suite: dedicated backend contracts for local richer analysis only if a local model-backed engine is approved, and continued source-honest handling for server-only unsupported operations.
- Research: event/bundle/artifact consistency, server run listing constraints, and local/remote session vocabulary alignment.
- Study/Evaluations: result artifact normalization, target catalog discovery if the server exposes it, and optional local equivalents for remote-only helper flows.
- RAG/Embeddings: local per-media embedding admin parity decision and server collection export contract if exposed.
- Audio/Voice: WebSocket transport slices, local speech-job/artifact parity, and source-honest history handling.
- Remote-only utilities: sharing, web clipper, translation, server tools, Text2SQL, server skills, claims, meetings, outputs, Kanban, and Prompt Studio should remain active-server owned unless a future local parity plan is approved.

Acceptance criteria:

- Each edge has an explicit source authority decision.
- Each unsupported edge is reported through the scope-service unsupported-capability seam.
- Tests cover service-level behavior without depending on the broken current UI layer.

### 5. UX Handoff Contracts

The parallel UI/UX effort needs stable contracts, not current-screen fixes.

Deliverables:

- Per-domain view-model contracts for local/server/source scope.
- Active-server status contract.
- Unsupported-action presentation contract.
- Notification feed contract.
- Unified MCP source-pane contract.
- Sync status/conflict contract once sync foundation exists.
- Source selector and workspace isolation requirements.

Acceptance criteria:

- UI can render local/server/workspace state without guessing authority.
- UI can disable or explain unsupported actions from machine-readable capability reports.
- UI can show server unavailable/auth-expired states consistently.
- UI can be rebuilt without changing domain service semantics.

## Domain Priority Summary

Highest priority:

- Connection/auth/server switching.
- Realtime/notifications.
- Sync/mirror substrate.
- Chat, media/reading, notes/workspaces, writing, research, study/evals, RAG/admin.

Medium priority:

- Audio/voice transports.
- LLM provider/model catalog adoption.
- Server runtime/config presentation.
- External connectors.
- Prompt Studio, outputs, meetings, claims, Kanban.

Deferred:

- Workflows, scheduler workflows, and chat workflows.
- Billing/admin/ops.
- Watchlist group editing.
- Cross-scope note/workspace moves.
- Full local mirrors for remote-only surfaces unless separately approved.

## Testing Strategy

Prefer service, scope, policy, and API-client tests over current UI tests.

Required test types:

- Connection/auth unit tests.
- Server switching invalidation tests.
- Runtime-policy hard-stop tests.
- Unsupported-capability report tests.
- Scope-service source routing tests.
- Event observation cancellation/retry tests.
- Notification delivery/settings tests.
- Sync identity/outbox/cursor/conflict tests.
- Domain-specific edge tests for remaining backend gaps.

UI tests should be limited to service wiring assertions until the UI/UX rewrite lands.

## Risks And Mitigations

- Risk: domain services continue reading config directly.
  - Mitigation: require server-context access through the connection foundation.

- Risk: server switching leaks stale remote state.
  - Mitigation: central invalidation hooks and tests that switch between fake servers.

- Risk: sync starts before identity and conflict rules exist.
  - Mitigation: sync eligibility defaults to false and every syncable domain requires explicit registration.

- Risk: local notifications become confused with server reminders.
  - Mitigation: source-scoped notification/event records and separate authority rules.

- Risk: UI rebuild guesses backend capabilities.
  - Mitigation: view-model contracts consume runtime-policy and unsupported-capability reports.

- Risk: remote-only utilities are over-localized.
  - Mitigation: keep them active-server owned unless a separate local parity design is approved.

## Next Step

Create an implementation plan for the first workstream: Connection and Auth. That plan should be scoped to active-server profile registry, durable credentials, token lifecycle, capability snapshots, server switching invalidation, and service-context adoption. It should not implement workflows or UI redesign work.
