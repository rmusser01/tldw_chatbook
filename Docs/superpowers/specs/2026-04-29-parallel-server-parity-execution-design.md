# Parallel Server Parity Execution Design

Date: 2026-04-29

Status: Draft for review

Related docs:

- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/Development/server-client-provider-migration-audit.md`
- `Docs/Parity/2026-04-21-capability-matrix.md`
- `Docs/Parity/2026-04-21-gap-ledger.md`
- `Docs/Parity/2026-04-21-execution-roadmap.md`

## Purpose

This design defines how to execute the remaining Chatbook/server parity work with maximum safe parallelism.

The previous connection/auth/server-switching tranche established the shared connection foundation: runtime-policy active server state, credential storage, provider-backed server clients, token lifecycle handling, provider-client caching, and a migration audit. The next phase should exploit that foundation without letting parallel teams create conflicting source-authority, event, sync, or provider patterns.

The goal is parallel execution with explicit integration gates.

## Non-Goals

- Do not implement workflows, scheduler workflows, or chat workflows.
- Do not use the MCP SDK.
- Do not redesign the UI. UI/UX work remains owned by the parallel UI effort.
- Do not treat broad UI test failures as blockers unless the test validates service wiring or backend contracts.
- Do not introduce a second active-server authority.
- Do not enable write sync for any domain in this phase.
- Do not add new direct legacy `tldw_api` config client construction.

## Authority Baseline

Parallel lanes must share one active-server authority. These are the only allowed seams for server/profile/credential/capability state:

- `runtime_policy.bootstrap.RuntimePolicyContext`: active source, active server ID, reachability, and auth probe state.
- `MCP.server_target_store.ConfiguredServerTargetStore`: persisted server profile and target metadata. It must not store secret material.
- `runtime_policy.server_context.RuntimeServerContextProvider`: credential lookup, active server context, credential-bound client construction, and provider-client cache invalidation.
- `runtime_policy.server_capabilities.ActiveServerCapabilityService`: active-server capability snapshot refresh.

No lane may add:

- A second active-server registry.
- A second selected-server state file.
- A second capability snapshot authority.
- Domain-specific server credential caches.
- Server profile JSON fields containing bearer, API, refresh, OAuth, BYOK, or session secrets.

Any implementation plan that needs server identity, credentials, clients, or capability state must name which baseline seam it uses. Any exception requires a separate design review before work starts.

## Recommended Approach

Use **Parallel Foundations With Integration Gates**.

Run independent lanes in parallel, but force shared behavior through small common contracts and staged merge gates. This gives higher throughput than a purely sequential foundation-first plan while avoiding the drift risk of domain teams inventing their own event, sync, source, and provider semantics.

Rejected alternatives:

- **Domain-sliced parallelism only:** fast visible movement, but high risk of duplicated event/sync/source behavior.
- **Infrastructure-only parallelism:** safest architecturally, but too slow to close user-visible parity gaps.

## Parallel Lanes

### Lane 0: Shared Schema Slice

Owns the first integration slice. This lane must land before other lanes depend on shared event, notification, sync, or provider migration metadata.

Deliverables:

- `NormalizedEventRecord`
- `EventCursor`
- `EventDedupeKey`
- `NotificationPresentationRecord`
- `SyncIdentityMapEntry`
- `SyncReadinessReport`
- `ProviderMigrationStatus`

Rules:

- Schemas must be lightweight dataclasses, protocols, or typed dictionaries.
- Schemas must not import UI modules.
- Schemas must not perform network or database work.
- Schema changes after initial landing require integration review.

### Lane A: Realtime And Notifications Foundation

Owns event observation and notification presentation semantics.

Deliverables:

- Event model dataclasses for normalized event records.
- Event cursor model.
- Dedupe key model.
- Server SSE observer interface with reconnect, backoff, cancellation, and cursor resume hooks.
- Local event producer interface.
- Notification presentation adapter that keeps local notification state separate from server-owned reminders, feeds, and notification records.
- Tests for dedupe, cursor resume, active-server switch cancellation, and retention bounds.

Required event identity fields:

- Source authority: `local` or `server`.
- Active server profile ID for server events.
- Stream name and stream instance ID.
- Event ID or server cursor when available.
- Fallback dedupe key based on source, stream, event kind, entity ID, timestamp, and payload hash.
- Emitted timestamp and received timestamp.
- Transport type: local producer, SSE, WebSocket, polling, or manual refresh.
- Payload kind and normalized entity reference.
- Delivery state for local presentation, separate from server-owned read/dismiss state.

Cursor and acknowledgement rules:

- Cursor keys are scoped by active server profile ID, stream name, and stream instance ID.
- Cursor state must never be shared across server profiles.
- Cursor advancement happens only after local processing acknowledges the event.
- Unsupported or stale server cursors must produce a typed reset/requery result rather than silently replaying across streams.
- Dedupe retention must be bounded by count and/or time.
- Reconnect tests must cover duplicate events, unacknowledged events, stale cursors, and active-server switch isolation.

### Lane B: Provider Migration Batches

Owns migration of remaining server-backed services to `RuntimeServerContextProvider`.

Initial high-priority batch:

- `Chat/server_chat_conversation_service.py`
- `Chat/server_chat_loop_service.py`
- `Character_Chat/server_character_persona_service.py`
- `Character_Chat/server_chat_dictionary_service.py`
- `Chatbooks/server_chatbook_service.py`
- `Media/server_media_reading_service.py`
- `Notes/server_notes_workspace_service.py`
- `Prompt_Management/server_prompt_service.py`
- `Prompt_Management/prompt_scope_service.py`
- `Prompt_Studio_Interop/server_prompt_studio_service.py`
- Prompt/chatbook startup consumers in `app.py`

Deliverables:

- Provider-backed factories for each migrated service.
- Existing direct-client and `from_config()` compatibility preserved until the migration audit says otherwise.
- App/service wiring migrated to provider-backed construction only after tests exist.
- Migration audit updated after each batch.
- Tests proving direct-client compatibility, provider-backed construction, policy-denied no-client-build behavior, and no new direct config builders.

### Lane C: Sync And Mirror Dry-Run Substrate

Owns sync metadata and readiness contracts only. It must not enable write sync or queue replayable mutations.

Deliverables:

- Sync identity map model.
- Per-server sync profile state.
- Remote pull cursor model.
- Local outbox model shape, with no replay worker and no write dispatch path.
- Conflict strategy enum and default conflict policy.
- Per-domain sync eligibility registry.
- Sync readiness report.
- Dry-run/read-only mirror report API.
- Source-aware unsupported reports for unsyncable domains.

Rules:

- Every domain defaults to `sync_eligible=False`.
- Write-enabled sync is forbidden until a domain passes the identity-readiness checklist.
- Chat metadata remains read-only/server-owned until server create/delete and persist identity limitations are modeled.
- Workspace-scoped records must preserve workspace boundaries.
- Server switching must not replay queued operations against the wrong server.
- This tranche exposes no replay worker.
- This tranche exposes no remote mutation dispatch from sync.
- This tranche exposes no write-enabled domain registration.
- This tranche writes no authoritative local mirror copy of server-owned records.
- Tests must prove dry-run and readiness paths do not call server create, update, delete, or mutation APIs.

Identity-readiness checklist:

- Stable local entity IDs.
- Stable remote entity IDs.
- Explicit local-to-remote identity map.
- Defined source and scope mapping.
- Create/update/delete parity or explicit unsupported-operation handling.
- Version, timestamp, hash, or ETag strategy.
- Conflict strategy.
- Server-switch safety.
- Redaction policy.

### Lane D: Domain Edge Closure Prep

Owns domain-specific non-UI parity edges that can proceed without violating shared foundations.

Initial domains:

- Chat.
- Media/reading.
- Notes/workspaces.

Deliverables:

- Domain edge specs and service-level tests.
- Source-authority decision for each edge.
- Unsupported-capability reports for unresolved or deferred edges.
- No UI implementation except service wiring tests.
- No new server client construction outside the provider migration rules.

Initial edge focus:

- Chat: local chat-loop execution decision, streaming/persist handoff alignment, source-separated history contracts.
- Media/Reading: ingest-job event/status normalization, saved-view behavior constrained by server contract, chunk-level TTS adoption decision.
- Notes/Workspaces: graph semantics, workspace-aware sync design, local/offline graph generation decision, cross-scope moves still deferred.

Per-domain acceptance template:

- Source owner: local, server, workspace, shared, or deferred.
- Read behavior by source.
- Write behavior by source.
- Server-switch behavior.
- Workspace isolation behavior.
- Unsupported-capability IDs and user-facing reason codes.
- Provider dependency status: migrated, compatibility mode, or not applicable.
- Required service/scope/API-client tests.
- UI impact: service wiring only, contract fixture, or no UI touch.

### Lane E: UX Handoff Contracts

Owns backend/view-model contracts for the parallel UI/UX rewrite.

Deliverables:

- Active-server status contract.
- Unsupported-action presentation contract.
- Notification feed contract.
- Source selector contract.
- Workspace isolation contract.
- Future sync status/conflict contract.
- Contract fixtures/tests usable by the UI coworker.

Rules:

- Do not rebuild current screens.
- Do not require broad UI tests.
- Contracts consume runtime-policy state and unsupported-capability reports instead of guessing authority.

Required fields per UX contract:

- Contract ID and version.
- Source owner and active source.
- Active server profile ID when relevant.
- Capability/action ID.
- Unsupported reason code and user message when unsupported.
- Server reachability/auth state when server-backed.
- Workspace scope ID when workspace-scoped.
- Fixture payloads for local, server, unavailable server, unsupported action, and workspace isolation cases.
- Required contract tests.

## Shared Schemas First

The first integration slice should define shared contracts before lane-specific implementation grows.

Shared records:

- `NormalizedEventRecord`
- `EventCursor`
- `EventDedupeKey`
- `NotificationPresentationRecord`
- `SyncIdentityMapEntry`
- `SyncReadinessReport`
- `ProviderMigrationStatus`

These should be small dataclasses, protocols, or typed dictionaries with tests. They should avoid importing heavy UI modules and should not perform network or database work directly.

## Integration Gates

### Gate 1: Shared Schema Gate

Lanes that need event, notification, sync, or provider migration metadata must use the shared schemas. Schema changes require an integration review before dependent lane work merges.

### Gate 2: Provider Migration Gate

Any new server-backed service path must use `RuntimeServerContextProvider`.

Existing compatibility factories may remain temporarily only if already listed in `Docs/Development/server-client-provider-migration-audit.md`. No lane may add:

- New direct `build_runtime_api_client_from_config()` consumers.
- New direct `build_runtime_api_client(app_config=...)` consumers.
- New indirect `from_config()` wrappers that construct server clients from app config.
- New prompt/chatbook compatibility factory consumers.
- New UI/event helper call sites that construct server clients from app config.

Every provider migration batch must update the direct and indirect migration audit.

### Gate 3: Event Before Notification Contract Finalization

UX notification contracts can draft early, but cannot finalize until event identity, cursor, dedupe, reconnect, and retention semantics are stable.

### Gate 4: Sync Dry-Run Gate

Sync work may define state, reports, identity maps, and dry-run behavior. Write-enabled sync is blocked until a domain-specific readiness report is approved.

### Gate 5: Source Authority Gate

Every domain edge must declare whether the action is local-owned, server-owned, workspace-owned, shared, or deferred. Unsupported actions must be exposed through the scope-service unsupported-capability seam.

### Gate 6: Tranche Integration Review

Each lane can commit independently, but a tranche is not complete until:

- Focused service/scope/policy/API tests pass.
- Service-wiring UI tests pass where wiring changed.
- Provider migration audit is updated.
- No-secret checks pass where credential or target state is touched.
- Server-switch behavior is tested for any remote cache, event, or sync state.
- A cross-lane final review approves the integration.

## First Parallel Tranche

Run these work items concurrently:

1. **Lane 0:** Shared schema slice.
2. **Lane A:** Realtime/event observer skeleton using shared event schemas.
3. **Lane B:** Provider migration high-priority batch 1: chat, media, notes/workspaces.
4. **Lane C:** Sync readiness registry and dry-run report core, no outbox replay.
5. **Lane D:** Domain edge specs/tests for chat, media/reading, notes/workspaces.
6. **Lane E:** UX handoff contracts for active server, unsupported actions, and notifications.

Recommended merge order:

1. Shared schemas.
2. Provider migration batch.
3. Realtime observer skeleton.
4. Sync readiness dry-run.
5. Domain edge service tests/specs.
6. UX handoff contracts.

The lanes can develop concurrently, but later merge order should respect these dependencies.

## Parallel Ownership

Suggested branch/worktree ownership:

| Lane | Branch/worktree | Primary file ownership | Restricted files | Merge rule |
| --- | --- | --- | --- | --- |
| Lane 0 | `parity-shared-schemas` | Shared model/schema modules and schema tests | App wiring, domain services, UI screens | Lands first. Later schema edits require integration review. |
| Lane A | `parity-events-foundation` | Event observer services, event cursor/dedupe store, notification presentation adapter, event tests | Provider migration files, sync write paths, UI screens | May depend on Lane 0. Must cancel on active-server switch. |
| Lane B | `parity-provider-migration-batch1` | Server service constructors/factories, app service wiring, provider migration tests, migration audit | Event observer internals, sync substrate, domain edge specs except audit notes | Owns app wiring for migrated services. Must not add legacy builders. |
| Lane C | `parity-sync-dry-run-substrate` | Sync readiness models, dry-run report services, sync eligibility tests | Domain mutation dispatch, event observer internals, UI screens | No replay worker or remote mutation calls. |
| Lane D | `parity-domain-edge-contracts` | Domain edge specs, service/scope tests, unsupported-capability reports for chat/media/notes | App wiring and service constructors until Lane B lands | Specs/tests only for overlapping provider-migrated domains until Lane B merges. |
| Lane E | `parity-ux-handoff-contracts` | View-model contracts, fixtures, contract tests | Current UI screen implementation | Contracts only; no UI redesign. |

Conflict rules:

- Lane B owns service constructor and app wiring edits for migrated services.
- Lane D may add domain tests/specs in the same domain areas, but must not change provider construction until Lane B lands.
- Lane A and Lane C may share schema dependencies only through Lane 0 outputs.
- Any lane touching authority baseline seams needs explicit integration review.
- Any lane touching `app.py` must list the exact initialization block it owns before implementation starts.

## Testing Strategy

Prefer backend tests over UI tests.

Required tests by lane:

- Lane A: event dedupe, cursor resume, reconnect, cancellation on active-server change, retention.
- Lane B: provider-backed construction, direct-client compatibility, policy-denied no-client-build, migration audit guard.
- Lane C: sync eligibility default false, readiness failures, server-switch isolation, workspace boundary preservation.
- Lane D: source-authority hard stops, unsupported-capability reports, service-level behavior.
- Lane E: contract fixture shape and mapping from runtime-policy/unsupported-capability inputs.

Cross-lane tests:

- Active-server switch invalidates provider clients, event observers, remote selections, and sync cursors.
- No local fallback for server writes.
- No server credential leakage into app config, target-store JSON, docs, logs, or cache-key reprs.

## Risks And Mitigations

- Risk: parallel lanes diverge on event or sync identity.
  - Mitigation: shared schemas first and integration-gated schema changes.

- Risk: provider migration creates merge conflicts in `app.py`.
  - Mitigation: migrate in small batches and keep app wiring edits centralized.

- Risk: sync infrastructure becomes accidental write sync.
  - Mitigation: sync eligibility defaults false and write replay is out of scope.

- Risk: domain lanes add new direct legacy client construction.
  - Mitigation: audit guard and provider migration gate.

- Risk: UI handoff contracts bake in unstable event semantics.
  - Mitigation: notification contract finalizes only after event identity and retention are stable.

## Success Criteria

- Multiple lanes can progress without editing the same files unnecessarily.
- Shared foundations remain source-honest and server-switch-safe.
- High-priority services move toward provider-backed clients.
- Event and sync foundations are testable before broad domain adoption.
- UI coworker receives stable contracts without requiring current UI fixes.
- The provider migration audit stays current.
- No write-enabled sync ships from this tranche.
