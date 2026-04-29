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

## Recommended Approach

Use **Parallel Foundations With Integration Gates**.

Run independent lanes in parallel, but force shared behavior through small common contracts and staged merge gates. This gives higher throughput than a purely sequential foundation-first plan while avoiding the drift risk of domain teams inventing their own event, sync, source, and provider semantics.

Rejected alternatives:

- **Domain-sliced parallelism only:** fast visible movement, but high risk of duplicated event/sync/source behavior.
- **Infrastructure-only parallelism:** safest architecturally, but too slow to close user-visible parity gaps.

## Parallel Lanes

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

Owns sync metadata and readiness contracts only. It must not enable write sync.

Deliverables:

- Sync identity map model.
- Per-server sync profile state.
- Remote pull cursor model.
- Local outbox model, disabled by default.
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

Any new server-backed service path must use `RuntimeServerContextProvider` or be explicitly documented as compatibility mode in the migration audit. No new direct `build_runtime_api_client_from_config()` consumers may be added.

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

1. **Lane A:** Realtime/event shared schema plus observer skeleton.
2. **Lane B:** Provider migration high-priority batch 1: chat, media, notes/workspaces.
3. **Lane C:** Sync readiness registry and dry-run report core, no outbox replay.
4. **Lane D:** Domain edge specs/tests for chat, media/reading, notes/workspaces.
5. **Lane E:** UX handoff contracts for active server, unsupported actions, and notifications.

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

- `parity-events-foundation`: event records, observers, notification presentation adapter.
- `parity-provider-migration-batch1`: high-priority provider migration.
- `parity-sync-dry-run-substrate`: sync readiness and dry-run substrate.
- `parity-domain-edge-contracts`: chat/media/notes edge specs and service tests.
- `parity-ux-handoff-contracts`: view-model contracts and fixtures.

File ownership should be explicit in implementation plans to avoid merge conflicts. Shared schema files should be owned by the event/sync foundation lane first, then treated as integration-gated.

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
