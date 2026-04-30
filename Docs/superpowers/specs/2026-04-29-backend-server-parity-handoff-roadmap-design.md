# Backend Server Parity Handoff Roadmap Design

Date: 2026-04-29

Status: Approved design draft

Related docs:

- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-28-next-tranche-parallel-execution-design.md`
- `Docs/superpowers/specs/2026-04-29-connection-auth-foundation-design.md`
- `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`
- `Docs/Development/server-client-provider-migration-audit.md`

## Purpose

This design turns the remaining Chatbook-to-Server parity work into a backend-first execution roadmap that can be handed to the UX developer without forcing the UX layer to infer source authority, permissions, sync behavior, or unsupported actions from current screens.

Assume the `tldw_server` parity PR that closes research, writing, feedback, study deck delete/job listing, sharing/MCP event-stream, scoped catalog, and durable MCP event replay gaps has landed. The remaining work is no longer broad route coverage. It is backend foundation hardening, source-honest event/sync behavior, narrow domain edge closure, and explicit UX contracts.

## Goals

- Preserve Chatbook as a standalone local client that works without a server.
- Make server mode explicit, permission-aware, and safe for a single Chatbook user accessing a multi-user server.
- Keep one active-server authority, one credential lifecycle, one capability snapshot seam, and one server-client provider path.
- Build sync/mirror behavior only behind explicit eligibility gates.
- Provide UX handoff contracts after each backend tranche.
- Avoid workflow implementation, billing/admin/ops work, and current UI redesign.

## Non-Goals

- Do not implement workflows, scheduler workflows, or chat workflows.
- Do not redesign current UI screens.
- Do not make UI tests blockers while the UI/UX layer is being rewritten, unless a test validates service wiring or contract behavior.
- Do not create local mirrors for remote-only utilities unless a separate local parity design approves it.
- Do not introduce second authorities for active server, credentials, events, capabilities, or sync.
- Do not use the MCP SDK.

## Recommended Approach

Use a foundation-gated backend roadmap.

The backend work lands in strict foundation order first, then domain edges fan out once shared seams are stable. UX handoff contracts are produced as tranche outputs instead of driving backend design from current UI needs.

Rejected alternatives:

- Domain-sliced first: faster for individual screens, but likely to duplicate credential, event, server-switch, and sync decisions.
- UX-contract first: useful if UX is blocked immediately, but contracts would churn until connection/auth and sync foundations stabilize.
- All-at-once parity push: too broad and likely to recreate merge contention and hidden authority drift.

## Baseline

The baseline assumes server parity route/API coverage has landed for:

- Research run list/detail/delete/bundle/artifact coverage.
- Writing manuscript/version/trash restore coverage.
- Explicit feedback CRUD coverage.
- Study deck delete and job listing coverage.
- Sharing and MCP event-stream coverage.
- Scoped catalog multi-user fixture coverage.
- Durable MCP Hub event replay beyond the in-process event ring.

Any remaining work should treat those APIs as available and focus on client reliability, source authority, and UX-facing contracts.

## Tranche 1: Connection/Auth Foundation

This is the first backend merge gate. It must extend existing seams rather than adding another selected-server or credential authority.

Existing authorities to preserve:

- `RuntimePolicyContext` remains the app-authoritative active-source and active-server state.
- `ActiveServerCapabilityService` remains the capability snapshot authority.
- `ConfiguredServerTargetStore` remains the persisted server target/profile base and may be wrapped or generalized.
- `RuntimeServerContextProvider` becomes the only supported way for domain services to obtain credential-bound server clients.
- Legacy config-based `tldw_api` construction remains only as audited compatibility input during migration.

Required backend behavior:

- Secure per-server credential storage with fake/in-memory test backend.
- Active-profile-only legacy credential import.
- Global sign-out clears all stored credentials, including orphaned entries.
- Server switching invalidates clients, capability snapshots, remote selections, event streams, and sync cursors.
- Missing, expired, unreachable, unauthorized, and auth-required states return typed failures.
- Secrets never appear in JSON profiles, logs, exports, exception strings, unsupported reports, or cache keys.

UX handoff output:

- Active server status contract.
- Auth state contract.
- Credential-required, auth-expired, unreachable, unauthorized, and unavailable error contract.
- Server-profile switching invalidation contract.

Acceptance criteria:

- No domain service introduces direct credential caching or config-owned active-server selection.
- Server mode cannot silently fall back to local writes.
- Local mode does not require credential state.
- Tests prove server switching cannot reuse credentials or cached clients from another profile.

## Tranche 2: Realtime And Notifications Foundation

This tranche creates a backend event-observation layer that feeds UX without exposing transport details to screens.

Core components:

- `ServerEventObserver`: observes server SSE streams now and reserves a transport interface for WebSocket later.
- `NormalizedEventRecord`: source authority, server profile ID, stream name, stream instance ID, event ID or cursor, dedupe key, timestamps, payload hash, payload kind, entity reference, and delivery state.
- `LocalNotificationStore`: Chatbook-owned local notification inbox for local/offline events.
- `ServerNotificationPresentation`: local presentation of server-owned feed/reminder/event state without making local notification state authoritative.
- Cursor and dedupe store scoped by source, server profile, stream, and retention policy.

Rules:

- Server streams stop immediately on server switch, logout, or credential clearing.
- Cursor advancement happens only after event processing succeeds.
- Reconnect uses server cursor where available.
- If the server lacks a stable cursor, dedupe uses source, server profile, stream, event kind, entity ID, timestamp, and payload hash.
- Server reminders and feed read/dismiss state remain server-owned.
- Local notification settings and category/global delivery settings apply to local producers.
- Server events may become local presentation records, but not local authoritative resources.

UX handoff output:

- Notification feed item contract.
- Event stream status contract.
- Local/offline versus server-owned reminder/feed contract.
- Cursor reset and requery error contract.

Acceptance criteria:

- Reconnects do not duplicate already-presented events.
- Server switching cannot replay old-server events into the active-server presentation.
- Local notifications work offline.
- Server notification/reminder state remains active-server owned.

## Tranche 3: Sync/Mirror Foundation

This tranche builds sync infrastructure without enabling write sync by default. The first implementation is dry-run/read-only mirror reporting.

Core components:

- `EntityIdentityMap`: local entity ID, remote entity ID, source scope, active server profile ID, and workspace ID where relevant.
- `SyncProfileState`: per-server state for cursors, last mirror report, last error, and eligibility.
- `LocalOutbox`: queued local mutations, disabled unless a domain passes eligibility.
- `RemotePullCursorStore`: per-server and per-domain pull cursors.
- `ConflictPolicy`: explicit enum such as `remote_wins`, `local_wins`, `manual`, and `unsupported`.
- `DomainSyncEligibilityRegistry`: every domain defaults to `not_eligible`.

Eligibility checklist before write sync:

- Stable local IDs.
- Stable remote IDs.
- Defined source and scope mapping.
- Create, update, delete parity or explicit unsupported operation handling.
- Version, hash, timestamp, or ETag strategy.
- Conflict policy.
- Safe server-switch behavior.
- Redaction policy.
- Tests proving no accidental mutation in dry-run mode.

Initial candidates remain read-only/dry-run only:

- Selected notes.
- Media metadata.
- Possibly chat metadata, but only as server-owned/read-only until server create/delete and persist identity are explicitly modeled.
- Workspace records only after explicit workspace-aware sync rules exist.

UX handoff output:

- Sync status contract.
- Dry-run mirror report contract.
- Conflict report contract.
- Unsupported sync capability contract.
- Server-switch invalidation behavior for sync views.

Acceptance criteria:

- No domain can enter sync accidentally.
- No domain can use write sync until it passes the checklist.
- Workspace-scoped records preserve workspace boundaries.
- Server switching cannot replay queued operations against the wrong server.

## Tranche 4: Domain Edge Closure

After the foundations land, domain work should be narrow and source-honest. Each domain must define authority, unsupported operations, and UX-facing contracts before implementation.

Priority edges:

- Chat: source-separated history, streaming/persist handoff, server-owned sync semantics, and local chat-loop execution decision.
- Media/Reading: ingest-job event/status normalization, chunk-level TTS playback adoption, and saved-view support only where the server contract exists.
- Notes/Workspaces: graph semantics, local graph unsupported reports, workspace-aware sync rules, and deferred cross-scope moves.
- Writing: source-honest server-only analysis operations and optional local richer analysis only if a local engine is approved.
- Research: run, event, bundle, and artifact consistency; listing constraints; local/remote session vocabulary alignment.
- Study/Evaluations: result artifact normalization and target catalog discovery.
- RAG/Embeddings: local per-media embedding admin decision and server collection export contract.
- Audio/Voice: WebSocket transport slices, speech job/artifact parity, and source-honest history.
- Remote-only utilities: sharing, web clipper, translation, server tools, Text2SQL, server skills, claims, meetings, outputs, Kanban, and Prompt Studio stay active-server owned unless separately approved for local parity.

Rules:

- Each edge has an explicit source authority decision.
- Each unsupported edge is reported through the scope-service unsupported-capability seam.
- Domain services consume connection/auth, event, and sync seams rather than inventing their own.
- Tests are service, scope, policy, and API-client tests. UI tests are not required while the UI layer is being rewritten.

UX handoff output:

- Per-domain capability matrix.
- Per-domain view-model contract.
- Unsupported action reason codes.
- Required source selector states.
- Workspace isolation rules where relevant.

Acceptance criteria:

- UX can disable or explain unsupported actions from machine-readable reports.
- UX can render source-separated local/server/workspace state without reading service internals.
- Domain service behavior remains stable if current UI screens are replaced.

## Tranche 5: UX Handoff Packet

This packet is updated after each tranche and finalized after domain edge closure.

Required packet sections:

- Active server profile and auth status.
- Source authority and source selector states.
- Capability and unsupported-action report shape.
- Notification and event feed records.
- Sync status, dry-run mirror, and conflict records.
- Unified MCP local/server pane contract.
- Per-domain view-model contracts.
- Workspace isolation rules.
- Server unavailable/auth-expired/error presentation rules.

The UX developer should consume this packet and service-level contracts, not current UI implementation details.

## Execution Model

Execution order:

1. Land Connection/Auth.
2. Start Realtime/Notifications after connection/auth invalidation rules stabilize.
3. Start Sync/Mirror after connection/auth and event identity rules stabilize.
4. Run domain edge closures in parallel once shared seams are stable.
5. Update the UX handoff packet after each tranche.

Parallelism rules:

- No tranche creates a second active-server, credential, event, capability, or sync authority.
- Shared files get one integration owner.
- Domain sub-batches use additive tests where possible.
- UI dev works from contracts, not current UI code.
- UI tests remain non-blocking unless they validate service wiring or contract behavior.

## Testing Strategy

Required tests:

- Connection/auth unit tests.
- Credential redaction and lifecycle tests.
- Server switching invalidation tests.
- Capability snapshot invalidation tests.
- Runtime-policy hard-stop tests.
- Event observation cancellation, retry, cursor, and dedupe tests.
- Local notification delivery and settings tests.
- Sync dry-run, identity, outbox-disabled, cursor, and conflict tests.
- Domain-specific service/scope/API-client tests.

Do not use UI tests as blockers for this roadmap while the UI is being rebuilt.

## Risks And Mitigations

- Risk: domain services keep reading legacy config directly.
  - Mitigation: provider migration audit and guard tests for new raw builders.

- Risk: server switching leaks stale state.
  - Mitigation: central invalidation hooks and fake multi-server tests.

- Risk: sync starts before identity and conflict rules exist.
  - Mitigation: sync eligibility defaults to false and dry-run is required first.

- Risk: local notifications become confused with server reminders.
  - Mitigation: source-scoped event records and separate server-owned presentation state.

- Risk: UX rebuild guesses backend behavior.
  - Mitigation: tranche-level handoff contracts and machine-readable unsupported-capability reports.

## Next Step

Create an implementation plan for Tranche 1: Connection/Auth Foundation. That plan should be scoped to active-server profile selection, secure credential lifecycle, provider/client invalidation, capability snapshot invalidation hooks, and typed unavailable/auth failures. It should not include workflows or UI redesign.
