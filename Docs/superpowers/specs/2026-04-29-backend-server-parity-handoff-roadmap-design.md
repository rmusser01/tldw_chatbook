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

Capability authority rule:

- `ActiveServerCapabilityService` is the aggregator and source for UX-facing capability reports.
- Domain scope services may contribute unsupported-capability inputs, but they do not become independent capability authorities.
- Per-domain capability matrices are derived UX contracts generated from the runtime-policy/capability aggregation seam and scope-service unsupported reports.

Required backend behavior:

- Secure per-server credential storage with fake/in-memory test backend.
- Active-profile-only legacy credential import.
- Global sign-out clears all stored credentials, including orphaned entries.
- Server switching invalidates clients, capability snapshots, remote selections, event streams, and active sync/event handles.
- Server switching does not delete persisted per-server cursors by default. Persisted cursor state is retained only under its server-profile scope and is cleared only by logout, credential removal, profile deletion, or explicit user action.
- Active-server/auth context captures authenticated principal ID, permission scope, and workspace/resource scope where exposed by the server.
- Missing, expired, unreachable, unauthorized, and auth-required states return typed failures.
- Stale authorization, lost workspace access, and profile-no-longer-authorized states return typed failures instead of silently hiding or reclassifying resources.
- Secrets never appear in JSON profiles, logs, exports, exception strings, unsupported reports, or cache keys.

Credential storage contract:

- Durable credential storage is cross-platform and explicit: macOS Keychain, Windows Credential Manager, Linux Secret Service/libsecret or KWallet where available, and fake/in-memory backends only in tests.
- If no secure OS-backed store is available, persistent server credentials are disabled and the auth seam returns typed `credential_store_unavailable`; Chatbook must not fall back to plaintext JSON, TOML, SQLite, environment-file, or config-file secrets.
- Headless Linux, locked keychains, missing Secret Service/KWallet sessions, denied OS-store permissions, and unsupported keyring backends are treated as secure-store unavailable. They may use process-local/in-memory credentials only in explicit test fixtures, never as durable production fallback.
- Stored credential records use a stable Chatbook-owned, listable namespace prefix such as `tldw_chatbook.server_credentials` plus server profile ID, normalized server origin, authenticated principal where known, and credential record type.
- Credential cleanup APIs enumerate the Chatbook-owned namespace prefix. Global sign-out must delete every matching record, including entries whose server profile is no longer present in the profile store.
- Stored credentials must not be keyed only by hostname, username, current active profile, or unscoped server origin.
- Access tokens, refresh tokens, API keys, and future auth artifacts use separate record types with lifecycle metadata so refresh, revocation, and deletion cannot accidentally preserve stale credentials.
- Legacy config tokens are imported only for the active profile. Successful import removes or redacts the legacy secret; failed import leaves the original config unchanged and logs no secret material.
- Global sign-out deletes every stored credential record across known profiles and orphaned namespaces, clears credential-bound client caches, and invalidates active auth/capability/event/sync handles.
- Tests must use fake or in-memory stores and assert that serialized profiles, logs, exceptions, unsupported reports, cache keys, and exported diagnostics contain no secret material.

Credential storage implementation checklist:

- A durable credential backend is acceptable only if the backend is OS-protected and scoped to the current user session: Keychain on macOS, Credential Manager on Windows, Secret Service/libsecret or KWallet on Linux.
- Plaintext keyring backends, file keyrings, null/fail keyrings, environment-variable stores, JSON/TOML/YAML/config files, and SQLite secrets are forbidden for durable server credentials.
- Every credential username/account key starts with the stable Chatbook-owned prefix and encodes `server_profile_id`, `normalized_origin`, `principal_kind`, `principal_id` when known, and `credential_type`.
- Global sign-out calls the credential store's namespace enumeration/delete path, not only per-known-profile deletion, so orphaned namespace records are removed.
- Per-server sign-out removes all indexed records for that server profile, including future/custom credential types under the Chatbook namespace.
- Legacy config token migration must be active-profile-only, must never import a token into a nonmatching profile, and must record enough cleanup state to prevent cleared legacy tokens from being silently reimported after server switching.

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

Single event authority:

- `EventStateRepository` is the single owner for normalized event records, server stream cursors, dedupe keys, and event retention.
- `ServerEventObserver` writes through `EventStateRepository`; it is transport logic, not event authority.
- `ServerNotificationPresentation` records are derived, invalidatable presentation cache state and never become authoritative server resources.
- Existing `ClientNotificationsDB` remains authoritative only for Chatbook-owned local notifications. Tranche 2 may add a thin `LocalNotificationStore` adapter name for UX/service vocabulary, but it must wrap `ClientNotificationsDB` rather than create a second local notification database.
- The existing in-memory `EventCursorStore` becomes either a test-only implementation of the event-state protocol or a repository-backed compatibility adapter. It must not remain an independent active cursor/dedupe authority after `EventStateRepository` lands.

Core components:

- `ServerEventObserver`: observes server SSE streams now and reserves a transport interface for WebSocket later.
- `NormalizedEventRecord`: source authority, server profile ID, authenticated principal where known, stream name, stream instance ID, event ID or cursor, dedupe key, timestamps, payload hash, payload kind, entity reference, and delivery state.
- `ClientNotificationsDB`/`LocalNotificationStore` adapter: Chatbook-owned local notification inbox for local/offline events.
- `ServerNotificationPresentation`: local presentation of server-owned feed/reminder/event state without making local notification state authoritative.
- Cursor and dedupe store scoped by source, server profile, authenticated principal where known, stream, and retention policy.

Rules:

- Server streams stop immediately on server switch, logout, or credential clearing.
- Cursor advancement happens only after event processing succeeds.
- Reconnect uses server cursor where available.
- If the server lacks a stable cursor, dedupe uses source, server profile, authenticated principal where known, stream, event kind, entity ID, timestamp, and payload hash.
- Server reminders and feed read/dismiss state remain server-owned.
- Local notification settings and category/global delivery settings apply to local producers.
- Server events may become local presentation records, but not local authoritative resources.

Cursor durability and retention contract:

- `EventStateRepository` durably stores server cursors, stream instance IDs, dedupe records, and retention metadata scoped by source, server profile, authenticated principal, stream name, and stream instance.
- Repository state tracks processed cursor and presented high-water mark separately. Processed cursor advances only after normalized event storage succeeds; presented high-water mark advances only after presentation records are updated.
- The event insert, dedupe registration, and processed-cursor advancement for a handled event happen in one durable transaction. If the transaction rolls back, none of those three records may be partially visible; if it commits, reconnect must observe all three consistently.
- Presentation updates and presented high-water advancement happen in a separate durable transaction from processed-cursor advancement. A presentation failure may leave the event processed but unpresented, but it must not advance the presented high-water mark.
- Retention has explicit max-age and max-count bounds per stream. The initial default is 30 days or 10,000 normalized records per `(source, server_profile_id, authenticated_principal_id, stream_name, stream_instance_id)`, whichever limit prunes first, unless a domain contract sets a stricter bound.
- Pruning must preserve the latest durable cursor, the latest presentation high-water mark, and dedupe records inside the reconnect window.
- On restart, observers resume from the durable processed cursor when the server supports cursors. When the server only provides replayable events without stable cursors, the dedupe window protects presentation from duplicate events.
- Server switch closes active stream handles immediately but preserves inactive per-server cursor state unless logout, credential removal, profile deletion, or explicit user reset clears it.
- Cursor reset and replay actions must produce typed status records so UX can distinguish initial load, replay, requery, degraded dedupe-only mode, and unrecoverable cursor loss.

Event repository implementation checklist:

- The repository stores event records, dedupe records, processed cursors, presented high-water marks, retention metadata, and observer status records in separate logical tables or record classes.
- Processed cursor and presented cursor are never represented by one shared field. A processed event that fails presentation update must not advance the presented high-water mark.
- Repository APIs expose atomic operations for `record_event_and_advance_processed_cursor`, `mark_event_presented_and_advance_high_water`, `reset_stream_cursor`, and `prune_stream_state`; callers must not compose these by writing raw tables directly.
- Each event has a stable dedupe key. If the server supplies an event ID or cursor, it participates in the dedupe key; otherwise the fallback key uses source, server profile, principal, stream, event kind, entity ID, timestamp bucket, and payload hash.
- Reconnect starts from the durable processed cursor. If no stable cursor exists, reconnect starts in dedupe-only replay mode and emits a typed degraded-mode status.
- Retention pruning is bounded, deterministic, and test-covered. It must not delete the latest processed cursor, latest presented high-water mark, or dedupe records still inside the reconnect window.
- Server switch, logout, credential clearing, and profile deletion have separate effects: switch stops active observers; logout/credential clearing/profile deletion may clear scoped durable cursor and dedupe state according to the source scope.

Current implementation anchors:

- `tldw_chatbook.Notifications.event_state_repository.EventStateRepository` owns durable event records, dedupe, processed cursors, presented high-water, retention policies, observer status, stream resets, and scoped profile cleanup.
- `EventObserver` consumes an event-state protocol so the old in-memory cursor store is a test/compatibility implementation, not a second durable authority.
- Event identity models in `runtime_policy.server_parity_models` include authenticated principal in cursor, event, and dedupe scope.

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

This tranche builds sync infrastructure without enabling write sync by default. The first implementation is dry-run/read-only mirror reporting and must not enqueue mutations.

Initial dry-run components:

- `EntityIdentityMap`: source/scope key, nullable local entity reference, nullable remote entity reference, mapping status, active server profile ID, authenticated principal ID, and workspace/resource scope where relevant.
- `SyncProfileState`: per-server state for cursors, last mirror report, last error, and eligibility.
- `RemotePullCursorStore`: per-server and per-domain pull cursors.
- `MirrorReportStore`: read-only comparison reports with local/remote identity, observed versions, status, and reason codes.
- `ConflictReportPolicy`: report-only enum such as `remote_changed`, `local_dirty`, `both_changed`, `unmapped`, and `unsupported`.
- `DomainSyncEligibilityRegistry`: every domain defaults to `not_eligible`.

Identity map contract:

- The identity scope key is always non-null and includes source, server profile ID, authenticated principal ID, workspace or resource scope where relevant, domain name, and entity type.
- The local-side uniqueness key is identity scope key plus local entity ID and exists only when local entity ID is present. The remote-side uniqueness key is identity scope key plus remote entity ID and exists only when remote entity ID is present.
- The full mapping record may include both local and remote IDs, but no primary uniqueness rule may require both sides to exist because orphan/candidate states intentionally have one side missing.
- Mapping records may have a nullable local entity ID only in `candidate`, `orphaned_remote`, or `unsupported` states.
- Mapping records may have a nullable remote entity ID only in `candidate`, `orphaned_local`, or `unsupported` states.
- A `confirmed`, `stale`, or `conflict` mapping must carry both local and remote entity IDs.
- The same local-side uniqueness key cannot map to multiple remote entities unless the domain explicitly supports aliases and reports them as aliases.
- The same remote-side uniqueness key cannot map to multiple local entities unless the domain explicitly supports duplicates and reports them as duplicates.
- Mapping status is explicit: `candidate`, `confirmed`, `stale`, `conflict`, `orphaned_local`, `orphaned_remote`, or `unsupported`.
- Mirror reports must surface duplicate, stale, cross-profile, cross-principal, and cross-workspace mapping conflicts before UX treats an entity as safely mirrored.
- Mapping records are never reused across server profiles, authenticated principals, or workspace/resource scopes without a domain-specific migration rule and audit record.

Identity map implementation checklist:

- `source_scope_key`, `local_side_key`, and `remote_side_key` are separate fields or generated columns in the repository contract.
- `source_scope_key` includes at minimum `source`, `server_profile_id`, `authenticated_principal_id`, `workspace_or_resource_scope`, `domain`, and `entity_type`.
- `local_side_key` is nullable only when `local_entity_id` is nullable; `remote_side_key` is nullable only when `remote_entity_id` is nullable.
- Uniqueness is enforced independently on non-null local-side keys and non-null remote-side keys inside the same source scope.
- Duplicate local-side or remote-side mappings create conflict report entries; they are not silently collapsed, overwritten, or hidden from dry-run mirror reports.
- Orphan records state which side is missing and why: `local_deleted`, `remote_deleted`, `permission_lost`, `scope_changed`, `unsupported_domain`, or `identity_unresolved`.
- UX-facing dry-run reports consume conflict records and orphan/candidate statuses directly; they must not infer safe sync readiness from a mapping that lacks one side.

Current implementation anchors:

- `tldw_chatbook.Sync_Interop.sync_state_repository.SyncStateRepository` is the durable dry-run state store for identity mappings, conflict reports, remote pull cursors, mirror reports, and domain eligibility.
- `SyncStateRepository` intentionally has no local outbox table, replay loop, or remote mutation dispatch API.
- Identity records persist source scope keys, nullable local-side keys, nullable remote-side keys, principal scope, workspace/resource scope, mapping status, and conflict records separately.

Future write-sync components, explicitly out of scope for the initial tranche:

- `LocalOutbox` mutation enqueue and replay.
- Persisted write queue processing.
- `local_wins`, `remote_wins`, `merge`, or `manual` write-conflict resolution.
- Any background mutation dispatch to the server.

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
- The initial sync tranche has no mutation enqueue path and no persisted outbox writes.
- Workspace-scoped records preserve workspace boundaries.
- Server switching cannot replay queued operations against the wrong server.

## Tranche 4: Domain Edge Closure

After the foundations land, domain work should be narrow and source-honest. Each domain must define authority, unsupported operations, and UX-facing contracts before implementation.

Priority edges:

- Chat: source-separated history, streaming/persist handoff, server-owned sync semantics, and source-honest chat request execution/persistence semantics. This excludes workflow orchestration, scheduler behavior, and multi-step automation.
- Media/Reading: ingest-job event/status normalization, chunk-level TTS playback adoption, and saved-view support only where the server contract exists.
- Notes/Workspaces: graph semantics, local graph unsupported reports, workspace-aware sync rules, and deferred cross-scope moves.
- Writing: source-honest server-only analysis operations and optional local richer analysis only if a local engine is approved.
- Research: run, event, bundle, and artifact consistency; listing constraints; local/remote session vocabulary alignment.
- Study/Evaluations: result artifact normalization and target catalog discovery.
- RAG/Embeddings: local per-media embedding admin decision and server collection export contract.
- Audio/Voice: WebSocket transport slices, speech job/artifact parity, and source-honest history.
- Remote-only utilities: sharing, web clipper, translation, server tools, Text2SQL, server skills, claims, meetings, outputs, Kanban, and Prompt Studio stay active-server owned unless separately approved for local parity.

Remote-only minimum contract:

- Every remote-only utility must report unsupported local/offline behavior through the same reason-code shape.
- Required reason codes include `server_required`, `server_unavailable`, `auth_required`, `permission_denied`, `capability_missing`, and `not_implemented_locally`.
- UX may render disabled or unavailable states from these codes without calling the underlying adapter.

Rules:

- Each edge has an explicit source authority decision.
- Each unsupported edge is reported through the scope-service unsupported-capability seam.
- Domain services consume connection/auth, event, and sync seams rather than inventing their own.
- Tests are service, scope, policy, and API-client tests. UI tests are not required while the UI layer is being rewritten.

Current implementation anchors:

- `tldw_chatbook.runtime_policy.domain_edge_contracts` exposes the backend-owned per-domain capability matrix, source selector states, workspace isolation hints, view-model contract IDs, and unsupported-action report builder.
- Remote-only domains use the shared unsupported-capability report shape and required reason codes before UX calls a domain adapter.
- Domain edge contracts are additive backend contracts for UX handoff; they do not depend on the current UI implementation.

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

Contract format rules:

- Each handoff contract is a versioned typed schema.
- Each schema includes owner, stability level, compatibility rules, and example payloads.
- Reason-code enums are explicit and shared across domains where possible.
- Service-level contract tests pin each schema before UX work depends on it.
- Breaking contract changes require a version bump and migration note in the handoff packet.

The UX developer should consume this packet and service-level contracts, not current UI implementation details.

## Execution Model

Execution order:

1. Land Connection/Auth.
2. Start Realtime/Notifications only after the Tranche 1 readiness gate passes.
3. Start Sync/Mirror only after the Tranche 1 gate and the event identity subset of the Tranche 2 gate pass.
4. Run domain edge closures in parallel once shared seams are stable.
5. Update the UX handoff packet after each tranche.

Readiness gates:

- A tranche is ready for dependents only when its typed contracts are committed, service-level contract tests are green, and shared schema owners have signed off.
- Tranche 1 additionally requires no new raw `tldw_api` client builders outside audited compatibility adapters, server-switch invalidation tests passing, credential redaction tests passing, and the app-wiring owner approving startup/logout/server-switch integration points.
- The event identity subset of Tranche 2 requires normalized event schema, stream identity schema, cursor/dedupe repository contract, and cancellation/reconnect tests before Sync/Mirror consumes event state.
- Domain edge work may begin before full tranche completion only as spec, adapter-interface, or additive-test scaffolding. It must not introduce production code that consumes unstable shared seams.
- Shared-file edits route through the assigned integration owner for that tranche; parallel contributors hand off deltas instead of editing shared audit, schema, or app-wiring files directly.

Objective gate checklist:

- Tranche 1 dependent work may consume connection/auth seams only after the credential lifecycle tests, server-context provider tests, runtime-policy bootstrap tests, capability status tests, UX connection-contract tests, and provider migration audit tests pass in the committed branch.
- The Tranche 1 app-wiring owner must approve the exact startup, logout, credential-clear, and server-switch invalidation blocks before dependent production code can use those hooks.
- The provider migration audit must report zero unaudited raw client builders and must use semantic audit keys, not line-number-only allowlists.
- The shared contract/schema owner must confirm the exported connection/auth contract module names, reason-code enums, and version numbers are final for the tranche.
- Tranche 2 dependent work may consume event seams only after `NormalizedEventRecord`, stream identity, cursor, dedupe, processed-cursor, presented-cursor, retention, and reconnect semantics have contract tests in the committed branch.
- Tranche 3 dependent work may consume sync seams only after identity-map uniqueness, orphan/candidate/null-side mapping states, dry-run mirror reports, and write-outbox-disabled behavior have contract tests in the committed branch.
- A gate is not satisfied by prose alone. Each gate requires committed contracts, committed tests, passing focused verification, and no unresolved ownership conflicts in shared files.

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
