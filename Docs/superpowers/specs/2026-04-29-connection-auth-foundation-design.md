# Connection And Auth Foundation Design

Date: 2026-04-29

Status: Draft for review

Related docs:

- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`
- `Docs/Development/server-client-provider-migration-audit.md`
- `Docs/Parity/2026-04-21-capability-matrix.md`
- `Docs/Parity/2026-04-21-gap-ledger.md`

## Purpose

This design defines the connection/auth hardening tranche for Chatbook's standalone-client-to-`tldw_server` model.

The base seams already exist in code:

- `runtime_policy.bootstrap.RuntimePolicyContext`
- `MCP.server_target_store.ConfiguredServerTargetStore`
- `runtime_policy.server_context.RuntimeServerContextProvider`
- `runtime_policy.server_capabilities.ActiveServerCapabilityService`
- `runtime_policy.server_credentials.KeyringServerCredentialStore`

This tranche does not replace those seams. It hardens them into one authoritative connected-server foundation with explicit credential, cache invalidation, legacy-import, and migration rules that later parity lanes must follow.

## Goals

- Keep runtime-policy active-server state as the only authoritative active-server selection.
- Keep server profile metadata and secret material separated.
- Provide durable cross-platform secure credential storage for macOS, Linux, and Windows with an in-memory test backend.
- Support one active server now while preserving explicit server switching behavior.
- Make provider-built client invalidation deterministic on logout, credential changes, and active-server switches.
- Preserve staged migration away from direct legacy `tldw_api` config builders.
- Fail closed when secure credential resolution or active-server context is invalid.

## Non-Goals

- Do not introduce a second server/profile registry.
- Do not redesign UI auth/server selectors.
- Do not enable write sync.
- Do not implement workflows.
- Do not make local mode depend on server presence.
- Do not persist bearer, API, refresh, OAuth, BYOK, or session secrets in JSON config or target-store metadata.

## Authority Baseline

These seams remain authoritative and must be extended rather than replaced:

- `RuntimePolicyContext` owns `active_source`, `active_server_id`, reachability state, and auth probe state.
- `ConfiguredServerTargetStore` owns persisted server profile metadata and default-target resolution.
- `RuntimeServerContextProvider` owns active server context resolution, credential-bound client construction, compatibility fallback, and provider-client cache invalidation.
- `ActiveServerCapabilityService` owns active-server capability snapshots.

This tranche must not add:

- a second selected-server registry
- a second active capability snapshot authority
- domain-owned credential caches
- domain-owned provider-client caches keyed independently from the runtime server context
- secret-bearing fields in server profile JSON

Any future workstream that needs server identity, credentials, or clients must name which baseline seam it uses.

## Existing Implementation Baseline

The current code already provides:

- runtime-policy state carrying active server identity
- persisted configured targets through `ConfiguredServerTargetStore`
- provider-backed client construction through `RuntimeServerContextProvider`
- keyring-backed and in-memory credential stores
- capability refresh through `ActiveServerCapabilityService`
- compatibility fallback from legacy `tldw_api` config

The remaining work is hardening and normalization, not first-time invention.

## Architecture

### 1. Server Profile Metadata

`ConfiguredServerTargetStore` remains the persisted registry for server profile metadata.

Rules:

- Target-store records may include server ID, label, base URL, auth mode, default status, last-known status metadata, and non-secret auth references.
- Target-store records must not include secret values.
- One active server remains the runtime behavior for now, but profile storage and switching rules must assume multiple configured servers can exist.
- The active profile is selected by runtime-policy state, not by hidden target-store side effects.

### 2. Secure Credential Storage

`KeyringServerCredentialStore` is the durable credential backend seam.

Rules:

- The default implementation must use OS-backed secure storage through Python keyring-compatible backends so the same seam works on macOS, Windows, and Linux.
- CI and isolated tests use `InMemoryServerCredentialStore`.
- The credential-store seam must support both per-profile clearing and global clearing. If the backing keyring cannot enumerate entries directly, Chatbook must maintain a non-secret credential-reference index sufficient to delete all stored server credential entries during global sign-out.
- Secrets are stored per `server_id` and per `purpose`.
- Minimum credential purposes:
  - `access_token`
  - `refresh_token`
  - `api_key`
  - `bearer_token`
- Secret material must never be written into:
  - target-store JSON
  - exported settings JSON
  - migration-audit files
  - unsupported-capability reports
  - cache keys
  - logs or exception strings

Failure handling:

- If the secure store cannot be read or written, server-authenticated operations fail closed with typed credential-store errors.
- Local-mode behavior must remain unaffected by secure-store failures.

### 3. Legacy Credential Import

Legacy `tldw_api` config remains a compatibility input, but it is no longer the target-state authority.

Rules:

- Legacy tokens are imported only for the currently active server profile.
- Import resolves from legacy config into secure storage on first authenticated use or explicit auth bootstrap.
- After successful import, secure storage becomes authoritative for that active profile.
- Legacy config values may remain physically present until a later cleanup pass, but runtime resolution should prefer secure-store credentials once imported.
- Import must not create credentials for inactive or unrelated server profiles.

### 4. Runtime Server Context Resolution

`RuntimeServerContextProvider` remains the only allowed seam for active server context and provider-built clients.

Rules:

- Domain services must obtain server clients through the provider or through explicitly audited compatibility wrappers that delegate to it.
- The provider cache key must remain scoped by:
  - active server ID
  - base URL
  - auth method
  - credential source
  - token fingerprint
- No domain service may keep its own long-lived credential-bound client cache independent of the provider.

Typed failure cases:

- no active configured server
- active server metadata missing
- secure credential store unavailable
- required credentials missing
- active server unreachable
- active server auth/session invalid

### 5. Invalidation And Global Sign-Out

This tranche treats server switching and logout as hard invalidation boundaries.

Server switch must invalidate:

- provider-built cached clients
- active-server capability snapshots
- active remote selections derived from prior server context
- event cursors for the prior active server
- remote sync cursors for the prior active server
- any cached auth/session state derived from the prior server profile

Global sign-out must:

- clear all stored server credentials across all profiles, including orphaned entries whose profile metadata may already be gone
- clear access and refresh tokens
- invalidate provider-built cached clients
- force subsequent capability/auth checks to re-resolve through runtime-policy and secure-store paths

Per-profile credential clearing must:

- clear only the selected profile's stored credentials
- invalidate any provider cache entries that could reuse those credentials

### 6. Capability Snapshot Ownership

`ActiveServerCapabilityService` remains the single active-server capability snapshot seam.

Rules:

- Capability snapshots must read active-server identity from runtime-policy state.
- Capability snapshots must not create their own selected-server memory.
- Snapshot refresh must operate on the provider-resolved active server context or typed unavailable/auth-required results.
- Snapshot results should remain source-honest and must be invalidated on active-server switch and global sign-out.

### 7. Migration Compatibility Layer

This tranche does not require a big-bang service rewrite.

Rules:

- Unmigrated services may continue through compatibility factories only if they are listed in `Docs/Development/server-client-provider-migration-audit.md`.
- New direct reads from raw `tldw_api` config are forbidden.
- New server-backed services must use `RuntimeServerContextProvider` directly from day one.
- Existing compatibility factories must gradually collapse behind the provider seam in audited batches.

Audit rules:

- The migration audit must track:
  - migrated provider-backed services
  - remaining compatibility factories
  - intentional bootstrap seams
  - excluded UI/event helper locations, if any
- Audit enforcement must use stable semantic matching, not line-number-only matching.
- CI should fail on newly introduced unlisted legacy builders or compatibility factories.

## Execution Plan

### Tranche 1: Credential And Invalidation Hardening

Deliverables:

- provider invalidation coverage for active-server switch
- global sign-out behavior for all stored credentials
- explicit per-profile credential clearing behavior
- typed credential-store failure handling
- legacy credential import rules enforced for the active profile only

### Tranche 2: Capability And Auth Normalization

Deliverables:

- capability snapshot invalidation on server switch and sign-out
- normalized auth-state transitions
- consistent unavailable/auth-required/session-invalid error surfaces
- target-store status update rules aligned with runtime-policy state

### Tranche 3: Compatibility Migration Guard

Deliverables:

- refreshed migration audit entries
- audit-backed compatibility list
- no new direct config-based builders
- service migration batch plan for remaining holdouts

## Testing

Required tests:

- keyring-store adapter tests with fake keyring backend
- in-memory credential-store tests
- no-secret-in-target-store serialization tests
- active-profile-only legacy import tests
- global credential clearing tests, including orphaned-profile secret cleanup
- provider cache invalidation on:
  - active-server switch
  - credential update
  - per-profile credential clear
  - global sign-out
- capability snapshot invalidation tests
- typed failure tests for missing server, missing credential, unavailable credential store, unreachable server, and auth/session invalid states
- migration-audit guard tests proving semantic baseline matching

Required negative tests:

- switching from server A to server B must not reuse server A credentials
- global sign-out must remove credentials for every configured profile
- exporting or serializing target metadata must not leak secrets
- logs and exceptions must redact credential material
- unmigrated services must not silently bypass the audit

## Acceptance Criteria

- Runtime-policy state remains the only active-server authority.
- `ConfiguredServerTargetStore` remains metadata-only and secret-free.
- `RuntimeServerContextProvider` remains the only allowed server-client construction seam for new work.
- Durable secure credential storage works through a cross-platform keyring-backed seam, with in-memory CI/test support.
- Legacy config credentials import only into the currently active server profile and stop being runtime-authoritative after import.
- Global sign-out clears credentials for all configured server profiles and invalidates provider-built clients.
- Active-server switch invalidates provider clients, capability snapshots, event cursors, and remote sync cursors tied to the previous server.
- Migration away from direct config builders remains staged, audited, and fail-on-new-regression.

## Open Follow-Ons

- refresh-token refresh scheduling and retry policy details
- explicit server-profile delete UX and confirmation rules
- credential scrub/migration of physical legacy config files
- multi-user server account switching semantics under one server profile
- UI contract updates for auth/bootstrap status presentation
