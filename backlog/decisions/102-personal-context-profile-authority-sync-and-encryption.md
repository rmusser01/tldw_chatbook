# ADR-102: Unify encrypted Personal Context Profile authority across Chatbook and tldw_server

Status: Accepted
Date: 2026-08-28

## Context

Chatbook and tldw_server already expose personalization features, but they do
not share one durable authority for user-owned context. Treating those stores
as equivalent projections would allow records to lose identity, provenance,
privacy controls, lifecycle state, or conflict history as they move between
runtimes. It would also leave room for a local CRUD path and a server CRUD path
to race the synchronization path.

The profile contains sensitive user-authored and inferred material. It must be
usable by an offline Chatbook installation, readable by authorized server-side
agents when the user chooses to synchronize it, inspectable and removable by
the user, and protected against plaintext persistence and stale-device
resurrection. Those requirements establish long-lived boundaries for data
ownership, synchronization, service access, encryption, migration, agent
authority, and deletion.

The existing Chatbook `Personalization_Interop` integration remains a
server-only compatibility seam. It cannot become the local canonical store.
Personas, roleplay characters, assistant state, authentication identity,
Companion tracked goals, and Persona/session memory have distinct owners and
must not be absorbed into a human Personal Context Profile.

## Decision

Chatbook and tldw_server are peer replicas of one canonical Personal Context
Profile contract published by `tldw-profile-core`. Chatbook is local-first;
tldw_server is the single V1 home peer. Each runtime owns one encrypted
repository and one mutation service. Sync V2 transports canonical whole
objects and never application-specific projections.

Profile records use global or random canonical workspace scopes, separate
proposal and conflict objects, per-record syncability and agent visibility,
runtime-local agent grants, server-trusted Sync payloads over TLS, per-version
envelope encryption at rest, keyed integrity tags, and a purge-generation
barrier. Personas, auth identity, Companion goals, and Persona/session memory
remain separate authorities.

The following constraints make that decision operational:

- `tldw-profile-core` is a separately versioned, pinned contract package. It
  owns canonical models, lifecycle validation, deterministic serialization,
  JSON Schema, and conformance fixtures, but not databases, encryption-key
  custody, network clients, provider calls, application configuration, or UI.
- Chatbook and tldw_server each expose exactly one `PersonalContextService` as
  the authorized mutation boundary. Repositories own encrypted persistence and
  transactions; consumers may not decrypt or mutate profile tables directly.
  A separate read-only `ProfileContextService` creates immutable, budgeted
  request snapshots.
- Chatbook stores the profile in a dedicated Personal Context SQLite database.
  tldw_server extends each user's existing `Personalization.db` with canonical
  encrypted tables. Each peer stores the same canonical bytes and stable
  object IDs while encrypting them under its own key hierarchy.
- Every profile has one global scope and random canonical workspace scopes.
  Application workspace IDs and readable labels stay in encrypted,
  peer-local mappings. An unmapped incoming workspace scope remains unlinked
  and is unavailable to agents until the user maps it.
- Canonical facts are granular, typed, immutable-version records. Pending
  suggestions remain separate `ProfileProposal` objects and never enter model
  context. Sync conflicts remain durable Sync V2 review objects rather than
  record lifecycle states; conflicted context uses the last mutually
  acknowledged version or omits the record.
- Privacy is expressed per record. `device_only` records never leave Chatbook.
  `user_only` records may synchronize for the user but are excluded from every
  agent-facing path. Runtime enablement and `read_only`, `propose`, or
  `direct_write` agent grants are peer-local and never synchronize.
- Personal Context extends Sync V2 with negotiated whole-object manifest,
  scope, record, proposal, and content-free purge-barrier domains. Chatbook
  commits a canonical immutable revision and its encrypted exact-wire outbox
  snapshot atomically. Direct server personalization CRUD is never a second
  Chatbook write path.
- Syncable payloads use the existing `server_trusted_v1` posture: authenticated
  clients decrypt locally, send canonical payloads over TLS to the authorized
  home server, and each peer re-encrypts at rest. HMAC-SHA-256 integrity tags
  occupy the negotiated object-hash position and detect canonical-byte
  differences without serving as signatures or authorization.
- Each content-bearing version or artifact receives a random DEK and
  AES-256-GCM envelope. Associated data binds the peer envelope, object type,
  object ID, version ID, and serialized schema version. Per-profile keys wrap
  DEKs. Chatbook protects its profile key with an OS keyring or a
  scrypt-derived passphrase wrapper; the server uses a configured master key
  or KMS-backed protector. There is no plaintext-key fallback.
- A monotonic `purge_generation` is the deletion fence. Delete-everywhere
  advances the generation, destroys readable canonical and derived content,
  distributes a content-free barrier, and rejects every older-generation
  write. Removing one device's copy is a distinct operation and does not claim
  to erase offline or stolen devices remotely.

## Alternatives considered

### Keep separate Chatbook and server personalization models

Rejected because translations between application-specific projections would
lose canonical identity or controls, make conformance impossible to prove, and
permit divergent meanings for an apparently shared fact.

### Make tldw_server the only V1 authority

Rejected because Chatbook must create, inspect, edit, and use a profile while
offline or without any server. A home server is a peer for syncable records,
not a prerequisite for local personalization.

### Put canonical models, storage, crypto, and transport in one shared package

Rejected because key custody, persistence, authentication, provider use, and
UI are runtime responsibilities. Sharing them would couple release and threat
boundaries. Only the deterministic contract belongs in Shared Profile Core.

### Reuse Chatbook configuration or `ChaChaNotes_DB`

Rejected because TOML cannot provide the required transactional record,
proposal, conflict, outbox, purge, and encryption lifecycle, while the main
conversation database has a different authority and migration surface.

### Synchronize summaries or compatibility projections

Rejected because summaries are derived views and legacy personalization shapes
cannot represent the complete canonical record losslessly. Sync transports the
canonical whole object; compatibility APIs project through the canonical
service and fail when a legacy mutation cannot be represented without loss.

### Use last-write-wins or make conflicts record states

Rejected because concurrent edits and same-key creations can silently erase
valid user choices. Conflicts require both immutable heads and a durable server
token, and remain distinct review objects so ordinary mutation cannot bypass
resolution.

### End-to-end encrypt every synchronized record from the home server

Rejected for V1 because authorized server-side agents must read eligible
syncable content. The product discloses server readability, uses TLS in
transit, and encrypts independently at both peers. `device_only` remains the
option for content that must never reach the server.

### Synchronize runtime agent permissions

Rejected because a grant is local execution policy, not user profile content.
Each runtime independently intersects its local grant with enablement, scope,
lifecycle, and record privacy controls.

### Delete rows without a generation fence

Rejected because a stale or dormant device could replay an older cursor and
resurrect deleted profile content. Tombstones cover record deletion; the
profile purge generation fences whole-profile deletion.

### Fold Personas, authentication, Companion goals, or session memory into the profile

Rejected because those domains have distinct human/assistant identity,
execution, retention, and authority contracts. Conflating them would recreate
the Persona/User Profile inversion rejected by ADR-037.

## Consequences

- Both applications must pin compatible Shared Profile Core releases and test
  the oldest and newest schema/package combinations they advertise.
- Every settings action, interview commit, agent tool, context read, API
  handler, sync adapter, compatibility adapter, and migration must use the
  runtime's one canonical service.
- Chatbook can deliver a useful local profile before server work exists, while
  stable IDs and canonical bytes preserve a later multi-device sync path.
- Initial server linking requires an encrypted, cancellable reconciliation and
  an atomic provisional-to-canonical profile rebind before normal push/pull.
- Restrictive privacy changes take effect locally immediately but are not
  fully acknowledged until agent indexes, caches, summaries, and other derived
  artifacts have been cleaned.
- Unknown newer objects are retained opaquely. Older runtimes may not decrypt,
  edit, index, inject, or approve them.
- Context construction is an immutable per-request snapshot. Profile context
  is escaped user-owned data, budgeted below the current request, and never
  system or safety authority.
- V1 deliberately accepts operational complexity for encrypted version
  envelopes, key custody, cleanup acknowledgements, conflict review, initial
  reconciliation, and purge fencing in exchange for one inspectable authority
  and conservative privacy behavior.

## Security/privacy consequences

- Profile values, labels, kinds, provenance, drafts, proposals, conflicts,
  outbox snapshots, Undo before-images, and recovery artifacts are encrypted at
  rest. Logs and diagnostics contain only bounded reason codes, opaque IDs,
  counts, and content-free summaries.
- Encryption does not hide ciphertext sizes, random object/scope IDs, revision
  frequency, scope relationships, tombstones, or timing. Authorized home
  servers can read syncable content; the UI must say so plainly.
- Authentication and authorization occur before decryption. Missing or changed
  key protectors lock the profile and never trigger silent replacement.
- Crypto-shredding destroys wrapped DEKs for retired content, but cannot claim
  immediate Python-memory zeroization or erase historical/external plaintext
  backups created before migration.
- `user_only` applies to automatic context, search, tools, summaries, derived
  indexes, and background agent jobs. Private duplicate checks reveal only that
  user review is required, not the private record's existence or value.
- Interviews and agent boundaries reject credentials and other recognized
  secret material. The profile is not a credential or general secret vault.
- Verification must scan ordinary databases, WAL, outboxes, logs, diagnostics,
  caches, migration snapshots, and application-owned backups with unique
  canaries; this is regression evidence, not proof that plaintext never
  existed in process memory or external snapshots.

## Migration/rollback

tldw_server migrates one user at a time behind a legacy-write fence. It creates
a bounded encrypted recovery snapshot, performs an idempotent canonical
backfill, preserves stable IDs and applicable controls, classifies uncertain
memories without guessing, validates canonical and compatibility projections,
and atomically switches legacy routes to the canonical service. Shadow reads
are permitted; dual writes are not. Existing Chatbook server-personalization
surfaces become compatibility consumers until they can be retired.

After plaintext legacy authority is retired, migration is forward-only.
Rollback restores an encrypted canonical snapshot under compatible software or
ships a forward fix; it never re-enables an old binary as a second authority.
Deployment must inventory and expire legacy plaintext backups because the new
encryption layer cannot retroactively protect copies outside its control.

Feature rollout remains gated and phased: contract and decision, Chatbook local
profile, interview and controlled learning, server canonical store, Sync and
multi-device behavior, then legacy retirement. An implementation may be
disabled at runtime without deleting data or silently stopping synchronization.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md)
- [Implementation index](../../Docs/superpowers/plans/2026-08-28-personal-context-implementation-index.md)
- [Backlog task 23193](../tasks/task-23193%20-%20Record-personal-context-profile-authority-ADR.md)
- [ADR-008 — Sync V2 client M1 contract alignment](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-037 — Persona and User Profile separation](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
