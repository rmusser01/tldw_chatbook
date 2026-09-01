# Unified Personal Context Profile — Chatbook and tldw_server

**Date:** 2026-08-28

**Status:** Draft for owner review

**Initial product:** tldw_chatbook

**Shared consumers:** tldw_chatbook and tldw_server

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: This design establishes a new encrypted data authority, schema and
migration contract, Sync V2 domains and conflict policy, cross-runtime service
boundary, agent permission model, and long-lived setup/settings UX.
```

ADR-102 was selected after sweeping local decisions, active worktrees, and
fetched refs because this repository has concurrent decision work and
historical duplicate numbers.

## Summary

Chatbook and tldw_server will share one logical **Personal Context Profile** for
the human user. The profile contains granular, user-owned records such as
preferences, likes and dislikes, corrections, constraints, relationships,
goals, conventions, and bounded working context. Agents may read or propose
changes only through explicit runtime-local permissions.

Chatbook is local-first. It can create and use a profile without a server. When
the user links a home tldw_server, Chatbook and the server become replicas of
the same canonical records through Sync V2. A record has the same stable ID,
scope, meaning, lifecycle, privacy controls, provenance, and revision on both
systems. The databases remain separate transactional stores; they do not hold
application-specific projections of supposedly equivalent facts.

The first-run flow optionally offers a bounded, 20-question "get to know you"
interview after ordinary application setup has completed. Workspace creation
can likewise offer an interview about the project's purpose and durable
context. Settings provides complete inspection, editing, re-interview,
proposal review, local removal, and global purge controls.

All profile content is encrypted at rest in both runtimes. Syncable records use
the existing server-trusted Sync posture because server-side agents must be
able to read eligible content. Device-only records never leave Chatbook.

## Goals

- Give agents bounded, accurate knowledge of a user's explicit preferences and
  durable context without turning conversation history into hidden memory.
- Let the user inspect, edit, reject, archive, export, and delete everything
  that can influence personalization.
- Support personal context and workspace-specific context without leaking one
  workspace into another.
- Make Chatbook and tldw_server replicas of the same logical records rather
  than maintaining incompatible personalization stores.
- Support multiple Chatbook devices linked to one home server.
- Permit agent learning only through explicit tools and local runtime policy.
- Preserve conservative review for inferred facts and conflicts.
- Encrypt profile content, proposals, drafts, conflicts, outbox snapshots, and
  recovery artifacts at rest.
- Provide deterministic, budgeted context construction that never outranks the
  user's current request.

## Success criteria

- A new Chatbook user can skip setup personalization or complete at most 20
  questions, review the resulting diff, and create a usable local profile.
- A user can create and manage equivalent workspace context without exposing it
  in unrelated workspaces.
- Settings exposes all canonical profile records and every pending proposal or
  conflict relevant to the active installation.
- Two Chatbook devices and one home server converge on identical syncable
  records after non-conflicting edits.
- Conflicting edits never silently overwrite one another.
- Device-only and user-only controls are enforced across storage, search,
  context injection, tools, derived artifacts, and synchronization.
- Agents cannot read or mutate profile data outside their effective local
  runtime authority.
- A global purge prevents stale devices from resurrecting older content.
- Unique canary profile text does not appear in ordinary database, WAL, outbox,
  log, diagnostic, or backup artifacts covered by automated tests.

## Terminology

- **Personal Context Profile:** The technical domain and the user-owned set of
  canonical context records. The UI may shorten this to **My Profile**.
- **Profile manifest:** Stable logical identity, schema version, and global
  purge generation for a profile.
- **Profile record:** An active, archived, or tombstoned canonical fact.
- **Profile proposal:** A suggested create, update, archive, or promotion that
  cannot affect context until a user accepts it.
- **Global scope:** Context available outside a specific workspace when other
  controls permit it.
- **Workspace scope:** Context available only while its mapped workspace is
  active.
- **Home server:** The one tldw_server authority to which a V1 Chatbook profile
  may be linked.
- **Runtime authority:** Local read/propose/direct-write permission configured
  independently in each Chatbook or server runtime.
- **User-only:** A record that may sync for backup and user editing but is
  unavailable to all agent reads, tools, search, and generated agent context.
- **Device-only:** A Chatbook record that never enters an outbox or server
  payload.

## Owner decisions

1. Chatbook is local-first and both Chatbook and tldw_server may write canonical
   records.
2. One global base is combined with one active-workspace overlay. Workspace
   context never leaks or promotes automatically.
3. Agent authority is configured per runtime and scope as `read_only`,
   `propose`, or `direct_write`; the default is `propose`.
4. Canonical data is granular records. Overviews are generated projections and
   never become authority.
5. Interviews are adaptive but bounded to 20 questions and finish with a
   record-by-record review.
6. V1 links to one home server while keeping authority and identifiers suitable
   for future federation.
7. Model requests receive a bounded overview plus relevant records and optional
   read tools.
8. Privacy is controlled per record through syncability and agent visibility.
9. Whole-profile deletion has distinct local-copy and delete-everywhere flows.
10. Raw interview Q&A is discarded after approved records are committed.
11. V1 has one human profile per Chatbook installation.
12. A narrow Shared Profile Core publishes the canonical contract for both
    applications.
13. Adaptive interviews may use a configured model only after disclosing its
    provider and model; a fixed local questionnaire is always available.
14. Multiple Chatbook devices may link to the same home-server profile.
15. All profile content is encrypted at rest in both applications.
16. Personas, roleplay characters, assistant state, authentication identity,
    and Companion tracked goals remain separate domains.

## Non-goals for V1

- Multiple home servers, federation, or account merge/split behavior.
- End-to-end encryption of syncable data that must be readable by server-side
  agents.
- Semantic embeddings, vector search, or RAG reranking over profile data.
- Autonomous consolidation, background inference, or automatic promotion of
  workspace context.
- Synchronizing runtime agent permissions.
- Cross-workspace agent browsing.
- Allowing agents to change privacy controls, approve their own proposals,
  purge a profile, or permanently delete records.
- Replacing assistant Personas, roleplay characters, Persona state documents,
  Persona/session memory, authentication accounts, or Companion goal tracking.
- Retaining raw interview transcripts as a long-term personalization source.
- Using the profile as a password, credential, API-key, or general secret
  vault.

## Architecture

### Shared Profile Core

A separately versioned, standalone `tldw-profile-core` package is the canonical
contract. Both applications install a pinned compatible release; the same
release publishes its JSON Schema and conformance fixtures. It contains:

- Profile manifest, scope, record, proposal, tool request/result, and provenance
  models.
- Typed payload schemas and lifecycle validation.
- Canonical serialization and deterministic comparison rules.
- Versioned question packs, interview coverage rules, and proposal-output
  schemas.
- JSON Schema and cross-language conformance fixtures.
- Supported serialized-schema ranges independent of package SemVer.

It does **not** contain database access, HTTP clients or routes, encryption key
custody, provider calls, prompts, UI code, runtime permissions, application
configuration, or application workspace IDs.

Both repositories test the oldest and newest Shared Core versions they claim
to support. TypeScript consumers validate payloads at runtime against the
published JSON Schema; generated compile-time types alone are insufficient.

### Per-runtime service boundary

Chatbook and tldw_server each own exactly one `PersonalContextService`. Every
settings action, interview commit, agent tool, context read, API handler, sync
adapter, compatibility adapter, and migration uses this service. No consumer
may decrypt or mutate profile tables directly.

The service owns:

- Authentication and effective-scope checks.
- Runtime-local read/propose/direct-write authorization.
- Canonical validation and lifecycle transitions.
- Encryption/decryption repository access.
- Record and proposal mutation.
- Atomic local revision/outbox transactions.
- Conflict, cleanup, migration, and purge fencing.
- Content-minimized, peer-local activity receipts.

`ProfileContextService` is a separate read-only component. It receives only
authorized repository views and creates immutable snapshots for model
requests. It cannot mutate records, proposals, or policy.

Chatbook stores the profile in a dedicated Personal Context SQLite database,
not `ChaChaNotes_DB` and not configuration TOML. Its repository owns manifests,
scopes, immutable record versions, proposals, encrypted drafts, local scope
mappings, runtime authority, Undo before-images, outbox snapshots, quarantine,
and peer-local receipts. tldw_server extends the existing per-user
`Personalization.db` with canonical encrypted tables rather than creating a
second personalization database.

### Existing Chatbook seam

The current `Personalization_Interop` seam is explicitly server-only. It does
not become the local canonical store. New local-first Personal Context modules
own the canonical service and repository. Existing server personalization UI
or API calls become compatibility consumers during migration and are removed
after deprecation; they must never form a second write path beside Sync.

### One logical record, separate replicas

A syncable `ProfileRecord` uses the same canonical bytes and stable object ID in
Chatbook and tldw_server. Each peer encrypts those bytes under its own at-rest
key hierarchy. Application databases may use different physical tables and
indexes, but neither application translates the record into a weaker local
schema.

Device-only data is the only intentional replication exception. It remains a
canonical record locally but has no shared-server representation.

## Canonical data model

### ProfileManifest

The manifest contains:

- `profile_id`: the canonical random logical profile ID after server binding.
- `schema_version`: the serialized contract version, separate from package
  SemVer.
- `purge_generation`: a monotonically increasing content-free generation.
- Creation and current-version metadata.

A standalone Chatbook profile initially has a provisional local manifest ID.
First server linking reconciles its records and atomically adopts the server's
canonical profile identity. The provisional ID remains only in encrypted local
lineage metadata needed to make interrupted linking idempotent.

### ProfileScope

Every record belongs to a random `profile_scope_id`:

- Exactly one global scope exists per profile.
- Each linked workspace has its own workspace scope.
- Human-readable scope labels are encrypted.
- Chatbook and tldw_server keep peer-local mappings from canonical scope IDs to
  their application workspace IDs.
- Application workspace IDs never enter canonical records.

An incoming workspace scope without a local mapping is retained as **Unlinked
workspace context**. It cannot be injected, searched by an agent, or edited
until the user explicitly maps it.

### ProfileRecord

A record contains:

- **Identity:** profile ID, record ID, scope ID, and immutable kind.
- **Meaning:** typed payload, structured semantic key where the kind supports
  one, and optional polarity for likes/dislikes.
- **Lifecycle:** `active`, `archived`, or content-free `deleted` tombstone.
- **Controls:** `syncable` or `device_only`; `agent_visible` or `user_only`.
- **Provenance:** bounded source enum, actor type, reason code, source
  references/hashes, optional `derived_from` record ID, and current revision
  history links.
- **Concurrency:** immutable version ID, parent version ID, canonical integrity
  tag, and timestamps.
- **Retention:** optional expiry for working context. No expiry requires an
  explicit `no_expiry` decision.

Record payloads are discriminated, versioned models rather than arbitrary
JSON. The initial kinds are:

- `identity`
- `preference`
- `relationship`
- `correction`
- `constraint`
- `goal`
- `convention`
- `working_context`

`preference` represents likes and dislikes through a typed subject and
polarity. Workspace `goal` records are declarative project context; they are
not Companion tracked-goal objects.

Confidence does not belong on an active fact. It is meaningful only while an
inferred proposal awaits review.

New object IDs are random UUIDv4 values. IDs migrated from the existing server
remain bounded opaque strings so migration can preserve them exactly;
consumers must not parse either form for meaning.

### Semantic keys and overrides

Where a record kind has a structured semantic key, at most one active record
with that kind and key may exist within one scope. The active workspace record
overrides a global record with the same key. It does not delete or mutate the
global record. Other global context remains available.

The local service prevents ordinary same-scope key duplication. Concurrent
creations on separate peers can still produce different record IDs with the
same key; the server creates a durable `key_collision` review object. Context
keeps the last mutually acknowledged occupant of that key when one exists and
otherwise omits both records until resolution. Text that merely appears
semantically similar is shown as a possible duplicate and is never auto-merged
in V1.

Changing scope is not an in-place mutation. Workspace-to-global promotion
creates a new global record with a new ID and `derived_from` provenance. The
workspace record remains unless the user explicitly archives it.

### ProfileProposal

Proposals are separate from canonical facts so a suggested update cannot hide
or replace an active record before approval. A proposal contains:

- Proposal ID and target scope.
- Operation: create, update, archive, or promote.
- Proposed typed payload and controls.
- Target record and base version for updates.
- Bounded provenance and optional confidence for inference.
- State: `pending`, `accepted`, `rejected`, `superseded`, or `expired`.

Pending proposals are never injected into model context. Acceptance applies a
normal canonical mutation and records the proposal receipt. Pending agent
proposals expire after 90 days. Accepted, rejected, expired, and superseded
proposal content is crypto-shredded as part of the resolving transaction; only
a content-free receipt remains.

Interview diffs are not `ProfileProposal` objects. They remain within the
encrypted interview draft and create active records only when the user approves
the final review.

### Conflicts

Sync conflicts are review objects owned by the Sync V2 conflict mechanism, not
record lifecycle states. A conflict identifies both immutable head versions,
the last mutually acknowledged version when one exists, and a durable server
conflict token.

Conflict reasons include `same_record_revision` and `key_collision`. A key
collision may reference two different record IDs in the same scope. The
global/workspace override relationship is not a collision because the records
belong to different scopes.

While conflicted, the record is frozen for ordinary edits. Context uses the
last mutually acknowledged active version or omits the record if no such
version exists. Resolution submits the conflict token and both head IDs; it is
not an ordinary one-head upsert.

Deletion defeats stale or concurrent edits. A change from `agent_visible` to
`user_only` also wins conservatively while content reconciliation is pending,
so a concurrent edit cannot re-expose it.

## Privacy and lifecycle rules

### Agent visibility

`user_only` governs every agent-facing path: automatic context, summaries,
search, get tools, proposal matching, derived indexes, and background agent
jobs. It is not merely an injection flag.

When a user changes a record to `user_only`, the local runtime excludes it
immediately. Every server/device agent index, cached overview, and derived
artifact must be cleaned before synchronization reports the privacy change as
fully acknowledged.

Private duplicate detection may return only "possible private duplicate;
review required." It must not reveal the private record's existence, kind,
scope, value, or similarity.

Interview and agent boundaries reject credentials, access tokens, private keys,
and other recognized secret material rather than storing them as profile
records. Question packs do not solicit passwords or protected/sensitive traits
that are unnecessary for personalization. A user may still deliberately store
sensitive personal context, but the review must expose the user-only control.

### Syncability

Changing a shared record from `syncable` to `device_only` cannot reuse its
shared ID. The service:

1. Tombstones the shared record everywhere.
2. Creates a new device-only record with a new ID.
3. Stores an encrypted peer-local `derived_from` link.

This avoids a shared and private split-brain under one object ID.

Device-only records never enter outbox, diagnostics, server APIs, server
summaries, or server-assisted interviews.

### Disable versus delete

Data existence and runtime enablement are separate:

- Chatbook personalization enablement is runtime-local.
- Server personalization enablement is runtime-local.
- Agent tool authority and proactive behavior are runtime-local.
- Disabling context does not delete records or silently stop synchronization.

The UI must never describe Personal Context Profile deletion as account
deletion.

### Undo

User edits may retain encrypted peer-local before-images for a short default
window of 24 hours. Undo creates an inverse canonical revision; it does not
rewrite history. If the head changed elsewhere, Undo produces a conflict.
Before-images are not synchronized and are included in local removal and global
purge.

## Interview and setup experience

### First-run flow

The existing first-run wizard commits normal provider, model, RAG, speech,
tool, note, appearance, and key-protection setup before personalization begins.
After that commit, it offers an optional **Get to know you** step as the
immediate continuation of setup.

The final action distinguishes **Save and use with agents** from **Save only**.
The former enables local profile context with a clear checked control; the
latter stores the records while leaving runtime use disabled. Merely opening or
partially completing an interview never enables personalization.

Skipping, cancelling, or encountering an interview-provider failure never
rolls back or blocks ordinary application setup.

### Workspace flow

After a workspace is successfully created, Chatbook may offer **Define project
context**. The interview targets that workspace's canonical scope and covers
purpose, desired outcomes, audience, constraints, conventions, tools, current
state, risks, and durable non-goals. Cancelling does not remove the workspace.

### Interview coordinator

One reusable `ProfileInterviewCoordinator` supports personal and workspace
question packs. It:

- Covers versioned topic areas and asks adaptive follow-ups.
- Counts every model turn as one question.
- Rejects compound or invalid model questions.
- Stops at a hard maximum of 20 questions.
- Supports skip, finish early, save, discard, and fixed-questionnaire modes.
- Pins provider and model for the session.
- Discloses the selected provider/model and the fact that external-provider
  retention is controlled by that provider.
- Runs the interview model without tools or record-write access.
- Accepts only strict structured output validated by Shared Core.

The fixed local questionnaire requires no model call. Adaptive re-interviews
may receive only eligible agent-visible existing records. User-only records are
never exposed to the model; deterministic local key matching can still warn the
user about a possible private duplicate.

### Draft privacy

Raw questions and answers never enter TOML, logs, diagnostics, crash reports,
sync, or ordinary backups. Each saved draft has a random encryption key and a
30-day expiry. After approved records are atomically committed, discarding the
draft, or reaching expiry, destroying that draft key crypto-shreds residual
SQLite/WAL copies.

When protected key storage is unavailable and the user declines a
passphrase-wrapped key, the interview is memory-only and cannot be resumed.

### Final review

The final review displays additions, updates, keyed conflicts, archives, and
possible duplicates. The user can edit or deselect each change and choose its
syncability and agent visibility. One local transaction commits the selected
records and encrypted outbox items. No cross-system atomicity is claimed.

Re-running an interview diffs against current records rather than replacing the
profile or creating obvious duplicates.

## Settings experience

The canonical F9 Settings Screen gains **My Profile** under Data & Privacy, not
Roleplay and not the deprecated legacy settings surface.

It provides:

- Global and workspace record browsing and filtering.
- Add, edit, archive, restore, and delete actions.
- Agent-visible/user-only and syncable/device-only controls.
- Pending agent-proposal review.
- Sync-conflict review as a distinct queue.
- Interview and re-interview actions.
- Runtime enable/disable and effective agent-access explanations.
- Link/unlink and synchronization status.
- Export and encrypted recovery-export controls.
- **Remove this device's copy** and **Delete everywhere** actions with distinct
  previews and confirmation copy.

Proposal review, interview diff review, and sync conflict review are separate
surfaces because they have different authority and resolution semantics.

A human-readable export requires an unlocked profile, explicit confirmation,
and a user-selected destination. It is a plaintext operation with a warning and
may include the selected global/workspace records, including user-only records.
It excludes keys, raw drafts, Undo data, and peer-local receipts. A recovery
export is separately passphrase-encrypted and may include device-only records
needed to restore a standalone profile.

The user-facing profile overview is a deterministic editable-data projection.
Optional model-written display prose is disposable and user-reviewed. The
agent context overview is separately generated for bounded runtime use.

## Agent context and tools

### Immutable request snapshot

At the start of each model request, `ProfileContextService` pins:

- Profile purge generation.
- Record-set revision.
- Active canonical workspace scope.
- Runtime authority revision.

The snapshot includes only active, unexpired, agent-visible records from the
global scope and the mapped active workspace. Changes during generation affect
the next request only.

Priority order is:

1. The user's current explicit request.
2. Active-workspace corrections, constraints, and keyed overrides.
3. Global corrections and constraints.
4. Relevant active preferences and working context.
5. A compact deterministic overview.

Profile context is serialized as escaped structured user-owned data. It is not
system authority and cannot override safety rules or the user's current
request. V1 relevance uses structured-key and bounded text matching only; no
embeddings or personalization-based RAG scoring run.

### Context budget

Injected profile context is limited to the lesser of:

- 12 KiB of UTF-8 serialized data; or
- 10% of the model's available input budget after reserving required system,
  conversation, tool, and current-request space.

The runtime uses its provider tokenizer when available and a conservative
fallback otherwise. It drops lower-priority profile records rather than
truncating typed payloads or squeezing the current request.

### Runtime-local authority

Each runtime and scope independently configures:

- `read_only`: context plus eligible search/get tools.
- `propose`: read access plus proposal creation. This is the default.
- `direct_write`: proposal access plus narrowly verified explicit-statement
  writes.

These grants never synchronize. Effective access is the intersection of local
enablement, active scope, record controls, lifecycle state, and tool policy.
The tool catalog is rebuilt or invalidated after workspace, profile, binding,
permission, or authority-revision changes.

### Shared logical tools

- `profile_search`
- `profile_get`
- `profile_propose`
- `profile_update`
- `profile_promote`

`profile_update` is exposed only under effective direct-write authority. Its
request must include the current user-message ID and a bounded verbatim
evidence span. The service verifies trusted user authorship and exact
containment. It stores only the message reference and integrity hash, not the
verbatim span.

Ambiguous matches, inferred facts, semantic duplicates, conflicts, and agent
promotion requests always become proposals. An agent cannot approve its own
proposal, alter privacy/sync controls, access user-only records, enumerate
other workspaces, directly archive records, purge the profile, or permanently
delete records.

Tool results use explicit statuses such as `applied`, `proposal_created`,
`review_required`, `permission_denied`, `quota_exceeded`, `conflict`, and
`profile_locked`.

### Agent limits

- Canonical encoded record payload: 16 KiB maximum.
- Search: 20 results maximum, 5 by default.
- Direct-write evidence span: 1,000 characters maximum.
- Proposals: 5 per turn, 25 per session, and 200 unresolved per profile.

User-authored settings and interview changes are not subject to agent proposal
rate limits. Server-advertised operational quotas may be stricter, but linking
must identify an incompatibility before upload rather than silently discard
local data. While linked, Chatbook enforces the last negotiated server quota
before committing an agent mutation. If the server later lowers a quota,
already-committed local records remain local and Sync reports an actionable
quota-attention state rather than dropping them.

## Synchronization and multi-device behavior

### Sync domains

Personal Context extends Sync V2 through negotiated, whole-object domains:

- `personal_context.manifest`
- `personal_context.scope`
- `personal_context.record`
- `personal_context.proposal`
- a content-free profile purge barrier

Conflicts remain Sync V2 review objects rather than a normal profile domain.
Interview drafts, runtime authority, key protectors, local workspace mappings,
Undo before-images, and peer-local activity receipts do not synchronize.

For these domains, upserts and tombstones carry the canonical Sync V2 base
metadata: base object revision, base object integrity tag, and base server
cursor. Personal Context negotiates a versioned keyed-HMAC integrity tag in the
existing object-hash position. This extension applies only when the server
advertises the matching capability; it does not silently alter existing Sync
domains.

The HMAC conceals guessable plaintext hashes and detects byte differences. It
is not a signature or authorization mechanism. Authenticated TLS, stable user
identity, and registered device identity establish authorship.

### Stable binding

Profile binding uses:

- A stable server authority ID.
- The stable authenticated server user ID.
- A peer-local random Chatbook device ID.

URLs, labels, routing IDs, and credentials are not profile authority.

The server serializes first-link profile creation under transaction. One user
may register multiple Chatbook devices. Device acknowledgments and expiry are
tracked per Sync domain. A dormant expired device must rebootstrap instead of
replaying an old cursor.

The home server owns the sync integrity key and distributes it only to
authenticated registered devices through a wrapped bootstrap response. A
standalone Chatbook profile uses a provisional local integrity key; linking
adopts the server key and recomputes personal-context integrity tags during the
required rebaseline.

### Initial link reconciliation

No local record uploads before reconciliation completes:

1. Chatbook authenticates and fetches capabilities, the server manifest, and
   eligible canonical heads.
2. It compares local and server records in a temporary encrypted staging area.
3. The provisional global scope maps to the server global scope. For every
   local workspace, the user chooses an existing server scope, creates a new
   server scope, or keeps the overlay local and unlinked.
4. Identical IDs and bytes converge after scope mapping.
5. Same-scope, same-key differences become explicit key-collision reviews.
6. Merely similar content becomes a possible-duplicate warning.
7. Device-only records are excluded.
8. Quota and schema incompatibilities block linking with actionable detail.
9. The user approves the merge preview.
10. Chatbook journal-rebinds its provisional manifest to the server canonical
   profile, re-encrypting affected local envelopes where authenticated
   associated data changes.
11. Normal push/pull begins only after the binding commits.

Cancelling or interrupting the preview leaves both replicas unchanged. The
journal makes interrupted identity adoption resumable and idempotent.

### Normal mutation and synchronization

Chatbook always commits an immutable local revision and encrypted exact-wire
outbox snapshot in one transaction. It never mixes direct server
personalization CRUD writes with Sync writes.

The server UI and server agents write the server canonical repository and Sync
log transactionally. Synchronization:

1. Pushes local changes with their immutable base metadata.
2. Persists accepted server versions or durable conflicts.
3. Pulls newer server objects.
4. Validates and applies all eligible objects before advancing the cursor.
5. Completes privacy cleanup before acknowledging restrictive changes.
6. Invalidates context and tool caches after commit.

Unknown newer objects are retained opaquely. They are not decrypted, edited,
indexed, injected, or approved by an older runtime.

Different object IDs merge automatically only when they do not violate the
same-scope key uniqueness rule. Exact key collisions create review objects;
mere semantic similarity does not block convergence.

### Global purge

**Delete everywhere** may be initiated by an authenticated user in Chatbook or
tldw_server. It has higher priority than interview, migration, key rotation,
normal mutation, and synchronization:

1. The initiating Chatbook device immediately destroys local readable content,
   freezes profile writes, and retains only a durable content-free purge
   request.
2. The server advances `purge_generation` under transaction.
3. The server removes canonical content, proposals, conflicts, drafts,
   recovery material, derived artifacts, and readable sync history.
4. A content-free generation barrier is distributed to devices.
5. Older-generation writes are rejected.
6. Devices erase their replicas on their next connection.

The UI shows **Global deletion pending** until server acknowledgment. An
offline or stolen device cannot be physically erased remotely; local
encryption is the remaining protection. A minimal content-free tombstone
ledger remains until registered devices acknowledge or expire.

The user may deliberately create a new empty profile only after the purge
barrier is acknowledged. It uses the new generation and cannot revive old
records.

### Remove this device's copy

For a linked profile, Chatbook previews pending outbox items, unresolved
conflicts, and unacknowledged changes before unregistering and clearing the
device. The user must sync, export, or explicitly discard them. Removing the
local copy unregisters the device and prevents automatic rebootstrap until the
user links again.

For a standalone profile, the same action warns that no server replica exists
and requires either an encrypted recovery export or an explicit decision to
destroy the only copy. It then destroys the profile key and local data.

## Encryption and key custody

### Envelope hierarchy

Every record revision, proposal payload, interview draft, conflict payload,
outbox snapshot, Undo before-image, and migration/recovery snapshot receives a
random data-encryption key (DEK). The DEK is wrapped by a per-profile key. V1
payload envelopes use AES-256-GCM with a unique random 96-bit nonce and a
versioned algorithm header. Associated data covers the peer storage envelope,
object type, object ID, version ID, and serialized schema version. Sync
integrity tags use HMAC-SHA-256. Passphrase-based key wrapping derives its key
with versioned, calibrated scrypt parameters. Implementations use vetted crypto
libraries and never implement these primitives themselves.

Deleting content destroys its wrapped DEK. This crypto-shreds recoverable
content that may remain in SQLite freelists, WAL files, ordinary backups, or
filesystem snapshots created after envelope encryption was deployed.

Python cannot promise immediate zeroization of every transient in-memory copy.
Implementations minimize plaintext lifetime, avoid global caches, and never
claim stronger process-memory erasure than the runtime can provide.

### Chatbook key protection

Chatbook uses a key-protector abstraction:

1. OS protected secret/keyring when available.
2. A passphrase-wrapped profile key as the fallback.
3. No plaintext-key fallback.

Keys never enter TOML. If the key protector is unavailable or cancelled, the
profile is `Locked`; an interview may proceed only as memory-only,
non-resumable work.

A linked device that permanently loses its local profile key can discard its
encrypted replica and rebootstrap eligible syncable records. A standalone
profile requires a user-created encrypted recovery export; the system must
state this before the user relies on local-only data.

### Server key protection

tldw_server encrypts content under per-profile keys wrapped by a configured
server master key or KMS-backed protector. Server startup validates key custody
before enabling profile access. A missing or changed master key locks affected
profiles; the server never generates a silent replacement.

Authentication and authorization occur before any decrypt. APIs, agents, jobs,
compatibility adapters, and migrations all use the authorized service boundary
rather than reading encrypted columns.

Ordinary rotation rewraps DEKs instead of re-encrypting all profile content.
The separately wrapped integrity key does not rotate through the ordinary DEK
path. An integrity-key compromise requires a versioned full rebaseline and, if
necessary, a profile-generation transition.

### Search and derived artifacts

V1 has no plaintext FTS or embedding index. Search decrypts a bounded eligible
set inside the authorized service process and performs structured-key/text
matching. Generated overviews and caches are encrypted at rest and invalidated
by profile generation, record-set revision, active scope, and authority
revision.

### Backup and metadata disclosure

Server recovery requires encrypted data, wrapped profile keys, and separately
protected master-key recovery. Chatbook recovery requires the existing key
protector or an encrypted recovery export. Migration recovery snapshots are
encrypted, single-purpose, included in purge, and destroyed after successful
validation or seven days, whichever comes first.

Encryption does not conceal all metadata. An observer of storage or sync may
see random object and scope IDs, revision frequency, ciphertext sizes, scope
relationships, tombstones, and timing. Scope labels, record kinds, content,
provenance, and values remain encrypted.

Syncable records are decrypted in the authenticated client process, sent over
TLS using `server_trusted_v1`, and re-encrypted at rest by the server. The UI
must disclose that an authorized home server can read syncable content.

## Server API and legacy migration

### Canonical API

tldw_server exposes a new versioned `/api/v1/personal-context` API for its own
UI and non-Sync consumers. Its handlers use the same canonical service as Sync.
The surface covers status/manifest, scopes, records, proposals, review actions,
runtime enablement, export, and purge.

Chatbook local-first mutations do not call this CRUD API. They use the local
service and Sync outbox.

### Compatibility adapters

Existing `/api/v1/personalization` routes become compatibility projections over
the canonical service after migration. They are not a second repository or
write path. A legacy write may update only fields the old contract understands
while preserving all other canonical fields. A request that cannot be
represented losslessly fails with a clear compatibility error.

### Per-user migration

Migration is automatic storage maintenance, not a new personalization opt-in.
It preserves each user's existing enablement state:

1. Fence legacy writes for the affected user only.
2. Create an encrypted recovery snapshot with the seven-day maximum described
   above.
3. Run an idempotent canonical backfill.
4. Preserve existing semantic-memory IDs, content, timestamps, and applicable
   controls.
5. Classify uncertain memories as `legacy_unclassified`; do not guess a kind.
6. Convert response style and preferred format into canonical preference
   records.
7. Keep scoring, proactive behavior, reflection settings, and agent grants as
   server runtime configuration.
8. Keep episodic/Persona/Companion/Auth objects in their owning domains unless
   an explicit canonical mapping exists.
9. Validate canonical and legacy compatibility projections without logging
   content.
10. Atomically switch compatibility routes to the canonical service.
11. Reopen writes through the one canonical service.
12. Crypto-shred the recovery snapshot after its bounded validation window.

Other users remain available through lazy/per-user migration. Shadow reads may
compare projections, but dual writes are prohibited.

Migration is forward-only after legacy plaintext storage is retired. Rollback
means restoring the encrypted canonical snapshot under compatible software or
shipping a forward fix; an old binary cannot safely resume authority.

Legacy plaintext backups and external snapshots cannot be retroactively
encrypted. Deployment therefore requires an explicit retention inventory,
expiry/removal procedure, and operator disclosure. Migrated SQLite storage uses
secure deletion and a database rewrite where practical, without claiming that
copies outside system control were erased.

Topics, embeddings, summaries, and RAG priors are derived data rather than
canonical records and do not synchronize. Existing Persona and Companion data
remain governed by their own contracts.

## Runtime states and error handling

The service reports explicit operational states:

- `Available`
- `Locked`
- `Migration required`
- `Migrating`
- `Sync attention required`
- `Review required`
- `Purge pending`
- `Unsupported records present`

Chat continues when profile context is locked or unavailable, but it sends no
profile context and displays a visible status indicator. It never falls back
to stale plaintext caches.

- A corrupt individual object is quarantined and omitted without disabling all
  other records.
- A manifest, generation, or key-custody failure locks the whole profile.
- Sync failure does not block local editing, but unsynchronized changes remain
  visible.
- Interview-provider failure preserves an encrypted draft and offers retry or
  the fixed questionnaire.
- Restrictive privacy changes take effect locally immediately and remain
  incomplete until cleanup acknowledgments arrive.
- Unknown newer records remain opaque and unavailable to agents.
- Profile-level operation priority is: global purge; privacy cleanup;
  migration/integrity rebaseline/key rotation; ordinary sync and mutation.

## Operational privacy and diagnostics

Logs and diagnostics contain reason enums, opaque version IDs, counts, and
bounded error summaries only. They never contain profile values, raw interview
answers, evidence spans, decrypted payloads, scope labels, or low-entropy
content hashes.

Telemetry follows the application's existing opt-in policy and aggregates
counts, timings, result codes, supported schema versions, and queue sizes
without stable profile or workspace identifiers. Fine-grained activity
receipts are peer-local, encrypted, retained for 30 days, and do not form a new
Sync domain. Canonical current provenance travels with the record.

A runtime kill switch disables profile use without deleting the data or
silently suspending synchronization.

## Verification strategy

### Shared Core conformance

- Canonical serialization and keyed integrity tags match across runtimes.
- Python and TypeScript runtime validation accept and reject the same fixtures.
- Lifecycle, scope, payload, provenance, expiry, proposal, and privacy rules are
  deterministic.
- Minimum and current supported package/schema combinations are exercised.
- Unknown-version fixtures remain opaque rather than partially interpreted.

### Repository and integration tests

- Real SQLite repository tests cover atomic revision/outbox commits,
  optimistic concurrency, quarantine, and purge fencing.
- Two Chatbook devices plus one server cover convergence, conflicts, initial
  link reconciliation, workspace mapping, tombstones, cleanup acknowledgments,
  device expiry, rebootstrap, integrity rebaseline, and global purge.
- Migration fixtures cover idempotency, single-writer fencing, projection
  equivalence, interruption recovery, and forward-only restoration.
- Context tests cover scope, priority, expiry, visibility, conflicts, budgets,
  escaping, cache invalidation, and immutable request snapshots.
- Agent tests cover local authority, catalog invalidation, private duplicate
  responses, direct-write evidence, proposal quotas, and forbidden actions.
- Interview tests cover provider disclosure, no-tool execution, strict output,
  fixed fallback, one-question turns, the 20-question maximum, review diffs,
  and raw-draft destruction.
- UI tests cover optional non-blocking setup, My Profile editing, re-interview,
  review queues, status states, link preview, local removal, and global purge.

### Security evidence

Tests write unique canary values and scan the ordinary database, WAL, outbox,
logs, diagnostics, caches, migration snapshots, and application-owned backups
for plaintext regressions. They also cover missing keys, changed master keys,
authorization-before-decryption, interrupted rotation, and restrictive cleanup.

Canary scanning is regression evidence, not proof that plaintext never existed
in process memory, historical backups, or external filesystem snapshots. It
supplements the threat model, service-boundary review, and operator controls.

During implementation, targeted tests covering changed functionality are
mandatory. A local full repository sweep remains subject to the repository rule
requiring explicit user opt-in; configured CI may run its normal suite.

## Delivery sequence

This architecture is one governing specification but is too large for one
atomic PR. Planning must create dependency-ordered Backlog tasks and
repository-specific implementation slices.

### Phase 0 — Contract and decision

- Create ADR-102 after rechecking the number.
- Publish Shared Core models, JSON Schema, fixtures, compatibility policy, and
  threat model.

### Phase 1 — Chatbook local profile

- Encrypted local repository and key protection.
- My Profile settings and manual editing.
- Local removal, locking, export/recovery, and read-only agent context.
- No server dependency.

### Phase 2 — Interview and controlled agent learning

- Personal and workspace interviews.
- Final diff and proposal review.
- Runtime-local read/propose/direct-write grants and tools.

### Phase 3 — Server canonical store

- Encrypted canonical repository and `/api/v1/personal-context` service/API.
- Fenced per-user migration.
- Lossless legacy compatibility projection.
- Server context and tool consumers use the canonical service.

### Phase 4 — Sync and multiple devices

- Capability negotiation and canonical profile binding.
- Initial reconciliation and workspace mapping.
- Whole-object domains, conflicts, cleanup acknowledgments, device expiry,
  rebootstrap, integrity rebaseline, and global purge.
- The feature remains gated until destructive and privacy-reduction integration
  tests pass.

### Phase 5 — Legacy retirement

- Deprecate and later remove compatibility routes.
- Retire legacy plaintext storage and backups under documented retention rules.
- Remove obsolete Chatbook server-only personalization presentation paths after
  all callers use the canonical service.

Each phase must preserve one write authority per runtime and deliver a usable,
testable state. No temporary dual-write bridge is permitted.

## Risks and mitigations

- **Profile data becomes prompt authority.** Context is escaped, explicitly
  labeled as user-owned data, budgeted below the current request, and never
  treated as safety or system policy.
- **Inferred facts become silently trusted.** Inference creates proposals only;
  active records require user acceptance unless a direct explicit statement
  passes the narrow evidence gate.
- **Workspace context leaks.** Canonical random scopes require an active
  peer-local mapping; unlinked and unrelated scopes are unavailable to agents.
- **Two stores diverge.** Shared canonical bytes, conformance fixtures, one
  mutation service, and Sync V2 whole-object domains replace app-specific
  projections.
- **Privacy is reduced only in the UI.** Restrictive changes immediately remove
  agent access and require cleanup acknowledgments for indexes, caches, and
  artifacts.
- **A stale device resurrects deletion.** Tombstones, device expiry, and the
  profile purge generation reject older writes.
- **Encryption keys are silently replaced.** Missing or changed protectors lock
  the profile. Recovery and rotation are explicit and journaled.
- **Initial linking duplicates two existing profiles.** Linking performs an
  encrypted, cancellable reconciliation before binding or upload.
- **Legacy rollback restores plaintext authority.** Migration is forward-only;
  recovery uses encrypted canonical snapshots and compatible software.
- **Cross-repository releases drift.** Supported-range tests, schema fixtures,
  pinned versions, and capability negotiation fail closed.

## References

- `tldw_chatbook/Personalization_Interop/personalization_scope_service.py`
- `tldw_chatbook/Personalization_Interop/server_personalization_service.py`
- `tldw_chatbook/Sync_Interop/local_first_sync_service.py`
- `tldw_chatbook/Sync_Interop/envelope_builder.py`
- `tldw_chatbook/Sync_Interop/envelope_applier.py`
- `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/Widgets/workspace_create_modal.py`
- `backlog/decisions/008-sync-v2-client-m1-contract-alignment.md`
- `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`
- `tldw_server2/Docs/Product/Personalization_Memory_Layer_PRD.md`
- `tldw_server2/tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- `tldw_server2/tldw_Server_API/app/api/v1/endpoints/personalization.py`
