# ADR-021: File-Backed Notes Disk Authority and Recovery Replica

Status: Proposed
Date: 2026-07-22
Related Task: [TASK-399](../tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
Supersedes: N/A

## Decision

Existing linked Markdown/text files are the sole content authority for
file-backed Notes. Their exact relative paths own identity on disk and their
filenames own displayed titles. Chatbook edits the actual files, preserving
opaque frontmatter and byte-style facts, so ordinary Git status, staging,
committing, and pushing remain the user's workflow.

SQLite supports that authority through two File Notes stores outside the
existing ChaChaNotes database:

1. A dedicated `file_notes.db` holds `file_notes_storage`, roots, local UUIDs,
   bindings, derived editable bodies, navigation metadata, triggerless file FTS,
   indexing state, and recovery-store pairing.
2. An independent, same-device `notes_recovery.db` holds the mutation journal,
   mandatory operation-safety bytes, confirmed-deletion snapshots, and the
   opt-in current replicas/checkpoints introduced for selected files and
   folders.

Existing `notes`, `notes_fts`, keyword, relation, Sync, MCP, export, and RAG
tables, triggers, schema version, migrations, constructors, backup/restore
controls, and write services remain Database-note-only and unchanged.
Intentional combined Library reads happen above separate Database-note and
file-note repositories; the two sources do not share a generic write path or
failure result.

Delivery is split into five independently verifiable milestones:

- **A:** cross-platform read-only root preview, isolated projection/search,
  scalable tree, external-change monitoring, preview/export, and Console
  handoff. Milestone A has no recovery-database prerequisite.
- **B0:** executable packaged-app proof of the required native APFS primitives
  on an explicitly declared, finite initial writable macOS support set. B0
  publishes a versioned machine-readable capability manifest bound to the native
  mutation-adapter ABI/artifact, exposes no write action, and must pass before
  B1 begins. The final B1 release candidate reruns the same qualification before
  writable controls can ship.
- **B1:** journaled create/save/rename/move in existing directories, initially
  writable only on verified local APFS roots on macOS, plus recovery-only
  enumerate/verify/exact-export without opening ChaChaNotes or `file_notes.db`.
- **B2:** confirmed file deletion and its verified 30-day snapshot plus working
  absent-original-path restore/exact-export fallback, delivered together.
- **B3:** opt-in file/folder protection, verified current replicas, coalesced
  checkpoints, minimal history, and expanded restore choices.

Linux, Windows, network, cloud-synchronized, and otherwise unverified roots
remain first-class read-only sources until separately approved native writable
adapters pass the same safety and durability suite.

## Context

Users may already have thousands of notes in a folder hierarchy managed by Git
and other editors. Their required workflow is:

1. open and manage those notes in Chatbook;
2. see Chatbook changes as ordinary working-tree changes;
3. use normal Git tooling to stage, commit, and push.

The legacy folder-sync design treats SQLite and disk as competing peers. Its
direction/winner policy, timestamp comparisons, and incomplete move/delete
identity are unsuitable for making a Git working tree the canonical store.

Chatbook still needs local identity, metadata, FTS, responsive navigation,
Console handoff, mutation recovery, and optional selected-note history. Those
capabilities require SQLite, but not SQLite content authority.

SQLite and a filesystem cannot share one atomic transaction. Writable support
therefore requires a durable operation journal, exact expected hashes,
round-trip-verified recovery bytes before destructive action, atomic
replacement that preserves the displaced target, explicit file/directory
durability barriers, and startup classification. Python's portable filesystem
APIs do not provide equivalent guarantees on every supported platform, so the
first writable contract is intentionally narrower than the read-only contract.

## Required Boundaries

- One cross-process-elected File Notes coordinator is the only file mutation
  authority.
  Writable paths are traversed from a pinned root handle with no-follow
  semantics; unsupported containment, replacement, permission, or durability
  primitives fail closed. Nested mounts remain read-only and cross-device
  mutation never falls back to copy/delete.
- Linking rejects any candidate root whose canonical location has an
  ancestor/descendant overlap with a File Notes database, recovery store,
  sidecar or marker, fixed runtime namespace, or any configured application
  data, database, configuration, cache, or log path. This is an activation
  boundary for read-only and writable roots, not merely a mutation-time check.
- A file carrying ACLs, extended attributes, Finder/resource metadata, file
  flags, unusual ownership, alternate streams, or other metadata the platform
  adapter cannot detect and round-trip remains read-only. No mutation may
  silently discard unsupported metadata. Retained revisions carry a versioned
  supported metadata manifest/fingerprint sufficient to reapply and verify
  those facts. Its canonical encoded form is limited to 64 KiB per retained
  revision or live operation, verified before mutation, and counted toward
  recovery capacity; a larger manifest is read-only. An unexpected
  metadata-only race retains the displaced object and enters
  recovery/Attention before cleanup.
- Every mutation checks a versioned SHA-256 raw hash immediately before writing,
  journals intent, required bytes, and expected/intended metadata
  manifests/fingerprints first, preserves an atomically displaced target or
  delete quarantine, verifies the observed result, updates the `file_notes.db`
  projection idempotently, and marks the recovery operation complete last.
- File FTS is a triggerless, retryable derivative outside the operation safety
  boundary. Projection failure keeps the operation pending and fails writes
  closed; FTS failure leaves indexing pending but cannot keep the operation open
  or block a later write. Stale FTS generations never surface as current.
  Every projection/body-cache/FTS batch performs free-space admission before
  writing and preserves the recovery free-space floor. If an inventory batch
  cannot safely cache bodies or index content, it publishes metadata/path-only
  rows with a visible degraded-indexing state; writable admission separately
  budgets its required journal and projection growth and otherwise fails
  closed.
- Confirmed Delete is absent until its verified snapshot and minimal Restore
  are both usable. B2 restores only to an absent original path and offers exact
  export if occupied or if the parent is missing; alternate-path and overwrite
  restore wait for B3.
- `notes_recovery.db` is self-contained enough to enumerate, verify, and
  exact-export retained content without opening ChaChaNotes or `file_notes.db`,
  beginning in B1. It is owner-only plaintext on the same device, not an
  off-device backup.
- `file_notes.db` caches relative paths, readable editable bodies, and a token
  index as owner-only plaintext for all indexed files. Protection selection
  controls exact-byte recovery guarantees, not whether content is
  normally cached/indexed; activation discloses and estimates these separately.
  Low-space metadata-only degradation is the explicit exception.
- At B1 bootstrap, `file_notes.db` and `notes_recovery.db` persist the same
  storage-instance ID, random recovery-instance UUID, and bootstrap generation.
  Creation requires complete absence of conflicting database/sidecar/marker
  evidence. A `bootstrap_in_progress` marker binds the proposed
  storage/UUID/generation in a versioned, checksummed payload. It is created
  exclusively, written completely, reread and verified, file-flushed with the
  adapter's required fsync/full-fsync primitive, and followed by a parent
  directory fsync before either database identity commit. Recovery identity
  commits first, projection-side identity commits second, and the marker is
  removed only after both verify. Startup may resume only that exact
  intermediate state. Missing, mismatched, lost, corrupt, or nonmatching
  orphaned state fails closed, preserves evidence, permits recovery-only export
  when possible, and never silently initializes or adopts a replacement.
- Recovery uses a fixed 1 GiB live-data cap covering compressed content and
  encoded manifests, plus a fixed 256 MiB post-reservation free-space floor in
  the initial release. Guaranteed or unresolved content is never silently
  evicted.
- `watchdog` is a declared core dependency for near-real-time packaged
  monitoring. A visible bounded polling fallback feeds the same hash-based
  reconciliation path; watcher events and mtimes never decide authority.
- B0 defines a finite writable macOS support set independent of the
  application's broader macOS support floor. It retains a checked-in
  packaged-app APFS capability/version matrix with explicit go/no-go evidence,
  including the named power-cut/reboot method and observed durability result,
  and emits the same approval as a versioned machine-readable manifest consumed
  by B1. Runtime admission matches the exact OS/filesystem entry and packaged
  native mutation-adapter ABI/artifact hash; the tested application commit is
  retained as audit provenance rather than an equality gate. An absent or
  mismatched combination fails to read-only, and the final B1 release candidate
  must requalify its exact packaged adapter before the release gate opens.
- A coordinator-election lease permits one active root and one
  monitor/reconciler without fencing legacy filesystem sync. A read-only A link
  rejects configured overlap but never acquires the mutation lease or pauses
  unrelated legacy work.
- Beginning in B1, a `LegacyRootOwnershipGate` fences every legacy mutation
  entry point. Cooperative legacy passes hold a cross-process mutation lease
  shared for their full lifetime; read/write upgrade drains them and holds it
  exclusively. While File Notes holds that exclusive ownership, every legacy
  filesystem pass is blocked or deferred, including nominally non-overlapping
  work. Relaxing this rule requires a later contract backed by hardened,
  containment-safe legacy traversal; path-prefix separation alone is
  insufficient.
- Any startup classification that changes journal, projection, or filesystem
  state requires both coordinator election and exclusive mutation ownership.
  Passive processes may inspect and report incomplete operations but never
  reconcile or clean them.
- Both owner-only leases live in a fixed runtime namespace independent of
  configurable user-data/main-database/repository paths. The supported contract
  is one current installation, one configured storage profile, one active root,
  and one cooperative coordinator. Older/different tools are external writers.
- No ChaChaNotes migration, constructor gate, compatibility token, or main
  backup/restore coupling is introduced. `file_notes.db` bootstrap, corruption,
  and schema incompatibility disable only the Files source. Existing Settings
  backup/restore controls remain Database-note-only and explicitly exclude both
  File Notes databases.
- Library mounts a separate `FileNotesWorkbench` whose
  `FileNotesSessionController` owns file-only navigator/editor state.
  Database-source selections switch the whole Notes content surface to the
  existing `LibraryNotesCanvas`; separate host-File and Database-editor Back
  targets preserve the delegated Database list while still restoring the exact
  File navigator state after both editors' leave guards succeed. A serializable
  in-process File session snapshot also restores that state after the Library
  screen is reconstructed or the user leaves and returns to Library.
- Milestone A uses a selectable read-only body reader with no dirty/autosave
  state. Combined search has independent pageable source groups and
  source-scoped errors. Unsafe draft status outranks root read-only/offline
  status, and focus always moves to a mounted visible target.
- A measured, versioned interactive body-size ceiling protects the Textual UI
  responsiveness budget. Files above the ceiling use a bounded read/export-only
  reader and never mount the full editable text widget or enter autosave until
  a separately verified large-file editor exists.
- Controlled shutdown closes File Notes mutation admission and crosses a File
  operation barrier before generic worker cancellation. Every dirty File draft
  must either complete a verified save or be durably represented as unresolved
  recovery/Attention state; shutdown cannot silently discard it.
- `Changed this session` is process-lifetime memory state. Only pending and
  Attention operations are durable across restart.
- Unlink retains and discloses plaintext projection/FTS until Forget. The
  zero-root File route lists detached roots, retained size, and Relink/Forget
  actions even when an original folder is unavailable. Forget explicitly
  deletes triggerless FTS rows with the projections, but cannot discard
  pending/Attention state or unexpired guaranteed recovery and is disclosed as
  logical removal, not secure erasure of SQLite pages, filesystem snapshots,
  or backups.
- Folder mutation, file templates, file keywords/links, file MCP/RAG, mixed
  bulk export, Git controls, general recovery purge, configurable quotas,
  additional writable platforms, and additional active roots are deferred.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Continue equal-peer bidirectional folder sync | Retains ambiguous authority, timestamp winners, weak move/delete identity, and avoidable Git noise. |
| Make SQLite canonical and import/export files | Chatbook edits would not be immediate working-tree changes, and external Git/editor changes would require a second authority loop. |
| Use disk without a database projection | Loses responsive FTS, stable local references, cached navigation, and Console handoff for large trees. |
| Put file rows in `notes` behind `storage_kind` | Existing direct CRUD, FTS, Sync, MCP, export, RAG, keyword, and relation paths assume every active `notes` row is Database-owned. Dedicated tables isolate ownership by construction. |
| Put dedicated file tables in ChaChaNotes | Still expands its schema, migration, constructor, backup/restore, corruption, and availability boundary without making it a content authority. A sibling `file_notes.db` isolates the same projection. |
| Store UUIDs in frontmatter or a repository manifest | Alters user repositories solely for Chatbook bookkeeping and creates Git-visible metadata noise. |
| Store recovery bytes in ChaChaNotes or `file_notes.db` | Prevents independent recovery enumeration and couples recovery capacity/corruption to an application database. |
| Ship macOS, Linux, and Windows writes together | Portable APIs do not offer equivalent beneath-root, atomic-displacement, permission, and durability guarantees. Read-only support still provides immediate value everywhere. |
| Stage and swap the File Notes database pair | Pair swapping adds live-handle, sidecar, cross-volume, and launcher coordination without making filesystem publication atomic; durable identity and orphan detection provide the required fail-closed boundary. |
| Add Git controls in the first release | Expands stateful risk before file authority, conflict, deletion, and recovery behavior are proven. |

## Consequences

### Benefits

- The filesystem and Git remain the unambiguous source of truth.
- Existing Database-note behavior is protected by a separate database file,
  instead of a schema migration or audit of every legacy caller.
- Milestone A can deliver a useful cross-platform Notes UI before writable
  filesystem guarantees are ready.
- Retained recovery remains independently inspectable if ChaChaNotes or
  `file_notes.db` is unavailable, beginning with B1.
- A separate workbench/controller prevents the file editor state machine from
  further concentrating responsibility in `LibraryScreen`.

### Accepted trade-offs

- A genuinely fresh namespace with no recovery evidence assigns new local
  UUIDs. A lost paired `file_notes.db` alongside retained recovery evidence is
  recovery-only/fail-closed in the core rollout; it is not permission to create
  a replacement projection store.
- Missed or ambiguous external moves appear as Missing plus a new identity
  until explicitly reassociated.
- Opaque frontmatter is preserved but not structurally edited.
- Both File Notes databases consume local plaintext disk; all readable indexed
  bodies are cached even when exact recovery protection is off. Neither protects
  against loss of the device. Under storage pressure, new body-cache/FTS
  publication degrades to metadata/path-only rows rather than consuming the
  recovery floor.
- Linux, Windows, and macOS combinations outside B0's finite writable manifest
  remain read-only in the first writable delivery.
- One active root and one cooperative current installation are support limits,
  not claims of multi-profile coordination.
- Changing the configured main-database path or user-data directory selects a
  separate File Notes profile. The first rollout neither discovers nor migrates
  stores across namespaces; returning to the prior configuration reopens its
  prior profile.
- File/database consistency is repaired through journaling and reconciliation,
  not a nonexistent cross-resource transaction.
- Symlink and hardlink mutation, network/cloud write guarantees, multi-host
  writing, folder mutation, and Git operations remain outside the initial
  guarantees. Files whose ACL/xattr/ownership metadata cannot be proven
  round-trippable remain read-only.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md)
- [TASK-399](../tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
- [ADR-001: canonical ADR workflow](001-adopt-backlog-decisions-as-canonical-adrs.md)
- [ADR-003: Settings/Library/RAG ownership](003-settings-library-rag-defaults.md)
- [ADR-004: storage restart boundary](004-settings-storage-defaults-restart-boundary.md)
- [ADR-008: remote Sync v2 contract](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-011: Workbench UI system](011-chatbook-workbench-ui-system.md)
- [ADR-015: shell destination ownership](015-shell-destination-ia.md)
