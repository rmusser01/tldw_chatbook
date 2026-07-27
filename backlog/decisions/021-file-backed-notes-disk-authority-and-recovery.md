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

SQLite supports that authority through two deliberately separate stores:

1. The main Chatbook database gains dedicated `file_notes_storage`,
   `note_file_roots`, `file_note_projections`, and `file_notes_fts` tables for
   local UUIDs, navigation, derived editable bodies, search, and recovery-store
   pairing.
2. An independent, same-device `notes_recovery.db` holds the mutation journal,
   mandatory operation-safety bytes, confirmed-deletion snapshots, and the
   opt-in current replicas/checkpoints introduced for selected files and
   folders.

Existing `notes`, `notes_fts`, keyword, relation, Sync, MCP, export, and RAG
tables, triggers, and write services remain Database-note-only. Intentional
combined Library reads happen above separate Database-note and file-note
repositories; the two sources do not share a generic write path.

Delivery is split into four independently verifiable milestones:

- **A:** cross-platform read-only root preview, isolated projection/search,
  scalable tree, external-change monitoring, preview/export, and Console
  handoff. Milestone A has no recovery-database prerequisite.
- **B1:** journaled create/save/rename/move in existing directories, initially
  writable only on verified local APFS roots on macOS.
- **B2:** confirmed file deletion and its verified 30-day snapshot plus working
  minimal restore, delivered together.
- **B3:** opt-in file/folder protection, verified current replicas, coalesced
  checkpoints, minimal history/restore, and independent enumerate/verify/export.

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

- One process-wide file coordinator is the only file mutation authority.
  Writable paths are traversed from a pinned root handle with no-follow
  semantics; unsupported containment, replacement, permission, or durability
  primitives fail closed. Nested mounts remain read-only and cross-device
  mutation never falls back to copy/delete.
- Every mutation checks a versioned SHA-256 raw hash immediately before writing,
  journals intent and required bytes first, preserves an atomically displaced
  target or delete quarantine, verifies the observed result, updates the main
  projection idempotently, and marks the recovery operation complete last.
- Confirmed Delete is absent until its verified snapshot and minimal Restore
  are both usable.
- `notes_recovery.db` is self-contained enough to enumerate, verify, and
  exact-export retained content without opening the main database. It is
  owner-only plaintext on the same device, not an off-device backup.
- At B1 bootstrap, the main and recovery stores persist the same random
  recovery-instance UUID plus their bound storage-instance identity. Once
  paired, missing/mismatched storage fails closed and never silently initializes
  an empty replacement. Recovery relocation/clone is an explicit future
  administrative action.
- Recovery uses a fixed 1 GiB live compressed-payload cap and a fixed 256 MiB
  post-reservation free-space floor in the initial release. Guaranteed or
  unresolved content is never silently evicted.
- `watchdog` is a declared core dependency for near-real-time packaged
  monitoring. A visible bounded polling fallback feeds the same hash-based
  reconciliation path; watcher events and mtimes never decide authority.
- A `LegacyRootOwnershipGate` fences every legacy sync engine mutation entry
  point. Before file-root activation, every cooperative legacy pass holds a
  cross-process shared OS root-mutation lease for its full lifetime. Activation
  closes shared admission and holds that lease exclusively. Passive processes
  run no legacy filesystem sync while it is held; the exclusive owner may admit
  only a non-overlapping local legacy pass under its stronger ownership.
- The coarse owner-only per-user OS lease lives in a fixed application runtime
  namespace independent of configurable user-data/main-database/repository
  paths and permits one active root across cooperative current Chatbook
  processes and configured storage instances. The supported concurrency
  contract is one current installation, one configured main database, one
  active root, and one cooperative coordinator. Older/different tools are
  external writers, not participants in an unprovable global version floor.
- A mandatory pre-constructor bootstrap and interprocess schema-maintenance lock
  guarantee that a verified SQLite online backup precedes the additive
  main-schema migration. Backup/preflight failure leaves the healthy immediate
  predecessor schema usable only in explicit Database-only compatibility mode;
  no file workbench/table access is allowed. The recovery database is paired
  independently before B1. There is no live database family rename, pair swap,
  automatic downgrade, or core pair-backup/restore mechanism.
- Library mounts a separate `FileNotesWorkbench` whose
  `FileNotesSessionController` owns file-only navigator/editor state.
  Database-source selections continue through the existing
  `LibraryNotesCanvas` and handlers both with and without an active file root.
- `Changed this session` is process-lifetime memory state. Only pending and
  Attention operations are durable across restart.
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
| Store UUIDs in frontmatter or a repository manifest | Alters user repositories solely for Chatbook bookkeeping and creates Git-visible metadata noise. |
| Store recovery bytes in the main database | Prevents independent recovery enumeration and couples recovery capacity/corruption to Database-note availability. |
| Ship macOS, Linux, and Windows writes together | Portable APIs do not offer equivalent beneath-root, atomic-displacement, permission, and durability guarantees. Read-only support still provides immediate value everywhere. |
| Stage and swap a main/recovery database pair | The recovery DB is new and main changes are additive. Pair swapping adds live-handle, sidecar, downgrade, and launcher coordination without protecting existing paired state. |
| Add Git controls in the first release | Expands stateful risk before file authority, conflict, deletion, and recovery behavior are proven. |

## Consequences

### Benefits

- The filesystem and Git remain the unambiguous source of truth.
- Existing Database-note behavior is protected structurally instead of by
  auditing every legacy caller for a discriminator.
- Milestone A can deliver a useful cross-platform Notes UI before writable
  filesystem guarantees are ready.
- Selected recovery remains independently inspectable if the main projection
  database is unavailable.
- A separate workbench/controller prevents the file editor state machine from
  further concentrating responsibility in `LibraryScreen`.

### Accepted trade-offs

- A fresh Chatbook database or clone assigns new local UUIDs.
- Missed or ambiguous external moves appear as Missing plus a new identity
  until explicitly reassociated.
- Opaque frontmatter is preserved but not structurally edited.
- Recovery consumes local plaintext disk and does not protect against loss of
  the device.
- Linux and Windows remain read-only in the first writable delivery.
- One active root and one cooperative current installation are support limits,
  not claims of multi-profile coordination.
- Moving the configured main database or user-data directory after recovery
  pairing can make file commands read-only until the exact paired recovery store
  is restored; the first rollout does not auto-relocate it.
- File/database consistency is repaired through journaling and reconciliation,
  not a nonexistent cross-resource transaction.
- Symlink and hardlink mutation, network/cloud write guarantees, ACL/xattr
  preservation, multi-host writing, folder mutation, and Git operations remain
  outside the initial guarantees.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md)
- [TASK-399](../tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
- [ADR-001: canonical ADR workflow](001-adopt-backlog-decisions-as-canonical-adrs.md)
- [ADR-003: Settings/Library/RAG ownership](003-settings-library-rag-defaults.md)
- [ADR-004: storage restart boundary](004-settings-storage-defaults-restart-boundary.md)
- [ADR-008: remote Sync v2 contract](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-011: Workbench UI system](011-chatbook-workbench-ui-system.md)
- [ADR-015: shell destination ownership](015-shell-destination-ia.md)
