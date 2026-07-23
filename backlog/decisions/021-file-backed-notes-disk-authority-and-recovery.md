# ADR-021: File-Backed Notes Disk Authority and Recovery Replica

Status: Proposed
Date: 2026-07-22
Related Task: [TASK-399](../tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
Supersedes: N/A

## Decision

Existing linked Markdown/text files are the sole content authority for
file-backed Notes. Chatbook will mutate them only through a hash-checked,
journaled file coordinator; keep a derived projection and local UUID metadata in
the main database; and keep mutation safety plus opt-in self-contained recovery
replicas/history in a separate `notes_recovery.db`.

File-backed Notes remain in Library's Local Notes owner beside, but visibly
separate from, existing Database notes.

## Context

Users may already have thousands of notes in a folder hierarchy managed by Git
and other editors. Their required workflow is:

1. open and edit those notes in Chatbook;
2. see the real working-tree changes immediately;
3. use ordinary Git commands to stage, commit, and push them.

The legacy folder-sync design treats SQLite and disk as competing peers and
uses direction/winner policy. That is unsuitable for a Git working tree:
timestamp winners can choose incorrectly, moves and deletions are ambiguous,
and writes can introduce noise or duplicate identity.

Chatbook still needs local UUIDs, metadata, FTS, links, optional RAG, Console
handoff, crash recovery, and protected history. Those capabilities require
SQLite, but not SQLite content authority.

SQLite and a filesystem cannot share one atomic transaction. Safe direct file
editing therefore needs a durable operation journal, exact-byte safety copies,
expected raw hashes, atomic same-directory replacement, result verification,
and startup reconciliation.

The user also wants recovery storage independent of the primary Notes
projection for selected files/folders. A dedicated recovery database satisfies
that requirement without writing Chatbook IDs or manifests into the repository.
It is a same-device plaintext recovery replica, not an off-device disaster
backup.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Continue equal-peer bidirectional folder sync | Ambiguous authority, timestamp winners, recurring conflicts, incomplete move/delete identity, and avoidable Git noise. |
| Make SQLite canonical and export/import files | Chatbook edits would not be immediate working-tree changes, and external Git/editor changes would require a second authority loop. |
| Use the filesystem without a database projection | Degrades Library FTS, metadata, stable local links, Console handoff, RAG invalidation, and large-tree responsiveness. |
| Store stable UUIDs in frontmatter or a repository manifest | Alters source files for Chatbook bookkeeping and creates unwanted Git-visible metadata. |
| Store history/recovery payloads in the main database | Does not provide independent recovery enumeration when the projection database is unavailable and couples recovery quota/corruption to Database notes. |
| Infer external moves from inode or matching content hashes | Risks attaching metadata/history to the wrong file after reuse, copying, or ambiguous bulk operations. |
| Add automatic Git staging/commit/push with file support | Expands stateful risk before file authority, conflict, deletion, and recovery behavior are proven. |

## Consequences

### Required boundaries

- File bytes and exact relative path own content and title; the main database is
  a derived body/title projection.
- `notes.storage_kind` distinguishes immutable `database` and `file` ownership;
  `keywords.sync_eligible` keeps file-only keyword definitions local until
  explicit Database-note promotion.
- Local UUIDs are not portable and are never injected into source files.
- One process-wide coordinator is the only file mutation and internal projection
  authority.
- Every mutation compare-checks a raw hash, journals its intent and any bytes
  whose only accessible copy could be displaced, uses same-directory atomic
  replacement/no-replace publication where applicable, and verifies the
  resulting paths and hashes.
- Normal operations remain pending until the idempotent main binding/projection
  transition commits, then complete in the recovery journal last; explicit
  recovery-only restores record a deferred main rebuild.
- Watcher/poll events are hints; current filesystem state and hashes determine
  truth. There is no timestamp winner or automatic merge.
- A separate `notes_recovery.db` contains self-contained revision and operation
  rows. Protected notes always have one verified current replica after a
  completed Chatbook save.
- Confirmed Chatbook deletion is enabled only with a verified exact snapshot,
  30-day retention, and a working minimal restore path.
- File-backed rows and disk are excluded from legacy sync ownership, Sync v2
  triggers/envelopes, generic public DB update/delete paths, and MCP writes.
- Database notes retain existing behavior and remain available if recovery
  storage fails.
- The first rollout permits only one active writable root; this is a release
  gate, not a schema limitation.

### Accepted trade-offs

- A fresh Chatbook database or clone assigns new local UUIDs.
- Ambiguous/missed external moves appear as Missing plus a new identity.
- Exact frontmatter is opaque and cannot be edited as structured metadata in
  Chatbook.
- Recovery consumes local disk and is unencrypted; quota or corruption can make
  file-backed commands read-only.
- File and database state are repaired by journaling/reconciliation rather than
  a nonexistent cross-resource transaction.
- The final compare/mutate race with an uncooperative external process cannot be
  eliminated for every replace/move/unlink primitive; immediate verification
  preserves evidence and surfaces Attention.
- Symlinks, hardlink mutation, network/cloud mounts, ACL/xattr preservation, and
  multi-host writing are outside guarantees.
- Git operations remain external until separately approved.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md)
- [TASK-399](../tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
- [ADR-001: canonical ADR workflow](001-adopt-backlog-decisions-as-canonical-adrs.md)
- [ADR-003: Settings/Library/RAG ownership](003-settings-library-rag-defaults.md)
- [ADR-004: storage restart boundary](004-settings-storage-defaults-restart-boundary.md)
- [ADR-008: remote Sync v2 contract](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-011: Workbench UI system](011-chatbook-workbench-ui-system.md)
- [ADR-015: shell destination ownership](015-shell-destination-ia.md)
