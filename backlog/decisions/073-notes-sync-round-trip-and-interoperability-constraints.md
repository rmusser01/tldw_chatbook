# ADR-073: Notes Sync Round-trip and Interoperability Constraints

Status: Accepted
Date: 2026-08-12
Related Design: [Notes Folder Import and Lasting Sync](../../Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md)
Amends: [ADR-059](059-notes-folder-import-and-device-local-sync-ownership.md)
Supersedes: N/A

Allocation note: ADR-059 was initially unallocated in the 2026-08-12 checkout,
but the ownership decision moved there after upstream assigned ADR-058. This
amendment was originally allocated ADR-060, then renumbered after add-commit
provenance showed that the Prompt batch-mutation decision had claimed ADR-060
earlier on the same date. A concurrent remote then claimed ADR-072 before this
implementation plan landed, so the amendment moved to the next free ADR-073.

## Decision

ADR-059 remains the authority for the Notes folder and device-local lasting-sync
model. This ADR adds the filesystem round-trip, binding, portability, and Sync-v2
boundaries required to implement that model without ambiguous ownership or silent
data transformation.

Each Database Note has at most one active lasting filesystem binding across all
local and server-backed roots on a device. Each normalized root-relative path and
observed single-link regular-file identity binds to at most one note. Setup,
migration, retargeting, and reconciliation fail closed on duplicate ownership.

Bidirectional admission is byte-representation aware. For supported UTF-8 text,
the binding records whether a BOM exists, the newline convention, and final-newline
presence. Writes preserve that profile. Mixed-newline, unsupported-encoding,
hard-linked, aliased, non-regular, or otherwise unsafe files are not silently
rewritten; where safe reading is possible they may be offered as folder-to-Notes,
otherwise they are skipped with an actionable reason.

For a newly published file with no existing profile, the deterministic default is
UTF-8 without BOM, LF line endings, and final-newline presence matching the note
body. In one-way roots, a change on the non-authoritative side pauses before an
automatic overwrite. The conflict resolver may explicitly choose either side for
that occurrence without changing the configured direction. Deletion remains
confirm-first in every direction.

Filesystem replacement preserves the supported mode and metadata defined by the
platform adapter. Preflight reports metadata that cannot be preserved and withholds
bidirectional capability when losing it would be unsafe. Database Notes sync may
reuse centralized path-containment, file-identity, atomic-replacement, and metadata
preservation primitives, but it does not reuse File Notes tables, editor authority,
recovery ownership, or high-level write orchestration.

A local folder-row subtree mutation may be atomic inside its owning database. A
managed subtree move spanning files and local or server records is instead one
resumable composite journal operation containing deterministic child operations.
Interruption exposes resume, restoration, or attention; the product never describes
the cross-authority result as atomic.

The existing Sync-v2 M1 `notes.note` contract continues to carry note content and
note lifecycle only. It does not implicitly gain folder membership or filesystem
ownership fields. Local folder membership is device-local organization unless a
separately versioned Sync-v2 folder domain is designed. Direct server-backed folder
operations use the server Notes folder capability required by ADR-059.

ChaChaNotes folder entities and memberships participate in the existing ChaChaNotes
backup and restore boundary. The device-private root registry, bindings, watcher
state, journals, and recovery copies do not participate in portable Chatbook export
or generic database backup/restore. They are device-bound recovery state, not an
off-device backup. A future explicit device-local export must restore roots paused,
redact or revalidate paths, and require a complete dry-run before activation.
Managed memberships restored without a matching device-local owner remain visible
but inactive until a restore review converts them to manual membership or removes
that organization.

Every server-backed mutation includes the current opaque claim token and version.
Takeover fences the former owner at the service boundary. Stale claims, expired
authentication, profile changes, and lost write capability pause the affected root
before Chatbook performs another local destructive mutation; direction is never
silently downgraded.

Title changes that imply filename changes and managed folder changes that imply
physical moves are explicit previewed filesystem actions. Root retargeting pauses
the root, performs a complete dry-run, and never infers deletions from absence in the
new directory. One-time **Update existing** independently previews content change and
folder-membership addition.

## Context

The first written-spec review of ADR-059 exposed several implementation choices that
could otherwise create competing writers or data loss. Existing safe Notes path code
already rejects files with multiple hard links. Existing Sync-v2 M1 note envelopes
carry title and content but no folder membership. Existing ChaChaNotes backups would
naturally include folder tables, while restoring active physical roots and recovery
journals on another device would be unsafe.

Text-mode reads and replacement writes can also normalize BOMs and newlines or lose
permissions and extended metadata unless the representation is made part of the
binding contract. Finally, a directory move that crosses a filesystem and one or two
database authorities cannot provide the same atomic guarantee as a single local
folder-table transaction.

## Required Boundaries

- Binding uniqueness is enforced transactionally in the device-private sync owner
  and revalidated before every mutation.
- Only single-link regular files with stable identity are admitted for bidirectional
  writes.
- Serialization and supported metadata profiles are captured before the first write
  and verified after replacement.
- Composite moves use durable parent and child journal states with deterministic
  replay and no copy/delete fallback advertised as atomic rename.
- Sync-v2 M1 note payloads remain folder-free; any new folder domain requires a
  separately versioned contract and ADR review.
- Portable exports and generic backups never reactivate device-bound roots.
- Restored managed memberships cannot imply an active owner that was not restored.
- Server claim fencing applies to every mutation, not only setup and takeover.
- Retarget, rename, and organization side effects remain previewed user actions.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Allow one note to bind into several roots and rely on optimistic versions | Creates competing filesystem authorities and durable conflict loops. |
| Normalize every supported text file to UTF-8 without BOM and LF | Makes harmless note edits rewrite source representation and creates noisy external diffs. |
| Treat hard links as ordinary duplicate paths | A mutation through an unseen alias defeats path-based recovery and identity checks. |
| Promise atomic managed subtree moves | No atomic transaction spans disk, local SQLite, and a remote Notes service. |
| Add folders to the existing `notes.note` payload | Silently changes the locked Sync-v2 M1 contract and conflates device organization with server folder authority. |
| Include the private sync database in ordinary backups | Restoring physical paths, leases, and pending journals on another device can activate stale authority. |
| Validate a server claim only during setup | A takeover or permission loss after setup would leave the former owner able to mutate local files against stale remote authority. |

## Consequences

### Benefits

- A note and file have one unambiguous lasting-sync owner.
- Note edits do not silently normalize compatible source files or discard supported
  filesystem metadata.
- Interrupted multi-item moves remain inspectable and recoverable.
- Sync-v2, direct server Notes, backup, and device-private recovery have explicit,
  non-overlapping ownership.
- Claim takeover and authentication loss cannot leave an unfenced writer running.

### Accepted trade-offs

- Bindings store a small serialization and metadata manifest.
- Some readable files are restricted to folder-to-Notes rather than bidirectional
  sync.
- Managed subtree operations need more journal state and may require item-level
  attention after partial completion.
- Folder portability through Sync-v2 is deferred until it has an explicit domain
  contract.
- Device-private sync recovery needs its own future export UX if users require
  off-device preservation.

## Links

- [ADR-008: Sync-v2 Client M1 Contract Alignment](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-021: File-Backed Notes Disk Authority and Recovery](021-file-backed-notes-disk-authority-and-recovery.md)
- [ADR-027: Portable Database Note Session Coordinator](027-portable-database-note-session-coordinator.md)
- [ADR-029: Local Private Data Boundary](029-local-private-data-boundary.md)
- [ADR-055: Library Destructive Action Reversibility](055-library-destructive-action-reversibility-rule.md)
- [ADR-059: Notes Folder Import and Device-Local Sync Ownership](059-notes-folder-import-and-device-local-sync-ownership.md)
