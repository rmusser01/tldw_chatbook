# ADR-059: Notes Folder Import and Device-Local Sync Ownership

Status: Accepted
Date: 2026-08-12
Related Design: [Notes Folder Import and Lasting Sync](../../Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md)
Supersedes: N/A

Allocation note: ADR-058 was initially unallocated in the 2026-08-12 checkout,
but was assigned upstream before this branch merged. The Notes decision moved to
the next available identifier during rebase.

## Decision

Database Notes use hierarchical folder entities plus many-to-many memberships
with semantic parity between local Chatbook and `tldw_server`. Local and server
physical schemas may differ. A normalized service owns the shared behavioral
contract.

Folder membership has two ownership classes:

- **manual membership**, owned by the user or created by a one-time import; and
- **sync-managed membership**, owned by one opaque lasting-sync root and derived
  from a bound file's relative path.

A synced note may retain its one managed placement and any number of additional
manual placements. Manual membership changes never mutate files. A managed
rename or move is an explicit guarded filesystem action.

One-time import and lasting sync share a user entry point but not authority:

- **Import once** copies selected files or a recursive directory into ordinary
  Database Notes and manual folders. The selected directory itself is the
  top-level folder. The relationship ends after the import receipt.
- **Keep a folder synced** creates a device-local root only after explanation,
  path-safety checks, and an approved dry-run. Multiple roots are supported;
  bidirectional is the default, with folder-to-Notes and Notes-to-folder choices.

Lasting sync is not stored as more columns on each note and is not stored on the
server. A separate, owner-only, profile-scoped local SQLite owner stores root
paths, bindings, hashes, versions, import receipts, reconciliation cursors,
journal states, recovery content, capacity, and retention. Physical paths,
hashes, watcher state, import provenance, and recovery content never enter
server payloads or ordinary persistent logs.

Cross-boundary mutations use a durable local journal. Required recovery content
is admitted before destructive mutation. Each operation records intent, mutates
the file/note authorities through their normal guarded services, verifies both
outcomes, updates the binding, and completes last. Interruption either resumes
against matching observations or produces **Needs attention** with recovery
choices. Pending, unresolved, and Undo-eligible recovery cannot be evicted to
admit new destructive work. Normal recovery retention is 30 days.

Filesystem notifications are scheduling hints only. Versions, canonical paths,
and hashes drive reconciliation. Chatbook performs full reconciliation at root
activation, startup, and **Sync now**. Missing roots are **Offline**, never mass
deletion. Deletions and both-side changes pause for explicit review. A large
deletion burst is grouped at root level.

One cross-process coordinator lease owns watcher and mutation authority for a
root. Passive processes may display status but cannot reconcile or write.
Shutdown closes admission and reaches a completed or durable journal state before
releasing ownership and entering generic teardown.

For server-backed Notes, the server may retain logical folders and an opaque
root/device claim. It must not receive the local path or filesystem state. At
most one active filesystem claim owns a server note; another device requires an
explicit takeover. Source-owned membership APIs remove or convert only the
calling root's memberships. Bulk membership and incremental-change contracts
prevent per-note polling. `tldw_server` must record these server-side decisions
in its own separately allocated, cross-linked ADR before implementation.

File Notes remains a separate disk-authoritative feature under ADR-021 and
ADR-029. Database Notes sync does not reuse File Notes tables, recovery store,
editor authority, or write paths.

## Context

Library Database Notes currently show a flat list. Their Import action creates
one note from one chosen file, while a separate legacy Sync panel configures one
directory and persists file metadata directly on note rows. That flow does not
explain one-time versus continuing authority before source selection, cannot
represent several roots, and has no durable cross-boundary journal or folder
ownership.

Users want to import existing directory structures, keep selected directories
continuously synchronized, organize notes manually in a folder tree, and see the
same model for local and server-backed Notes. These requirements make a single
`folder_path` column or a purely virtual tree insufficient. Many-to-many
membership is also necessary because a synced note can remain in its managed
file hierarchy while appearing in additional organizational folders.

Disk, local SQLite, and a remote server cannot share one atomic transaction.
Physical sync state also contains sensitive device-local information that must
not become server note metadata. Durable local orchestration and recovery are
therefore separate from both note authorities.

Multiple Chatbook processes and multiple client devices introduce different
ownership races. A local coordinator lease prevents two processes from mutating
one directory, while an opaque server claim prevents two devices from silently
binding different filesystems to one server note.

## Required Boundaries

- Folder behavior is shared through typed models and service operations, not by
  copying one repository's physical schema into the other.
- Local folder rename/move is an optimistic, atomic subtree operation with
  complete collision validation before mutation.
- Manual memberships and each sync owner's managed memberships remain
  distinguishable and independently removable.
- Generated ancestor memberships may collapse for display; explicit manual
  memberships retain every intended placement.
- Tree placement identity includes folder/membership context while edit identity
  remains the single note ID.
- One-time import matching is device-private and best-effort. It never updates a
  note from a title-only or uncertain match without user confirmation.
- A lasting root binds only approved discovered files and explicitly selected
  notes. It never exports every unbound Database Note.
- Bidirectional file formats are initially one-file-to-one-note UTF-8 text. The
  relative path owns binding, filename stem owns displayed title, and complete
  file text owns note body without heading/frontmatter transformation.
- Sync roots reject canonical overlap with each other, File Notes roots, and
  application-owned private paths. Symlink roots and directory-symlink traversal
  fail closed.
- Unsafe or insufficiently supported writable filesystems do not silently
  receive bidirectional capability.
- Root disconnection never deletes files or notes. It explicitly converts or
  removes only that root's managed memberships.
- Recovery admission precedes destructive action; capacity failure blocks the
  action.
- Persistent diagnostics exclude content, absolute paths, hashes, credentials,
  exception text, and recovery bytes.
- Server capability violations pause the root. Clients do not fall back to flat
  writes, unmanaged folder changes, or unowned claims.
- Existing legacy sync metadata migrates to one or more paused candidate roots.
  Migration performs no file or note mutation and requires a new dry-run before
  activation.
- File Notes authority and storage stay independent.

## User-facing Consequences

- Database Notes exposes **Add from files…**, which asks **Import once** or
  **Keep a folder synced** before opening a source picker.
- A selected imported directory appears as its own top-level folder.
- Sync roots appear as decorated top-level folder nodes with state, pause,
  settings, **Sync now**, attention, and disconnect controls.
- Conflicts show **Keep file**, **Keep note**, and **Keep both**. Deletions show
  delete/archive counterpart, restore missing side, or disconnect item.
- Status uses semantic theme colors plus glyphs and plain-language labels; color
  is never the only signal.
- Library's general ingestion entry is **Import media…** with destination-specific
  explanatory copy.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Derive the entire folder tree virtually from sync relative paths | Manual local folders, server folders, and virtual sync folders would have different identities and mutation rules in one UI. |
| Store one folder path directly on each note | Cannot represent many-to-many organization, folder identity/versioning, or atomic subtree operations. |
| Keep folder and sync ownership only in ChaChaNotes | Couples device-private paths/recovery to the note authority and cannot orchestrate server-backed Notes independently. |
| Put local paths and hashes in server note metadata | Leaks filesystem information and makes another device appear to share the same physical authority. |
| Let two devices synchronize one server note and rely on optimistic versions | Produces persistent conflict loops and cannot identify which device owns managed memberships. |
| Use filesystem notifications as the source of truth | Events can be dropped, duplicated, reordered, or misreported across remounts and platform backends. |
| Automatically choose the newest side in a conflict | Timestamps do not establish user intent or reliable authority. |
| Propagate deletion automatically | A missing/remounted/emptied root can appear as a destructive batch and cause data loss. |
| Run the legacy single-root engine alongside the new registry | Two active sync owners would race and preserve contradictory configuration models. |
| Merge Database Notes sync with File Notes | File Notes is disk-authoritative; Database Notes sync intentionally coordinates separate note and file authorities. |

## Consequences

### Benefits

- One coherent folder tree supports local Notes, server Notes, manual organization,
  one-time imported structures, and lasting roots.
- Ownership-aware memberships allow extra organization without accidental file
  moves or cross-root removal.
- Device-private journaling and recovery make interrupted cross-boundary changes
  inspectable and recoverable.
- Multiple processes and devices have explicit mutation ownership.
- Server payloads avoid leaking physical filesystem state.
- The flow explains authority before action and makes deletion/conflict behavior
  explicit.

### Accepted trade-offs

- Delivery requires a local schema migration, a new private sync database,
  coordinator lifecycle, and substantial server contract work.
- Server-backed parity cannot complete until `tldw_server` publishes the required
  versioned capability and its own ADR.
- A server note may have only one active filesystem binding; cross-device takeover
  is explicit rather than automatic.
- Bidirectional sync initially excludes structured/multi-note containers and
  unsafe writable roots.
- Recovery duplicates overwritten content locally for a bounded period and needs
  disclosed capacity/retention controls.
- Folder trees and repeated manual placements require placement-aware UI state and
  bulk/lazy loading.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-021: File-Backed Notes Disk Authority and Recovery](021-file-backed-notes-disk-authority-and-recovery.md)
- [ADR-027: Portable Database Note Session Coordinator](027-portable-database-note-session-coordinator.md)
- [ADR-029: Local Private Data Boundary](029-local-private-data-boundary.md)
- [ADR-055: Library Destructive Action Reversibility](055-library-destructive-action-reversibility-rule.md)
