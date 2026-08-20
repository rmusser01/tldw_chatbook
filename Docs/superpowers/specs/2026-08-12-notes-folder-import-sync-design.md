# Notes Folder Import and Lasting Sync — Design

Date: 2026-08-12
Status: Approved written specification (ready for slice planning)
ADRs:
[ADR-059](../../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md),
[ADR-073](../../../backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md)

## Summary

Replace the Database Notes toolbar's separate, under-explained **Import** and
**Sync** entry points with one **Add from files…** flow. The first step asks the
user to choose between:

- **Import once** — copy selected files or a recursively scanned folder into
  Database Notes without creating a lasting relationship; or
- **Keep a folder synced** — explain the continuing relationship, let the user
  choose a directory and direction, show a dry-run, and then create a persistent
  sync root.

Database Notes gain hierarchical folders and many-to-many folder membership,
semantically matching `tldw_server`. One-time folder imports include the selected
directory itself as the top-level folder. Lasting sync supports multiple roots,
each with independent direction, state, pause, manual sync, settings, and
disconnect controls. Bidirectional is the default direction, but the user can
choose folder-to-Notes or Notes-to-folder during setup.

The Library-wide source-import action is named **Import media…** and explains that
documents, web pages, audio, video, and other source material are entering the
Media library. Notes use **Add from files…** inside Notes, so the two destinations
cannot be mistaken for one another.

## Current State

- Library Database Notes currently render a flat list.
- The Notes **Import** button opens a single-file picker and creates one note.
- The separate Notes **Sync** panel persists one directory plus direction,
  conflict, and auto-sync settings. Existing note rows carry file path, relative
  path, root, hash, mtime, and sync flags.
- The legacy sync engine models disk and the Notes database as peers, but it has
  no durable multi-root registry, cross-process coordinator, operation journal,
  independent recovery admission, or folder ownership model.
- Local ChaChaNotes does not have the hierarchical note-folder entities now
  present in `tldw_server`.
- `tldw_server` has hierarchical folder rows and separate manual/source-managed
  membership concepts. Chatbook's server adapter currently discards returned
  folder data and lacks the mutation, bulk-membership, incremental-change, and
  opaque binding-claim APIs required by this design.
- File Notes is a separate disk-authoritative feature governed by ADR-021 and
  ADR-029. It must not be merged with Database Notes sync.

## Goals

1. Explain the difference between a one-time copy and a continuing sync before
   the user chooses a source.
2. Accept either individual files or one directory for one-time import; scan a
   directory recursively and preserve its hierarchy, including the selected
   root label.
3. Provide full hierarchical folder organization for local and server-backed
   Database Notes: create, rename, move, and remove folders; move notes; and
   attach one note to multiple folders.
4. Match the server's folder semantics through a normalized service contract,
   without requiring identical local and server database schemas.
5. Support multiple persistent sync roots with independent configuration and
   lifecycle controls.
6. Default new lasting roots to bidirectional sync while allowing direction
   choice during setup.
7. Detect changes continuously while Chatbook is running, reconcile at startup,
   and expose **Sync now**.
8. Pause conflicts and deletion propagation for explicit review.
9. Make cross-boundary file/note operations recoverable across interruption.
10. Keep absolute paths, hashes, watcher data, import provenance, and recovery
    content device-private.
11. Work in large note trees without blocking the Textual event loop.
12. Use semantic text coloring plus glyphs and plain-language labels so color is
    never the sole state signal.

## Non-goals

- Merge File Notes and Database Notes into one storage or write model.
- Turn every existing Database Note into a file when a root is activated.
- Synchronize JSON, YAML, CSV, or another multi-note container bidirectionally.
- Send local directory paths, file hashes, recovery bytes, or watcher state to a
  server.
- Follow directory symlinks or silently admit unsafe/unsupported writable roots.
- Propagate file or note deletion without explicit confirmation.
- Make a server API capability optional by falling back to unowned, flat writes.
- Implement the complete program in one pull request.

## Fixed Product Decisions

- One-time import accepts individual files or one recursive directory.
- A selected directory appears as the top-level imported folder.
- Repeated one-time import defaults to **Skip** for unchanged matches and
  **Create new** for changed matches, with per-item overrides.
- Imported folder memberships are ordinary manual organization and can be
  changed later.
- Folder organization is fully editable and many-to-many.
- A sync-managed note may also belong to any number of manual folders.
- Multiple lasting roots may be active concurrently.
- Bidirectional is the default direction; the setup screen also offers
  folder-to-Notes and Notes-to-folder.
- While Chatbook runs, filesystem events prompt debounced checks; authoritative
  reconciliation uses versions, paths, and hashes.
- Startup and **Sync now** perform complete reconciliation.
- Both-side changes become **Needs attention** with **Keep file**, **Keep note**,
  and **Keep both**.
- Deletions require confirmation before affecting the counterpart.
- Renaming or moving inside a managed tree is an explicit guarded filesystem
  action.
- Local and server-backed Database Notes use the same UI and normalized model.
- One Database Note may have at most one active lasting filesystem binding across
  all roots, and one normalized root-relative path may bind to only one note.
- Bidirectional writes preserve the admitted file's text representation and
  supported filesystem metadata; ambiguous or unsafe files are never silently
  normalized.

## Terminology and Ownership

### Database Note

A note owned by local ChaChaNotes or by the configured server Notes service. It
is distinct from a File Note.

### Manual folder membership

User-owned organization. A user may create it directly or receive it from a
one-time import. Manual membership never changes a source file.

### Sync-managed folder membership

Organization derived from one lasting root and a bound file's relative path.
It is owned by an opaque sync-root identity. Ordinary folder operations cannot
silently remove it.

### Sync root

A device-local record describing one directory, one Notes destination/scope,
one direction, and its bindings. The root has a special top-level tree node and
links to the logical root folder used by note memberships.

### Binding

The device-private relationship among a sync root, one normalized relative file
path, and one Database Note identity. For server-backed Notes, the server sees
only an opaque claim identity and the note ID—not the physical path or hashes.

Binding uniqueness is global to the device-private sync owner, not merely to one
root. A Database Note can have at most one active lasting binding across local
and server-backed roots. Within a root, a normalized relative path and observed
file identity can resolve to at most one note. Setup, retarget, migration, and
reconciliation stop on duplicates instead of choosing a winner.

## User Experience

### Database Notes toolbar

Replace the adjacent **Sync** and **Import** actions with **Add from files…**.
Root-level **Sync now** and **Manage sync folders** remain available when at least
one lasting root exists. Export remains separate.

The flow is available only in Database Notes mode. File Notes retains its own
link/open-folder behavior and disk-authoritative copy.

### Mode chooser

The first in-canvas step contains two descriptive choices:

- **Import once** — “Copy individual files or a folder into Notes. Later changes
  to the originals are not tracked.”
- **Keep a folder synced** — “Create a lasting connection. Changes continue to
  move between the folder and Notes.”

No path is read and no state is mutated before the user chooses a mode.

### One-time source selection

- The user may choose several individual files. They then choose an existing or
  new manual destination folder.
- The user may choose one directory. It is scanned recursively, and its basename
  becomes the proposed top-level folder label.
- If the label collides with an existing folder, the preview asks the user to
  use the existing folder, create a unique sibling such as `Work (2)`, or enter
  another name. Trees never merge silently.
- The selected root folder is created only when at least one item is approved for
  import. Empty and unsupported-only branches do not create misleading folders.

### One-time preview

Scanning and parsing finish before persistence. The immutable preview groups
items as:

- new;
- unchanged repeat;
- changed repeat;
- uncertain match;
- unsupported; or
- failed.

Defaults are:

| Classification | Default action |
| --- | --- |
| New | Import |
| Unchanged repeat | Skip |
| Changed repeat | Create new |
| Uncertain match | Create new; update unavailable until the user confirms the match |
| Unsupported / failed | Skip with reason |

Approved items expose **Skip**, **Create new**, and—only for an exact or
user-confirmed match—**Update existing**. Updating shows a diff, uses the current
optimistic version, and creates the same reversible Library receipt owed by any
persisted destructive replacement.

For **Update existing**, content replacement and folder placement are separate
previewed decisions. The preview states whether the existing note will gain the
imported parent-folder membership, and the user may update content without adding
that placement or add the placement without replacing content. Neither change is
implied by approving the other.

If one structured one-time source produces several notes, every resulting note
inherits that file's parent folder. Structured or multi-note formats are never
eligible for lasting sync.

### Lasting-sync explanation and setup

Before the picker, the setup explains:

- Chatbook watches the directory only while running;
- startup and manual reconciliation still occur;
- bidirectional sync can change both files and Database Notes;
- conflicts and deletions pause for review;
- local paths, hashes, and recovery data stay on this device; and
- server content overwritten during recovery may be retained locally for up to
  30 days in owner-only storage.

The form collects:

- display name, defaulting to the selected directory name;
- directory;
- local or server-backed Notes destination/profile;
- direction, defaulting to bidirectional; and
- capability-dependent advanced settings.

Activation always follows a path-safety preflight and dry-run. It binds only
approved discovered files and explicitly selected existing notes. It never
exports unrelated Database Notes.

The display name is also the logical top-level managed folder label. Renaming it
changes the Notes tree label only; it does not rename the selected physical root
directory. Retargeting does not silently replace the display name.

### Direction semantics

Direction governs automatic propagation, not whether the user may explicitly
resolve an exception:

- **Folder-to-Notes** applies one-sided file creates, changes, renames, and moves
  to Notes. A note-side change becomes **Needs attention** before any later file
  update can overwrite it.
- **Notes-to-folder** applies one-sided note creates, changes, renames, and managed
  moves to files. An external file change becomes **Needs attention** before a
  later note write can overwrite it.
- **Bidirectional** applies a one-sided change toward the unchanged side and marks
  simultaneous changes as a conflict.

An explicit conflict resolution may choose either side once even when that choice
runs opposite the configured automatic direction. It does not change the root's
future direction. Deletions always require confirmation in every direction.

### Folder tree

The navigator renders manual folders, notes, and decorated sync-root nodes in
one lazy tree:

```text
▾ Personal
  ├─ Ideas
  │  └─ Garden redesign
  └─ Reading list

▾ ⇄ Work Notes  Up to date
  ├─ Projects
  │  └─ Alpha
  │     ├─ Plan
  │     └─ Decisions
  └─ Weekly review

▸ ⇄ Research Archive  2 need attention
```

Notes without memberships appear under the virtual **Unfiled** node. A note with
several explicit manual memberships appears in every intended branch while
opening any row resolves to the same note identity.

Generated ancestor memberships for the same sync owner collapse to the deepest
effective placement. Explicit manual membership in both an ancestor and a
descendant remains visible because it reflects user intent.

Tree row identity includes folder and placement identity in addition to note ID;
selection/open/edit identity remains the one note ID. Search results include a
breadcrumb for every shown placement.

### Sync-root controls

Every sync root exposes:

- state and last successful reconciliation;
- **Sync now**;
- **Pause** / **Resume**;
- settings and direction;
- attention items;
- retarget directory; and
- **Disconnect**.

Retargeting first pauses the root and scans the proposed directory. It presents
a complete dry-run using stored hashes and file identities, never interprets
absence from the newly selected directory as deletion, and activates only after
every ambiguous binding has been resolved or disconnected. The old directory
remains authoritative until activation commits; canceling leaves it unchanged
and paused or resumes it at the user's explicit choice.

Disconnect never deletes files or notes. It asks whether to:

1. **Keep folder organization** (default), converting that root's managed
   memberships to manual; or
2. **Remove synced organization**, removing only that root's managed
   memberships and leaving notes in their other folders or Unfiled.

### Media import wording

Every user-visible Library entry that routes to the content-ingestion canvas is
named **Import media…**. Its helper copy states that documents, web pages, audio,
video, and other source material are being added to the Media library. Internal
code may retain established `ingest` identifiers and server API names.

## Semantic Presentation

Use theme tokens/classes rather than hard-coded colors:

| Meaning | Presentation |
| --- | --- |
| Connected root, selected path, active link | Accent/blue |
| Up to date, completed | Success/green |
| Paused, offline, pending, needs attention | Warning/amber |
| Failed, unsafe, blocked | Error/red |
| Skipped, disabled, informational metadata | Muted text |

Every state also has a glyph and plain-language label. Monochrome themes,
reduced-color terminals, and screen-reader-like text capture retain the complete
meaning.

## Architecture

### 1. Local folder repository

Add a versioned local folder model with semantic parity to the server:

- stable folder identity;
- display name;
- normalized hierarchical path;
- parent identity;
- optimistic version and soft deletion;
- manual note memberships; and
- source-owned managed memberships.

The implementation may differ from the server's physical schema. The shared
contract is behavior and normalized data, not table identity.

Folder create/reuse is case-insensitive while preserving display casing. Rename
and move are subtree operations: validate the entire resulting path set, detect
case/Unicode collisions, update descendants atomically, increment affected
versions, and retain memberships.

Removing a manual folder previews affected descendants and memberships. It
removes organization only; notes survive in other folders or Unfiled. A managed
folder cannot be removed through ordinary manual-folder deletion.

### 2. Normalized folder service

Expose one typed service over local and server implementations:

- list/batch-load folder trees and memberships;
- create, rename, move, soft-delete, and restore folders;
- attach/detach manual memberships;
- reconcile one owner's managed memberships;
- convert or remove one owner's managed memberships during disconnect; and
- report source capabilities and actionable unsupported reasons.

The UI consumes this service and never branches on SQLite versus HTTP payload
shape.

### 3. Folder-tree state builder

A Textual-independent state builder owns:

- lazy folder nodes and pageable note children;
- placement-aware row identities;
- ancestor collapsing for generated memberships;
- explicit manual duplicates;
- sync-root decorations and semantic states;
- breadcrumbs;
- filter/search projection; and
- stable selection, focus, expansion, and scroll identities.

### 4. Import planner and executor

The planner performs bounded scanning, parsing, path mapping, folder-collision
analysis, and device-private receipt matching without mutations. It returns an
immutable plan with per-item proposed actions and reasons.

The executor accepts only that approved plan. Local work commits in bounded
batches under one durable import-session receipt. Server work uses idempotent
item identities where supported. Cancellation stops between batches; it does not
roll back already confirmed items or report them as missing. The final receipt
lists imported, updated, skipped, failed, and retryable-failure items. Retry
targets failures only.

The import ledger stores source path/provenance locally and treats recognition
after root move/rename as uncertain. It never guesses an update from title alone.
Its approved action model records content mutation and folder-membership mutation
separately, including when both target the same existing note.

### 5. Device-private sync store

A private, profile-scoped SQLite owner separate from ChaChaNotes stores:

- sync roots and their logical root-folder IDs;
- canonical physical paths and opaque server-origin IDs;
- root direction, state, capabilities, and last reconciliation cursor;
- relative-path/note bindings and last-known note versions;
- content hashes and last-known file identities;
- import receipts;
- durable operation journal entries;
- exact recovery content and metadata;
- retention and capacity accounting; and
- coordinator ownership state.

The database is owner-only under ADR-029 and independent enough to enumerate and
restore pending sync recovery without opening ChaChaNotes. Absolute paths,
content, hashes, and exception messages containing them are excluded from
persistent ordinary logs.

Folder entities and memberships stored in ChaChaNotes remain part of the normal
ChaChaNotes backup and restore boundary. The device-private sync database is
excluded from portable Chatbook export and generic database backup/restore:
physical roots, leases, journals, and recovery copies are device-bound recovery
state rather than a portable backup. Any future explicit device-local export must
restore roots paused, revalidate every path and claim, and require a complete
dry-run before activation.

Because a ChaChaNotes backup can therefore contain managed memberships without
their device-only owner, restore treats unmatched managed memberships as inactive.
After all local sync owners have loaded, a restore review offers conversion to
manual membership (default) or removal of that organization; it never activates
or invents a physical root. Until resolved, the placements remain visible and
clearly labeled **Restored — no sync folder**.

Recovery defaults to 30 days. Admission proves that required bytes and metadata
fit before a destructive file/note replacement begins. Pending, unresolved, or
Undo-eligible recovery is never evicted to admit another operation. A visible
quota error blocks the destructive action.

### 6. Sync-root registry and coordinator

The registry owns multiple roots and rejects canonical ancestor/descendant
overlap with:

- another configured lasting-sync root;
- active File Notes roots;
- application data, configuration, cache, runtime, database, and recovery paths;
  or
- another reserved root class introduced by an applicable ADR.

One cross-process coordinator lease in a fixed runtime namespace owns watcher and
mutation authority for a root. The lease is independent of UI screen lifetime and
profile-relative configurable paths. Passive Chatbook processes may display
status but cannot reconcile or mutate that root. Shutdown closes new admission,
finishes or durably journals the active operation, stops watchers, and releases
the lease before generic database/worker teardown.

The registry enforces one active lasting binding per Database Note across every
root it owns, whether the note is local or server-backed. It also enforces one
note per normalized root-relative path and stable file identity. These constraints
are revalidated during setup, legacy migration, retarget, claim takeover, and
before journaled mutation.

### 7. Reconciler

The reconciler is pure with respect to user content: given root configuration,
bindings, filesystem observations, note summaries, and last-known hashes/versions,
it returns a typed plan:

- no change;
- create/update either side;
- path/title move;
- conflict;
- deletion attention;
- missing/offline root;
- unsafe path or capability loss; or
- stale/contended server claim.

Filesystem notifications only schedule a debounced incremental reconciliation.
Startup and **Sync now** perform complete reconciliation. Server-backed roots use
incremental version/change cursors or event delivery when the capability exists;
they do not poll every note detail or issue per-note folder lookups.

### 8. Journaled executor

Each cross-boundary operation follows an explicit state machine:

1. validate current versions, hashes, containment, and destination availability;
2. admit and persist required recovery content;
3. persist journal intent and expected outcome;
4. mutate one authority;
5. mutate the counterpart through its optimistic/idempotent service;
6. update binding and managed membership state;
7. verify both observed outcomes; and
8. mark the journal entry complete.

An interrupted or failed entry is resumed only when observations still match its
expected state. Otherwise it becomes **Needs attention** with **Resume**,
**Restore**, and **Disconnect item** choices. General worker cancellation cannot
erase or skip a journal stage.

A managed subtree move is a composite journal operation whose ordered child
entries cover affected files, note mutations, and managed memberships. Only the
single-authority local folder-row mutation may be described as atomic. The
cross-authority move is resumable and recoverable but not atomic; partial progress
is visible, and deterministic replay never substitutes copy/delete for a failed
rename without fresh approval.

### 9. Watcher coordinator

Watchers are hints, coalesce bursts, and never apply mutations. Missing roots are
**Offline**, never mass deletion. Directory symlinks are not traversed; symlink
roots are rejected; symlinked files are skipped with a visible reason. Unsafe,
network, cloud-synchronized, or insufficiently supported filesystems may offer
folder-to-Notes but cannot silently enable bidirectional writes.

Bidirectional admission requires a stable single-link regular file. Hard-linked
or otherwise aliased files are marked unsupported for writing because another
path could mutate the same inode outside the binding and recovery boundary. If
safe reading remains possible, setup may offer folder-to-Notes with the limitation
shown in its dry-run.

A large deletion burst is grouped into one root-level attention event with a
preview and explicit batch resolution rather than hundreds of prompts.

## File Mapping and Round-trip Rules

Lasting sync initially accepts stable, one-file-to-one-note UTF-8 text formats,
including Markdown and plain text.

- The normalized relative path owns binding identity.
- Filename stem owns the displayed Database Note title.
- The complete file text owns the note body. Headings and frontmatter are not
  stripped, regenerated, or canonicalized.
- Editing the title of a bound note invokes the guarded filesystem rename flow.
- Moving the note within its managed tree invokes the guarded filesystem move
  flow.
- Adding/removing a manual membership never renames or moves the file.
- A Database Note explicitly selected for initial Notes-to-folder publication
  receives a previewed, sanitized filename and collision check.

Each writable binding stores a serialization profile captured from bytes before
the first write: UTF-8 BOM presence, newline convention, and final-newline
presence. Notes-to-file writes preserve that profile. Mixed-newline files,
unsupported encodings, and undecodable bytes are never silently normalized; the
dry-run either restricts them to folder-to-Notes or skips them with a reason.
For a newly published file with no prior profile, the preview states the
deterministic default: UTF-8 without BOM, LF line endings, with final-newline
presence matching the Database Note body exactly.

Writable preflight also records the platform-supported replacement metadata that
must survive a write, including file mode and any supported ACL, extended
attribute, ownership, or platform flag fields. The platform adapter must prove it
can preserve and verify required metadata before enabling bidirectional writes.
When it cannot, the root or item is read-only/folder-to-Notes. Database Notes sync
may share centralized containment, identity, atomic-replacement, and metadata
preservation primitives with File Notes, but it retains separate tables, editor
authority, recovery ownership, and high-level write orchestration.

These rules avoid progressive file transformation across repeated syncs.

## Conflict, Deletion, and Move Behavior

### Both sides changed

Pause only the affected binding. Show file and note versions, timestamps,
relative path, and a text diff:

- **Keep file** — admit recovery, update the bound note, verify, and retain Undo.
- **Keep note** — admit exact file recovery, update the file, verify, and retain
  Undo.
- **Keep both** — keep the file as the bound version and create the conflicting
  Database Note content as a selected, unbound note in a child named after the
  root under the manual **Conflict copies** folder. The receipt names both outcomes
  and offers Undo while recovery is retained.

Bulk resolution is allowed only when every selected item receives the same
explicit choice.

### One side deleted

The binding becomes **Needs attention**. The user may:

- delete/archive the counterpart;
- restore the missing side from the survivor; or
- disconnect the item and retain the survivor.

Database Notes use soft delete and the Library receipt/Undo convention from
ADR-055. File removal first admits exact recovery and then uses a platform-safe
recoverable operation; failure leaves the journal unresolved. Root absence or a
large deletion burst never counts as confirmation.

### Managed move or rename

Show old/new paths and affected descendants. Before action, recheck containment,
current hash/version, destination absence, filesystem capability, recovery
capacity, and server claim. A destination collision or external change blocks the
move and creates attention; copy/delete fallback is not silently substituted for
an unavailable atomic move. A title change that would rename a bound file first
shows the proposed old and new filenames; saving note content does not implicitly
approve that rename. Case-only renames use a platform-proven safe operation or
remain blocked for explicit resolution.

Moving a managed folder subtree uses the composite journal described above. The
UI says **Resumable move**, reports completed and pending children, and never
claims that the filesystem-plus-database operation is atomic.

## Server-backed Contract

`tldw_server` requires a separate, cross-linked ADR and versioned capability. No
ADR number or future task ID is assigned from this repository.

The server contract must provide:

- folder list/create/rename/move/delete/restore;
- atomic subtree path mutation with optimistic versions;
- bulk folder memberships with paginated note summaries or a batch endpoint;
- manual membership attach/detach;
- source-owned managed-membership reconciliation and conversion/removal;
- opaque filesystem-binding claim, release, explicit takeover, and fenced
  mutation token/version;
- incremental note change/version discovery; and
- idempotent mutation identities suitable for retry.

The server may retain an opaque device/root claim and logical folder paths. It
must never receive the local absolute path, file identity, content hash, watcher
state, recovery payload, or private import provenance.

At most one active filesystem claim may own a server note. Setup that encounters
another claim shows **Managed by another device** and requires explicit takeover.
Takeover invalidates the old opaque claim; it cannot infer or delete the other
device's files. Every server-backed note, folder, and managed-membership mutation
includes the current claim token/version; validation occurs at the service
boundary rather than only during setup.

Expired authentication, changed profile identity, stale claims, and lost folder
or note write capability pause the root before another local destructive action.
Chatbook does not silently downgrade direction. The user may reauthenticate,
resolve takeover, choose a supported direction in settings, or disconnect.

An older server may show a read-only folder grouping when its responses contain
sufficient folder data. Otherwise it retains flat Notes access. Unsupported
mutations and lasting sync are disabled with upgrade guidance. If a server
advertises a capability and then violates it, the root pauses; Chatbook does not
fall back to flat writes or unowned memberships.

### Sync-v2 interoperability

The existing Sync-v2 M1 `notes.note` envelope remains responsible for note title,
content, lifecycle, and its established version metadata. Filesystem-driven local
note writes still pass through the normal Database Note service, so its existing
Sync-v2 outbox behavior is preserved.

Folder entities, memberships, managed ownership, physical bindings, and claim
tokens are not added to that M1 payload. Local folder membership remains
device-local organization when a local-first Note is mirrored through Sync-v2.
Portable folder synchronization requires a separately versioned Sync-v2 folder
domain and contract; it is outside this design. A root targeting server-backed
Notes directly uses the server Notes folder and claim capability described above,
not an inferred extension of `notes.note`.

## Migration

The local schema migration creates folder storage without assigning invented
folders. Existing active Database Notes appear under **Unfiled**.

Legacy sync migration examines both configuration and per-note metadata. It
creates one paused candidate root per distinct canonical safe root. It preserves
recognizable bindings without touching files or notes. Missing, overlapping,
out-of-root, duplicate, or invalid metadata is attached to a migration review
report rather than silently repaired.

No candidate watches or synchronizes until the user opens it, reviews the
dry-run, chooses direction and collision outcomes, and explicitly activates it.
Retired legacy configuration remains readable only for migration/rollback until
the implementation's documented compatibility window ends; it is not maintained
as a second active sync engine.

Because the change bumps the ChaChaNotes schema, migration verification uses
in-memory or isolated scratch databases only. A feature branch must not launch
against the user's shared real database before compatible code is the common
runtime baseline.

## Error and State Model

Root states are explicit and non-overlapping:

- **Up to date**;
- **Checking**;
- **Paused**;
- **Offline**;
- **Pending**;
- **Needs attention**;
- **Blocked — unsafe root/capability**; or
- **Failed**.

Item failures do not block unrelated bindings unless root safety, coordinator
ownership, server capability, or recovery capacity is compromised. Root-level
failures pause mutation admission.

Authentication expiry, server profile identity change, claim fencing, or loss of
write capability transitions the root to **Paused** or **Blocked** before any
destructive local write. Resuming requires restored capability and a new dry-run;
changing direction requires an explicit settings action.

Notifications summarize outcomes; durable receipts remain at the point of action.
Errors name a safe next action without exposing note content, credentials, full
local paths in logs, or raw remote exception text.

## Scale and Responsiveness

- Directory scanning, hashing, parsing, server access, and reconciliation run off
  the Textual event loop.
- Scans and imports are cancellable between bounded batches and report progress.
- Previews are paginated and filterable.
- Folder trees load children lazily and use bulk membership reads.
- Watcher events coalesce; repeated changes supersede stale unexecuted plans.
- Server-backed roots use cursors/events and adaptive backoff, not full-detail
  polling.
- Numerical scan, batch, latency, and memory limits are set only after benchmarks
  against representative large trees; the implementation plan must record the
  measured evidence.

## Security and Privacy

- All filesystem candidates pass the centralized path-validation and containment
  boundaries.
- Symlink/reparse traversal fails closed.
- Root overlap considers canonical ancestor and descendant relationships.
- Writable candidates must remain single-link regular files with stable identity;
  hard links and aliases fail closed for bidirectional mutation.
- Sync databases and recovery content follow ADR-029 owner-only storage.
- Server profile identity namespaces every remote root, claim, and receipt.
- API keys and authentication material use existing credential resolution and are
  never stored in sync records.
- Persistent logs contain category, root/item opaque IDs, counts, and state only;
  no content, absolute paths, hashes, exception messages, or credentials.
- UI disclosure explains local recovery of overwritten server content, 30-day
  retention, capacity, and explicit local recovery clearing.
- Portable exports and generic database backups exclude active root, lease,
  journal, and recovery state; ChaChaNotes backups retain logical folder data.

## Delivery Roadmap

This design is an umbrella. Each slice requires its own atomic Backlog task,
acceptance criteria, plan, verification, and closeout.

0. **Media import clarity** — independently rename/clarify the Library entry as
   **Import media…**.
1. **Local folder foundation** — schema, repository, subtree operations, bulk
   queries, normalized contracts, lazy tree, Unfiled, manual memberships, and
   ChaChaNotes backup/restore coverage.
2. **One-time import** — planner, preview, receipt ledger, bounded executor,
   recursive folder structure, repeats, and partial-failure retry.
3. **Local sync substrate** — multi-root registry, binding uniqueness, private
   store, backup exclusion, coordinator lease, serialization/metadata admission,
   journal, recovery admission, dry-run, and paused legacy migration.
4. **Local continuous sync** — watcher hints, reconciliation, conflicts,
   deletion review, guarded/composite moves, root retargeting, root management,
   and shutdown recovery.
5. **Server contract** — separate `tldw_server` ADR and APIs for bulk folders,
   ownership, claims, cursors, and idempotency.
6. **Chatbook server parity** — capability adapter, remote folder tree, import,
   persistent sync, per-mutation claim fencing, capability-loss pause, claim
   conflict, and takeover UX.
7. **Closeout** — accessibility, large-tree performance, crash recovery,
   multi-process behavior, live local/server verification, docs, and lessons.

## Verification Strategy

### Storage and migration

- Migrate from every supported prior ChaChaNotes schema in isolated databases.
- Re-run migrations to prove idempotence.
- Cover multiple legacy roots, missing roots, out-of-root paths, duplicates, and
  invalid metadata.
- Prove no candidate root starts or mutates during migration.
- Mutation-test schema/version guards and folder uniqueness constraints.

### Folder model and tree

- Unit-test create/reuse, subtree rename/move, collision, soft delete/restore,
  manual memberships, source-managed memberships, and owner-specific conversion.
- Property-test separator, case, Unicode, dot-segment, and ancestor normalization.
- Test explicit ancestor+descendant manual placement versus generated-ancestor
  collapsing.
- Test placement-aware row identity, breadcrumbs, stable expansion/focus, lazy
  paging, and bulk loading without N+1 requests.

### Import

- Individual files, recursive directory, selected-root inclusion, structured
  multi-note placement, empty/unsupported branches, and folder-label collision.
- New, unchanged, changed, uncertain, unsupported, and failed classifications.
- Per-item overrides, optimistic update conflict, cancellation between batches,
  itemized receipt, and failure-only retry.
- Prove content update and imported-folder placement are independently previewed
  and independently executable for an existing note.
- Prove source paths and provenance never enter server payloads or ordinary logs.

### Sync and recovery

- Pure reconciliation transition tests for every root/item state.
- Property/state-machine tests proving repeated reconciliation is idempotent.
- Temporary-root integration tests for create/update/move/delete, external races,
  root loss, symlinks, hard links, unsafe destinations, and mass-deletion grouping.
- Round-trip BOM, LF/CRLF, final-newline, mixed-newline, invalid-encoding, mode,
  and supported platform-metadata fixtures without silent normalization.
- Constraint tests proving one note cannot bind to multiple roots and one
  normalized path/file identity cannot bind to multiple notes.
- Root-retarget tests proving the root pauses, performs a dry-run, preserves the
  old authority until activation, and infers no deletion from the new directory.
- Direction tests proving out-of-direction edits pause before overwrite, explicit
  one-time conflict choices do not change configuration, and deletion always
  requires confirmation.
- Composite subtree-move crash injection proving deterministic child replay and
  honest partial-progress/attention reporting.
- Crash injection after every journal stage, proving the next startup exposes a
  valid Resume, Restore, or Disconnect path.
- Capacity admission and retention tests proving unresolved/Undo data is not
  evicted.
- Cross-process lease tests proving one active mutator and passive status readers.
- Shutdown tests proving journal safety precedes generic worker/database teardown.

### Server contract

- Assert against real client/server signatures and verbatim captured payloads.
- Real server integration for bulk memberships, subtree operations, idempotency,
  incremental cursors, claim conflict, release, and explicit takeover.
- Prove payloads never contain local paths, hashes, watcher data, or recovery
  content.
- Prove advertised-capability violations pause the root without fallback.
- Prove every server mutation carries a current claim token/version and that
  takeover, authentication expiry, or permission loss fences the old writer
  before any further local destructive mutation.
- Contract-test that Sync-v2 M1 `notes.note` envelopes remain folder-free and
  direct server-backed folders use the distinct Notes folder capability.

### Backup and portability

- Verify ChaChaNotes backup/restore preserves logical folders and memberships.
- Verify portable export and generic backup manifests exclude roots, physical
  paths, leases, journals, watcher state, bindings, and recovery content.
- Verify restored managed memberships with no device-local owner remain inactive
  and visible until review converts them to manual or removes their organization.
- Verify any future device-local recovery import restores all roots paused and
  cannot activate without path, claim, and dry-run validation.

### Textual and live product verification

- Pilot tests for mode choice, explanatory copy, preview, folder tree, root
  controls, conflict/delete resolver, receipts, keyboard navigation, focus
  restoration, compact terminals, and color-independent labels.
- Benchmark representative large trees before fixing numerical limits.
- Live TUI checks at the user-facing surface with disposable config, data,
  recovery, local folders, and server state.
- Verify the complete flow: source selection → preview → commit → tree placement →
  external edit → automatic detection → conflict/deletion review → recovery.

## Documentation and Governance

ADR required: yes

ADR paths:

- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: this design changes the local Notes schema, folder and membership
ownership, sync/conflict/deletion policy, private recovery storage, cross-process
runtime authority, server service contract, privacy boundary, and long-lived
Library structure. ADR-073 amends the accepted ADR-059 with the reviewed
round-trip representation, binding uniqueness, composite-operation, Sync-v2,
backup/restore, and per-mutation claim-fencing constraints.

The `tldw_server` portion requires its own separately allocated ADR in that
repository before server implementation begins. Every implementation plan must
link the applicable ADR(s) and repeat the ADR check required by `AGENTS.md`.

## Approved Alternatives Review

| Option | Decision |
| --- | --- |
| Server-compatible hierarchical folder entities plus ownership-aware memberships | Selected; supports many-to-many organization, local/server parity, and managed ownership. |
| Derive virtual sync folders only from file paths | Rejected; creates incompatible manual, local, server, and virtual folder behaviors. |
| Add one `folder_path` column to each note | Rejected; cannot support many-to-many membership, independent folder identity, or clean subtree operations. |
| Automatically propagate deletions | Rejected; unsafe when roots disappear, remount empty, or external tools remove large trees. |
| Newest-version automatic conflict resolution | Rejected as the default; timestamps do not prove user intent. |
| Store physical sync details on the server | Rejected; leaks device-private paths/state and confuses cross-device ownership. |
| Keep the legacy single-root sync engine active in parallel | Rejected; two active ownership systems would diverge and race. |
