# File-Backed Notes Authority and Recovery Design

Date: 2026-07-22
Status: Amended and internally reviewed; pending user review before implementation planning
Backlog: [TASK-399](../../../backlog/tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
ADR: [ADR-021](../../../backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md)

## Summary

Chatbook will make existing Git-managed Markdown and text folders first-class
Notes sources inside Library. The bytes on disk remain authoritative. Editing a
file-backed note in Chatbook changes the actual file, so the user's existing
`git status` / `git add` / `git commit` / `git push` workflow continues without
translation, export, or a second synchronization step.

SQLite has two supporting roles, neither of which competes with the files or
changes the existing Database-note schema:

1. A dedicated `file_notes.db` holds roots, local identities, a searchable
   derived projection, and file FTS. It is isolated from the existing
   ChaChaNotes database and its Database-note tables, migrations, constructors,
   backup/restore controls, and failure domain.
2. A separate `notes_recovery.db` holds the mutation journal, mandatory
   operation-safety copies, and opt-in recovery replicas/history for selected
   files and folders.

The recovery database is independent of both the Database-note store and
`file_notes.db`. It contains enough information to list, verify, and
exact-export retained notes if either is unavailable. Both file-note databases
are unencrypted same-device storage. Neither is an off-device or disaster
backup.

The feature lives in the existing Library Notes surface. Wide terminals show a
collapsible folder navigator beside the editor. Narrow terminals switch between
Navigator and Editor views instead of squeezing both. Search replaces the tree
until cleared. Existing Database notes retain their current behavior in a
separate group.

Git controls are deliberately excluded from this tranche. A later, separately
approved feature may add staging, commit, and push actions after file authority
and recovery have proven reliable.

## Problem

The current Notes database and legacy bidirectional folder sync do not provide a
safe primary interface for a large, independently Git-managed note tree:

- the database and files behave like competing peers;
- last-write-wins and incomplete baselines can silently choose the wrong side;
- external moves can duplicate identities;
- deletion and reappearance semantics are unclear;
- the current scan/list paths are not a 1,000–5,000 file tree browser;
- watcher events are incomplete and can trigger expensive full rescans;
- file writes do not provide the compare-and-swap, crash recovery, and byte
  preservation required for a Git working tree;
- the existing Database-note store is not an independent recovery copy of
  selected source files.

The desired workflow is simpler: work in Chatbook, inspect ordinary Git changes,
and commit them with normal Git tooling. Chatbook should add navigation, search,
editing, Console handoff, and local recovery without becoming a second source of
truth.

## Goals

- Treat existing supported files as the canonical note content.
- Make a 1,000-file tree comfortable and a cached 5,000-file tree a release
  benchmark.
- Write changes directly and safely to the Git working tree.
- Reconcile external editor and Git changes near real time.
- Preserve exact frontmatter bytes while editing only the body.
- Keep stable local UUIDs for file selection, local metadata, Console handoff,
  recovery, and continuity across known moves.
- Maintain a self-contained current recovery replica for user-selected files and
  folder prefixes.
- Preserve exact bytes before every destructive Chatbook action and before every
  confirmed file deletion.
- Keep existing Database notes and Library/Console workflows working when no
  file root is configured.
- Fail closed when authority, recovery, path, or concurrency checks are not
  satisfied.

## Non-Goals

- Replacing Git or automatically committing/pushing changes.
- Making SQLite an equal content authority.
- Synchronizing file-backed notes through remote Sync v2.
- Portable note identity across a fresh Chatbook database or clone.
- Injecting UUIDs, metadata, or manifests into the note repository.
- Parsing, normalizing, or editing YAML frontmatter.
- Automatic three-way merges or timestamp-based conflict winners.
- Automatically converting Database notes into file-backed notes, or the
  reverse.
- Following symlinks, mutating hardlinks, or guaranteeing network/cloud-mounted
  filesystem behavior.
- Encryption, secure erasure, ransomware protection, or protection from loss of
  the device holding both the files and recovery database.
- Interpreting `.gitignore` as a Notes inclusion policy.

## Terms

| Term | Meaning |
| --- | --- |
| **Database note** | Existing content in the `notes` table; SQLite owns its title and body. |
| **File-backed note** | A supported regular file under a linked root; disk owns its path, title, and bytes. It is represented only in dedicated `file_notes.db` tables. |
| **Projection** | The dedicated `file_notes.db` title/body/search representation derived from a file. It is not a Database-note row or recovery copy. |
| **Binding** | Durable association between a local note UUID and a root-relative path. |
| **Protected note** | A file selected by a folder-prefix rule or file override for a continuously maintained recovery replica. |
| **Current replica** | The one verified, exact current-byte copy retained for a protected note. |
| **Checkpoint** | A sealed, coalesced recovery revision retained by editing session rather than by autosave. |
| **Safety snapshot** | A temporary exact-byte copy required to make one Chatbook mutation recoverable. |
| **Deletion snapshot** | A verified exact-byte copy pinned for the confirmed-deletion guarantee. |
| **Displaced target** | The exact file object atomically moved aside by a Chatbook replace or delete so its last-moment bytes can be inspected and retained before removal. |
| **Attention** | A durable unresolved operation holding a draft and the observed disk state for user resolution. |
| **App session** | One Chatbook process lifetime, identified by a new random UUID at startup. It is the boundary for `Changed this session`. |
| **Editing session** | One continuous open/edit interval for a note. For protected notes, autosaves within it update the current replica but do not each create a checkpoint. |

## Authority and Identity

### Disk owns content

For a file-backed note:

- the file's exact bytes are canonical;
- the exact root-relative path is its canonical locator;
- the exact basename, including extension, is its displayed and projected title;
- the dedicated file projection stores the decoded editable body only;
- frontmatter is not included in the projection, FTS, RAG, or Console body;
- filesystem mtime is observation metadata, never a conflict winner or logical
  version.

`file_notes.db` may temporarily lag disk, but Chatbook may never overwrite disk
merely because its projection is newer by timestamp.

### SQLite owns local identity and metadata

Every file-backed note receives a random local UUID in SQLite. The UUID is used
for selection, file-local metadata, Console references, operation recovery, and
optional later integrations. It is not written into frontmatter, a sidecar, or
a repository manifest.

The UUID is preserved automatically only when identity evidence is explicit:

- a Chatbook move or rename completed by the mutation coordinator; or
- a paired move event delivered by the watcher and verified by reconciliation.

A missed or ambiguous external move is represented as one missing binding and
one newly discovered file with a new UUID. Chatbook does not guess identity from
inode reuse, content hash, or filename similarity. A fresh Chatbook database or
clone also assigns new UUIDs.

### Versioned hashes, two purposes

- Version 1 uses SHA-256 and persists the algorithm/version with every durable
  hash. `raw_hash` is `sha256:<hex>` over the complete file bytes. It protects
  mutations, exact export, delete, restore, and recovery verification.
- `semantic_hash` uses a separately domain-tagged, length-delimited SHA-256 input
  containing the decoded editable body plus the filename-derived title. It
  decides whether FTS and any later RAG work is necessary.

Changing the algorithm or semantic input format requires a schema/version
migration. Hash strings without a recognized version never authorize a
mutation.

A frontmatter-only or line-ending-only change can increment the local projection
generation while leaving the semantic hash unchanged and avoiding an FTS
rebuild.

## Supported File Contract

### Delivery and writable-platform contract

The read-only file source, projection, tree, search, monitoring, preview, and
export workflow is supported in packaged macOS, Linux, and Windows builds for
ordinary local files that pass the read-safety checks below.

The first writable release is deliberately narrower:

- macOS on a local APFS volume is the only B1 writable target;
- activation probes the actual root volume and required primitives rather than
  trusting the operating-system label;
- the platform adapter uses native `renameatx_np` exchange/exclusive operations,
  pinned directory descriptors, no-follow opens/traversal, file and directory
  flushes, and macOS full-fsync support where required;
- Linux and Windows roots remain first-class read-only sources until a
  separately approved native adapter passes the same displaced-target,
  beneath-root, permission, crash, and durability suite;
- network, cloud-synchronized, and otherwise unverified mounts remain
  read-only on every platform.

This scope keeps Chatbook itself cross-platform without pretending that Python's
portable path APIs provide equivalent write guarantees. The capability probe
reports the exact failed primitive and never silently falls back to a weaker
write path. Broader writable-platform support is an expansion, not a gate for
the first useful read-only release or the macOS writable release.

### B0 writable-substrate proof

No B1 write implementation or control may begin until B0 checks in
`backlog/docs/file-backed-notes-apfs-capability-matrix.md` with an executable
packaged-app probe and explicit go/no-go result for every supported macOS
release. The retained artifact records runner hardware, macOS/APFS versions,
local-volume detection, exchange/no-replace behavior, pinned no-follow
traversal, file/directory durability and full-fsync behavior, metadata
detection/post-exchange classification, failure output, the named
power-cut/reboot method and observed durability result, and the exact build
tested.

B0 exposes no write action. A failed or missing matrix keeps all roots
read-only. The runtime probe still revalidates the actual root and primitives;
the checked-in matrix proves feasibility and packaging, not permanent
capability.

The initial fixed, case-insensitive extension allowlist is:

- `.md`
- `.markdown`
- `.txt`
- `.text`

Other extensions are counted as ignored during activation preview but are not
bound as notes. Per-root extension configuration is deferred until there is a
demonstrated need.

Supported content is a regular file that:

- is strictly decodable as UTF-8, with or without a UTF-8 BOM;
- is no larger than the existing Library pre-read guard of 8,000,000 raw bytes;
- has an editable body no larger than the existing Library editor limit of
  2,000,000 Unicode characters;
- is not beneath `.git/`;
- is not a Chatbook same-directory write temporary;
- is not a symlink;
- has one filesystem link.

An allowed-extension file that is oversized, unreadable, encoded differently,
has malformed frontmatter, or has multiple hardlinks remains path-visible with a
specific read-only reason. Symlinks and special files are reported in activation
and diagnostics but are never followed or mutated.

A nested mount beneath a linked root may remain path-visible and readable, but
it is a separate write boundary. B1 mutation requires every traversed component,
source, destination, and parent directory to remain on the pinned verified APFS
volume. A cross-device create/move/replace is rejected rather than copied or
weakened into a non-atomic fallback.

Every bound path component must round-trip through the platform's native path
API. A POSIX name represented only through surrogate escapes is diagnostic-only
and is not bound. Names containing C0/DEL or bidirectional-control characters
are escaped visibly and remain read-only because terminal rendering could make
their confirmation ambiguous. Mutation tokens always bind the exact native path,
never the rendered label.

### Opaque frontmatter

For Markdown files only, Chatbook recognizes a possible opaque leading
frontmatter span after an optional BOM:

1. The first line is exactly `---`, excluding its original line ending.
2. The first later line exactly equal to `---` closes the span.
3. Every byte from the opening delimiter through the closing delimiter and its
   line ending is retained verbatim.

Chatbook does not parse or regenerate that span. The editor receives only the
body bytes after it. If an opening delimiter has no closing delimiter, or the
user says the leading delimiters are ordinary Markdown, the file is read-only
until the user explicitly selects `Treat entire file as plain body`. That choice
is stored in the local binding, never in the file.

For `.txt` and `.text`, the entire decoded file is the body.

### Byte-style preservation

Chatbook records and preserves:

- optional UTF-8 BOM;
- uniform LF or CRLF line endings;
- whether the file ends with a newline;
- POSIX mode bits where the platform supports them.

The Textual editor uses `\n` internally. A file with mixed line endings displays
a warning before its first Chatbook write. Lone-CR files use the same explicit
normalization path and normalize to LF. If the user proceeds, a mixed body is
normalized to the count-dominant LF/CRLF style (the first encountered wins a
tie); the exact prior bytes must already be recoverable. The acknowledgement is
bound to the observed raw hash, so a later external mixed-newline version asks
again.

New files use UTF-8 without a BOM and LF. Their final-newline state follows the
submitted editor body; an empty new file has no final newline. Initial
permissions honor the platform's owner-writable creation policy and process
umask and may never be broader than the containing directory policy.

Raw-hash equality is only a content no-op. Reconciliation still refreshes file
identity, type, link count, mode, and security facts. A later save uses the
freshly observed supported mode/security facts and must prove that replacement
will not broaden access.

Writable admission is fail-closed for filesystem metadata. If the platform
adapter cannot detect and round-trip an ACL, extended attribute (including
Finder tags/resource metadata), ownership fact, file flag, Windows alternate
stream, or other access-relevant metadata that is present, the affected file is
read-only with the exact reason. A later adapter may preserve an explicitly
tested metadata set; no successful content save may silently discard unsupported
metadata. This gate applies to save, rename, move, delete, and restore.
Preflight binds a supported metadata fingerprint; post-publication inspection
must detect metadata introduced during the race window and retain/roll back
rather than discard the displaced object. Network-filesystem atomicity remains
outside the writable contract.

## Data Model

File-backed state lives in two owner-only SQLite databases outside the existing
ChaChaNotes database. Existing `notes`, `notes_fts`, keyword, relation, and sync
tables, their triggers, schema version, migrations, constructors, backup/restore
controls, and ordinary connection lifecycle are not modified for file
ownership. Exact DDL is assigned in the implementation plan, but the following
state is required.

### Storage-instance identity

The canonical configured main-database path identifies the active Chatbook
storage profile and is hashed with a fixed domain tag to form a stable,
non-secret storage-instance ID. The owner-only
`<user-data>/file_notes/<storage-instance-id>/` directory contains that
instance's `file_notes.db`, `notes_recovery.db`, diagnostics, and recovery
bootstrap markers. Absolute main-database paths are not logged or used as
filenames.

The core rollout activates only one storage instance/root at a time under the
per-user File Notes coordinator election. Read-only A acquires no mutation
lease; exclusive root-mutation ownership begins only with the B1 read/write
upgrade. The namespace prevents a custom main DB from silently sharing another
DB's recovery history without introducing multi-profile coordination machinery.

Path-derived placement is not recovery identity. At first B1 bootstrap Chatbook
generates a random recovery-instance UUID and persists it in both the
`file_notes.db` storage row and a self-contained recovery-store identity row. A
recoverable bootstrap marker handles a crash between those commits. After a
pairing exists, an absent or mismatched recovery store fails closed: Chatbook
never silently creates an empty replacement over evidence of prior B1/B2/B3
state.

Changing the configured user-data directory or configured main-database path
selects a different, independent File Notes storage profile. The first rollout
does not auto-discover or diagnose stores in another namespace: a new namespace
may start fresh only when it contains no database, sidecar, marker, or other
recovery evidence, and returning to the prior configuration reopens the prior
profile. Within the selected namespace, a derived/stored storage-instance
mismatch or recovery UUID mismatch fails closed. Explicit quiescent
recovery-store relocation/clone is a later administrative capability, not an
automatic path heuristic.

### Existing ChaChaNotes database — unchanged

The `notes` table remains Database-note-only. Existing list, count, search,
create, update, delete, export, RAG, MCP, keyword, relation, and Sync behavior
continues without a new discriminator column or file-row branch. This is the
primary zero-root and non-degradation boundary.

No file projection UUID is a `notes.id`, and no generic Database-note method can
address a file projection. Any future shared keyword/link experience requires
file-specific association tables and an explicit combined read model; it is not
part of the core rollout.

### Dedicated `file_notes.db`

#### `file_notes_storage`

One file-feature storage row records the derived storage-instance ID, nullable
expected recovery-instance UUID, feature/schema state, and pairing/bootstrap
generation. It contains no note content. Only the explicit
`RecoveryStoreBootstrap` may establish first pairing; changing an established
pair is deferred to the quiescent relocation/clone design.

Pairing is between `file_notes.db` and `notes_recovery.db`; neither may silently
initialize, adopt, or overwrite the other:

| `file_notes.db` state | Recovery database or prior-pair marker | Required result |
| --- | --- | --- |
| Absent | Absent | Fresh A may exclusively create `file_notes.db` |
| Healthy, unpaired | Absent | A works; B1 may exclusively create and pair recovery |
| Healthy, unpaired storage S | Exact valid `bootstrap_in_progress` marker for storage S, UUID X/generation G, plus absent or healthy recovery identity S/X/G | Resume that pairing idempotently |
| Healthy, unpaired | Any nonmatching recovery/marker evidence | Orphaned recovery; never adopt/overwrite; recovery-only access |
| Healthy, paired storage S/UUID X/generation G | Healthy identity S/X/G | Normal paired operation |
| Healthy, paired S/X/G | Healthy S/X/G plus the exact valid in-progress marker | Verify both identities, remove the marker durably, then operate normally |
| Healthy, paired S/X/G | Absent, invalid, or any identity other than S/X/G | Browse/reconcile read-only; mutation/protection blocked |
| Absent, corrupt, or incompatible | Any recovery/marker evidence | Preserve both; no replacement projection DB; recovery-only access |

First B1 pairing requires exclusive no-follow creation and complete prior
absence of `notes_recovery.db`, its WAL/SHM sidecars, and the bootstrap marker.
Chatbook then binds storage S, generates UUID X and generation G, exclusively
persists an owner-only `bootstrap_in_progress` marker containing S/X/G, and
flushes its directory. It commits `recovery_store_identity` S/X/G first, commits
the expected S/X/G in `file_notes_storage` second, verifies both, then removes
the marker and flushes the directory. Startup may resume only this exact
marker/identity sequence and repeats each remaining step idempotently; any
nonmatching evidence is orphaned and preserved. A null projection-side UUID is
never by itself permission to initialize when recovery evidence exists.

#### `note_file_roots`

Each root stores:

- root UUID and user-facing label;
- local canonical path and a platform filesystem identity/fingerprint;
- runtime mode and lifecycle state;
- exact-path comparison policy and read/write safety capabilities;
- protection-prefix configuration;
- activation, scan, observation, metadata-verification, and raw-verification
  generations/deadlines;
- timestamps and diagnostic state.

The core schema is naturally keyed by root UUID, but the release permits one
active root. An overlap with the configured legacy Database-note sync root is
rejected.

#### `file_note_projections`

One row owns both the local binding and derived projection for a file:

- note UUID and root UUID;
- exact relative display path plus a filesystem-aware comparison key;
- filename-derived title and decoded editable body;
- raw and semantic hashes;
- presence state;
- observed size, mtime, mode bits, BOM/newline/final-newline facts;
- observed file identity, link count, and supported security facts;
- frontmatter mode and format/read-only diagnostic;
- local projection generation and last observation time;
- body-index state, indexed semantic hash/generation, and path-metadata
  generation.

The filesystem-aware comparison key is derived per root. If Chatbook cannot
determine a safe comparison policy or detects normalized collisions, that root
cannot enter `read_write`. The exact display path is never replaced by the
comparison key.

Projection presence states are:

| State | Meaning and projection behavior |
| --- | --- |
| `present` | Path is observed. A readable supported body is eligible for normal projection/search; a body-ineligible diagnostic remains path/title-visible through the navigator only. |
| `missing` | This path is absent while the root is online. Hide it from Browse and search results, but retain the last projected body for diagnosis only; it is not a recovery copy. |
| `tombstoned` | Chatbook confirmed deletion. Clear the projected body after the deletion snapshot is committed and unlink is verified. |

Path-only navigator results come from `file_note_projections`, not FTS.
Reappearance at a `missing` path retains the UUID. Reuse of a `tombstoned` path
creates a new UUID unless the user explicitly restores/reuses the tombstone.

Offline is a root lifecycle/effective UI state, not a mass binding transition.
When a root is Offline, its projections keep their last `present`, `missing`,
or `tombstoned` state. Browse may show cached rows under an Offline banner, but
no result claims freshness.

Projection is idempotent on root, filesystem-aware path key, raw hash, and
presence state. Its generation is monotonic and local, never mtime-derived.
Observation time and filesystem mtime remain distinct. Each root also persists
initial-body-index progress so indexing can resume after restart without making
path search wait.

If a previously readable file becomes unreadable, oversized, malformed, or
otherwise body-ineligible, reconciliation immediately suppresses its old body
from file FTS and optional later RAG results while retaining the cached row for
diagnosis only. The navigator continues to expose a path/title-only row with the
exact read-only reason. It may not show a stale body snippet as current or use
the projection to satisfy a recovery guarantee.

#### `file_notes_fts`

A dedicated external-content FTS5 table indexes the filename-derived title and
editable body of present, body-eligible `file_note_projections`. It has no
projection triggers. A content projection marks indexing pending unless its
semantic hash is already indexed. An idempotent file-body indexing worker finds
pending/mismatched rows without a separate queue table, then updates FTS and the
projection row's indexed semantic hash/generation in one `file_notes.db`
transaction. FTS lag or failure never rolls back a current projection, keeps a
recovery operation pending, or blocks a later file mutation. Search joins only
rows whose index generation/hash still matches the current present projection;
path-only results remain available while body indexing catches up. Existing
`notes_fts` and its triggers remain unchanged.

Path search comes from indexed comparison/display-path columns on
`file_note_projections` and is combined with file FTS by the file navigator
repository. Intentional cross-source search unions file results with existing
Database-note results above the two repositories; it never merges their write
paths.

### Independent `notes_recovery.db`

The recovery database has its own schema version, integrity checks, and
owner-only files. It does not depend on joining either `file_notes.db` or the
Database-note store to enumerate or verify retained content. Its
`recovery_store_identity` row records the bound storage-instance ID, random
recovery-instance UUID, and bootstrap generation expected by
`file_notes_storage`.

#### `note_file_revisions`

Every retained row is self-contained enough for recovery and includes:

- note UUID;
- root label and identity fingerprint;
- exact relative path;
- revision kind;
- versioned raw-hash algorithm/value and raw byte length;
- codec/version, compressed length, and compressed bytes;
- recorded BOM/newline facts;
- a versioned supported-security/metadata manifest and fingerprint sufficient
  to reapply and verify every supported mode/security fact during restore;
- creation, expiry, verification, and pin state.

A revision whose source carried unsupported or unround-trippable metadata is
ineligible for mutation, deletion, or restore. Chatbook may still enumerate and
exact-export its bytes, but it never claims that a content-only payload can
round-trip the file's access-relevant metadata.

Revision kinds cover:

- `current_replica`;
- `checkpoint`;
- `safety_before`;
- `intended_draft`;
- `deletion`;
- `conflict_base`;
- `conflict_disk`;
- `conflict_draft`;
- `recovered_draft`.

Payloads are compressed inline. There is no content-addressed blob store,
reference counting, deduplication layer, or separate activity table.

The recovery schema enforces at most one live `current_replica` per note UUID,
valid operation-to-revision references with deletion restricted while referenced,
and atomic live-set garbage collection. An operation, Attention item, guaranteed
deletion, or retained draft can never be left pointing at a removed revision.

#### `note_file_operations`

The operation journal stores:

- operation UUID and app-session UUID;
- note/root identity;
- action kind;
- exact source/destination paths;
- base, expected, intended, and observed hashes;
- expected, intended, and observed supported/unsupported metadata fingerprints;
- a bounded, versioned canonical expected/intended metadata manifest when an
  operation may need to reapply supported facts after a crash;
- referenced safety/draft/replacement revisions;
- editor buffer generation;
- state, filesystem outcome, projection state, outcome, and timestamps.

Its minimal state lifecycle is:

```text
pending -> complete
pending -> attention -> complete
```

`projection_state` is independently `pending` or `applied`. FTS/index state is
not part of the recovery operation state machine.

Failure or `not_applied` is an outcome, not another state taxonomy. A completed
operation transfers any longer-lived retention obligation to its revision rows
and is then eligible for deletion. `Changed this session` is an in-memory,
process-lifetime ordered map populated only after a working-tree-changing
operation completes. No-op saves, external changes, and rows recovered from a
prior process do not appear there. Only pending and Attention operations must
survive restart.

## Service Boundaries

### File mutation coordinator

One process-wide coordinator is the only component allowed to mutate a linked
file or project a file-note row internally. It provides explicit commands such
as:

- save body with expected path/raw hash and editor generation;
- rename/move with expected source and destination absence/hash;
- create at an explicit supported path whose parent already exists;
- confirmed delete with an exact confirmation token;
- restore from a verified recovery revision;
- reconcile one path or one root.

It serializes commands per note and runs at most one reconciliation flight per
root. A writable root is held through a platform handle pinned to its verified
filesystem identity; path traversal and mutation are descriptor/handle-relative
and reject symlink or reparse-point substitution in every component. Filesystem
work, hashing, compression, SQLite work, and scans run outside the Textual UI
thread.

Existing `path_validation.py` helpers remain picker/discovery validation only.
They never authorize a file mutation. The coordinator's narrow
`PinnedRootFilesystem` platform module owns every beneath-root open, stat,
publish, move, and delete operation.

### Projection and combined reads

`FileNotesRepository` owns `note_file_roots`, `file_note_projections`, and
`file_notes_fts`. It exposes paged tree, path search, FTS search, detail, and
internal idempotent projection commands. It has no public filesystem mutation
method.

A thin `DatabaseNotesNavigatorReadAdapter` supplies deterministic,
metadata-only list/search pages for the combined navigator. It does not reuse
the current capped `search_notes` seam, whose offset is not forwarded end to
end, and it exposes no write method.

`LibraryNotesReadRepository` intentionally requests Database-note reads from
that navigator adapter and file reads from `FileNotesRepository` for the
Library navigator, explicit Console handoff, and other approved read-only
consumers. Every result carries a source kind. Its response contains independent
File and Database payloads, cursors, and source-scoped errors rather than one
all-or-nothing result. A file projection/index failure cannot blank or disable a
healthy Database list/editor/search, and a Database read failure cannot blank
cached/current file navigation. Partial results are visibly labeled with a
source-specific retry. This is a composition boundary, not a common write
repository.

### Existing Database Notes services

`NotesScopeService`, `NotesInteropService`, the `notes` table, and their public
CRUD remain Database-note-only. File reads reach them only through an explicit
combined read consumer, and file writes never do. This makes legacy sync, Sync
v2, MCP create/update/delete, generic export, and old direct database callers
incapable of mutating a file projection by construction.

### Library workbench boundary

A separately mounted `FileNotesWorkbench` renders the file-source experience.
A `FileNotesSessionController` owns only file-note state: the open file buffer
and generation, cursor and selection, focus ID, scroll offsets, tree expansion,
selected file UUID, navigator mode, search query/generation/results, save state,
and Attention state. It never owns or reimplements a Database-note buffer or
command.

The Notes host routes every combined-navigator result by its explicit source
kind. A File result opens the file workbench. A Database result switches the
whole Notes content surface to the existing `LibraryNotesCanvas` and current
Database-note handlers, including create, edit/autosave/conflict,
keywords/links, template/import/export, and legacy sync. It does not attempt to
embed that whole canvas beside the file navigator.

`LibraryScreen` retains its existing Database editor state and handlers and owns
host routing. An exact `host_return_target` contains the prior File navigator
mode, query/generation, composite source/UUID selection, visible-row anchor,
tree-expansion generation, focused-control ID, and narrow view. A distinct
tagged `database_editor_back_target` records either the delegated Database list
context or the direct combined-result File origin; one target never overwrites
the other. In an active-root session the delegated Database surface shows the
host-level label `Database note • stored in Chatbook`; the file surface shows
`File • stored on disk`.

Opening `Open database notes` establishes the host target. A note subsequently
opened from that delegated list keeps the existing editor `Back to list`
destination while persistent host chrome remains `Back to linked notes`. A
Database result opened directly from the combined navigator instead gives the
editor the tagged File origin, so editor Back returns to that exact row.
Without a host target, direct Database deep links and Back keep existing
Database behavior. Before either source switch, the active source's leave guard
must complete or veto the switch: File uses its hash-checked save/retention
guard and Database uses its existing autosave/conflict guard.
`FileNotesSessionController` remains alive while the Database canvas is shown.
The host target clears only after the File workbench and visible focus restore
successfully; Database editor Back state clears through its existing list/editor
lifecycle. `LibraryScreen` does not absorb the File controller or coordinator.
With zero roots it mounts the existing Database canvas directly, apart from the
specified Link and genuine retained-Recovery actions, and adds no authority
banner.

The workbench keeps the logical editor mounted through Navigator/Editor layout
changes where Textual permits it and uses targeted widget updates rather than
whole-screen recomposition. A dedicated navigator widget lazily mounts bounded
child batches; it never renders a 5,000-row directory as 5,000 eager buttons.

## Mutation Protocol

SQLite and the filesystem cannot form one transaction. Every Chatbook mutation
therefore uses a durable intent journal plus hashes.

The file coordinator owns dedicated critical-write SQLite connections.
`notes_recovery.db` and the `file_notes.db` transactions that form file-operation
barriers use a release-tested durability profile with full synchronous commit
and platform full-fsync support where available. The exact supported
journal/synchronous combination is pinned per platform and verified at startup.
Ordinary Database-note connections need not adopt this cost. A failed
commit/fsync or an environment that cannot establish the required ordering keeps
file roots read-only.

### Common mutation invariant

Before any filesystem mutation, the coordinator must:

1. Hold the exclusive root-mutation lease and pinned root handle. Acquire the
   note lock plus normalized source/destination path locks in deterministic
   order.
2. Traverse from the root handle without following links and revalidate root
   identity, mode, containment, parent identity, file type/link count/security
   facts, expected source hash/state, and destination hash/absence.
3. Commit one durable `pending` operation and references to the intended state
   plus any bytes whose only accessible copy could be displaced.
4. Re-read every newly persisted safety/replacement revision after commit,
   bounded-decompress it, and verify codec, length, raw hash, bounded canonical
   metadata-manifest encoding/version, and manifest fingerprint before touching
   the working tree. The manifest must reproduce the freshly captured supported
   source facts. A reused current replica qualifies only when its verified
   hash/length and metadata fingerprint match the just-read source.
5. Use collision-safe no-replace semantics for create, move, and
   restore-to-empty. A replacement save/overwrite must atomically publish
   the intended file while preserving the displaced target as an
   operation-owned file via exchange or replace-with-backup. Confirmed delete
   must first atomically rename the target to an operation-owned same-directory
   quarantine. If an equivalent primitive is unavailable, the corresponding
   command and root stay read-only.
6. Flush the written file before publication and durably flush every affected
   directory after publication: destination for create/restore, source and
   destination for a cross-directory move, and the one directory once for a
   same-directory action. A flush failure leaves the operation pending/Attention
   and never reports durable completion.
7. Inspect the target and displaced/quarantined file through pinned no-follow
   handles, using stable-read rules, and classify all observed hashes and
   supported/unsupported metadata fingerprints before the idempotent file
   binding/projection transition.
8. After expected displaced bytes and their supported metadata manifest are
   represented by a round-trip-verified recovery revision with the applicable
   retention, unlink the operation-owned artifact, fsync its parent, and confirm
   absence. Any unexpected metadata fingerprint on the target or displaced/
   quarantined object counts as an unexpected side even when its raw bytes are
   unchanged: retain it, roll back when provably safe, or enter Attention before
   cleanup. Unexpected objects remain named and protected until conflict
   resolution. Cleanup failure leaves the operation pending/Attention.
9. Durably commit that projection transition while the recovery operation remains
   `pending`; durably mark the recovery operation complete last.

This removes the silent compare-then-replace loss window: an external version
that wins after preflight but before publication is the displaced target, not
discarded evidence. If either its hash or supported/unsupported metadata
fingerprint differs from the expected base, Chatbook retains it and enters
Attention. An uncooperative process that continues writing through an
already-open descriptor after displacement, plus network/cloud filesystems and
multi-host writing, remains outside the guarantee; Chatbook never deletes a
displaced object until its stable observed bytes and supported metadata are
durably captured.

Same-directory operation names contain the operation UUID and are recorded in
the journal. Newly created Chatbook temporary/new files are created exclusively
through the pinned parent, with no-follow semantics and restrictive owner-only
initial permissions before the supported target mode is applied. A displaced or
quarantined original necessarily retains its verified source mode/security facts
at the instant of atomic movement; Chatbook records those facts and may narrow,
but never broaden, access while it is operation-owned. Startup removes a leftover
temporary, displaced target, or quarantine only after proving it belongs to that
exact operation, classifying and durably capturing its stable bytes, and proving
it is not a current source/destination. Chatbook never deletes operation
artifacts by wildcard. A crash-left artifact may be Git-visible until startup
reconciliation removes or surfaces it.

### Save

For a non-no-op save:

1. Capture note UUID, path, base hash, body snapshot, and editor buffer
   generation.
2. Acquire the note serialization lock and verify the pinned root capability,
   mode, lease, path components, parent, regular-file status, fresh supported
   security facts, and recovery health/capacity.
3. Read and raw-hash disk immediately before mutation. If it differs from the
   expected base, atomically persist durable Attention as described below
   instead of writing.
4. Construct intended bytes from the preserved BOM/frontmatter/byte-style facts
   plus the editor body.
5. In one recovery-database transaction, persist a `pending` operation, the
   intended draft, and exact old safety bytes (or a verified reference to an
   already sufficient current replica), then perform the common round-trip
   verification barrier.
6. Write intended bytes to a uniquely created temporary in the same directory,
   apply the freshly observed supported mode/security facts, flush it, and use
   the platform exchange/replace-with-backup primitive so the displaced target
   remains operation-owned. Flush the parent directory.
7. Stable-read and hash both paths:
   - intended target plus expected displaced base: durably capture/promote the
     intended revision as the protected current replica when applicable, record
     the filesystem outcome while keeping the operation pending, run common
     step 8 to unlink/fsync/verify absence of the captured displaced artifact,
     durably project the observed file, and mark the operation complete last;
   - publication not applied and the expected old target remains canonical:
     record `not_applied`, retain the draft, and leave the editor dirty;
   - an unexpected displaced target or unexpected current target: durably retain
     every observed side; when the target still has the intended hash and the
     displaced object remains unchanged, attempt one atomic swap-back so the
     external version is canonical again. Enter Attention as `Disk changed
     during save` whether rollback succeeds or cannot be proven safe, and state
     which side currently occupies the path.
8. Clear the editor's dirty state only when its current generation still equals
   the saved generation. Typing that occurred during I/O stays `Draft`.

If the intended raw hash already equals disk, Chatbook performs no file rewrite
and creates no Git noise. It may refresh stale projection metadata, but the
action does not appear as a session change.

The required derived `file_notes.db` binding/projection is repairable but must
precede final journal completion. Recovery payload/current-replica promotion
durably commits first; the idempotent projection transition durably commits
second; the recovery row is durably marked complete last. A crash at either
boundary leaves a pending row for startup classification without weakening the
exact-byte recovery invariant.

If the binding/projection commit fails after disk and recovery payload commit,
the operation remains pending/Attention, all further file commands fail closed,
and the UI reports `Saved to disk • projection needs attention`. Once the
projection transition commits, an FTS failure is not an operation failure: the
operation completes, the projection keeps its index-pending generation, and the
UI reports `Saved to disk • search index updating` while the idempotent indexing
worker retries. If disk has the intended bytes but the recovery completion
transaction fails, the pre-written intended revision still holds those exact
bytes, the operation remains pending/Attention, all further file commands fail
closed, and the UI reports `Saved to disk • recovery needs attention`. `Save
failed` is reserved for a result in which the intended bytes did not become
canonical on disk.

Post-publication verification detects later target changes. The displaced-object
protocol also retains a version written inside the former compare/replace
window. Power-loss durability remains limited to the explicitly release-tested
local filesystems and durability profile.

Leaving the editor, changing the selected note, switching Library mode, or
navigating away runs the same save/attention flush before the logical draft is
released. A failed disk save does not discard the buffer; after durable draft
capture it becomes a recoverable draft/Attention item and navigation may
continue. If recovery storage itself cannot durably capture the live draft, the
editor remains mounted, controlled navigation/shutdown cannot discard it, and
Chatbook offers exact draft export plus an explicit discard acknowledgement.
It never labels that state `Recoverable draft`.

### Autosave and checkpoints

File-backed notes keep the existing two-second debounce. A 30-second maximum
interval from the first dirty keystroke prevents continuous typing from
postponing all persistence indefinitely. `Save now` bypasses both timers.

Before an editable file can become dirty, recovery accounting reserves bounded
headroom for its current base, maximum supported draft, and one competing
observed disk side. If that reservation cannot be maintained, the editor becomes
read-only before accepting more text and offers draft export. External changes
cannot be blocked, but Chatbook never claims durable Attention until all required
sides commit and verify.

Autosave writes the canonical file, but it does not seal a checkpoint on every
write:

- for a protected note, all autosaves update the one current replica;
- an identical raw hash creates no revision;
- a path- or supported-mode-only change refreshes current-replica metadata
  without creating a body checkpoint;
- for a protected note, navigation away, explicit `Save now`, conflict creation,
  delete, and clean app shutdown seal at most one distinct checkpoint for the
  editing session;
- a crash may omit the final sealed checkpoint, but the current replica and
  operation journal remain exact.

Unprotected notes retain only mandatory operation safety, unresolved
draft/conflict revisions, and guaranteed deletion snapshots. They do not
accumulate ordinary checkpoints.

Autosave pauses while the user has not acknowledged a mixed-newline
normalization warning. The body newline style is derived without counting the
opaque frontmatter span.

### Rename and move

Rename/move is an explicit expected-path operation:

- source containment and raw hash are rechecked;
- destination must be inside the same writable root for the pilot;
- destination parent must already exist;
- an existing or normalized-colliding destination is never overwritten;
- the binding and UUID follow only after the filesystem result is verified;
- the atomic no-replace move is followed by durable flushes of source and
  destination directories (once when they are the same);
- the idempotent File Notes binding move commits while the recovery operation remains
  pending, and the recovery operation completes last.

An explicit per-file protection override follows the UUID across a verified
Chatbook/paired-watcher move. Folder-prefix protection remains path-based and
is reevaluated in move preflight; if the move would change protection, the
confirmation states that outcome before mutation.

Case-only rename is enabled only when the root's filesystem comparison policy
has a tested safe path; otherwise it fails read-only with guidance to rename in
an external tool and reconcile.

### Confirmed deletion

Delete removes the actual supported regular file. Its confirmation token binds
the root, exact relative path, note UUID, and current raw hash. A stale token
is rejected when a changed file is observed at the final precondition check.
The quarantine protocol preserves a version that appears after that check but
before displacement instead of silently unlinking it.

The coordinator must:

1. revalidate mode, root identity, containment, regular-file status, and hash;
2. commit one `pending` delete operation and its exact deletion snapshot in the
   same recovery transaction;
3. decompress that snapshot and verify expected length, raw hash, and the
   bounded supported-metadata manifest/fingerprint;
4. atomically rename the exact target to an operation-owned same-directory
   quarantine and durably flush that directory;
5. stable-read and hash the quarantined object:
   - if it matches the confirmed hash, verify the deletion revision again,
     unlink the quarantine, durably flush the directory, and confirm target and
     quarantine absence;
   - if it differs, retain those exact bytes as `conflict_disk` and attempt a
     no-replace restoration only while the original path remains absent; enter
     Attention regardless and never report deletion complete;
6. durably commit the File Notes binding tombstone/projection transition;
7. durably complete the recovery operation last.

The exact snapshot is guaranteed for 30 days for Chatbook-initiated deletion of
a supported regular file. This guarantee applies even when continuous
protection is off. Delete is unavailable if Chatbook cannot reserve and verify
that retention. An externally observed deletion can guarantee only the last
bytes Chatbook had already captured.

The completion toast is explicit:

```text
Deleted from disk • recoverable until <date>
```

All folder mutation is deferred. Core file create/move/restore requires an
existing parent directory. Users create, rename, move, and remove folders with
their normal filesystem tools; reconciliation reflects those changes. No
Chatbook command recursively deletes descendants.

## Reconciliation and External Changes

### Watcher events are hints

`watchdog` is a declared core dependency and the supported near-real-time event
backend in packaged builds. It may not remain an undeclared environment
accident. If the observer cannot run, Chatbook uses a visible one-second
metadata-polling fallback labeled `Monitoring via polling`. Both feed the same
thread-safe, path-keyed bounded accumulator:

- duplicate events coalesce;
- a paired move enqueues source and destination;
- overflow collapses to one root-rescan sentinel;
- `.git/` and Chatbook temporary paths are excluded;
- no event is trusted without inspecting current filesystem state and hashes.

There is no time-window suppression for self-generated events. A watcher event
whose observed raw hash already matches the committed binding is a content
no-op, but reconciliation still refreshes identity, type, link count, supported
mode/security facts, and path metadata.

### Scan triggers

Reconciliation runs on:

- activation and startup;
- app resume;
- watcher overflow;
- manual refresh;
- root identity/availability recovery;
- opening or saving a note;
- a rotating low-priority verification sweep.

Metadata narrows candidates and helps detect settlement, but only a raw hash
proves content equality. Stable reads use a pinned no-follow file handle and
compare identity, size, mtime, ctime/change token, and link/type facts before and
after. When platform metadata is too coarse to prove a stable snapshot, Chatbook
double-reads/hashes or retries as unsettled. Git checkout/merge/rebase storms are
coalesced into bounded root work. They never populate `Changed this session`.

Watcher health has bounded freshness: every bound path receives a metadata
verification at least once per 60 seconds while the root is active, and every
present file receives a low-priority raw verification at least once per 24
active hours. The generations/deadlines persist across restart. Missing either
deadline labels the root `Monitoring degraded`, stops freshness claims, and
queues a bounded root sweep; normal watcher events still target settled local
changes within the two-second p95 goal.

### Root and path outcomes

- If the root is absent or its filesystem identity changes, mark the root
  Offline and do not mass-mark files missing.
- If one path disappears while the root is online, mark it Missing.
- If a missing path reappears, retain its UUID and reconcile its bytes.
- A paired watcher move preserves identity only when the source had one unique
  present binding and the destination had no prior binding before the event.
  Rename of an unbound temporary over a bound destination is a content
  replacement of the destination identity, not a note move. Directory moves
  trigger subtree reconciliation.
- A missed/ambiguous move produces Missing plus a new binding, without guessing.

`Needs attention` offers `Reassociate as moved from…` for a user-selected missing
binding and newly discovered file. It shows both paths and hashes, rechecks both,
and preserves the old UUID/metadata only after explicit confirmation. This is a
manual identity decision, never an automatic content-hash inference.

### Editor behavior

The live draft records the binding path, base raw hash, exact loaded base bytes
until they have a durable revision/reference, editor generation, and focus
state.

When disk changes:

- A clean buffer may reload automatically only when its generation is unchanged
  and the editor is not focused.
- A focused clean buffer shows a non-destructive `Disk changed` notice.
- A dirty buffer never silently reloads. Chatbook updates the projection to disk,
  preserves the draft in recovery, and creates durable Attention.
- A verified external paired move updates the breadcrumb and expected path
  without remounting the editor. External deletion of a dirty file creates
  Attention with an explicit absent disk side; external deletion of a clean file
  makes the editor Missing/read-only.

Creating Attention is one recovery transaction: a pending operation transitions
to `attention` while pinning the exact draft, the editor base (or an already
verified equivalent revision), and the latest stable observed disk bytes.
Returning from the save/reconcile worker is not allowed until that transaction
commits. Later external changes replace the pinned `conflict_disk` side only
after the new bytes verify; the base and draft are not lost.

If that recovery transaction cannot commit, the live editor remains mounted
under `Draft not yet recoverable`; navigation cannot silently release it and the
user is offered exact export. The root becomes read-only until conflict headroom
and recovery health are restored or the user explicitly exports and discards the
draft.

When a protected file changes externally, successful reconciliation also
replaces its verified current replica. Until that capture verifies, the UI says
`Recovery copy behind` and Chatbook file commands remain read-only for that
note.

The conflict actions are:

- `Compare changes`
- `Save draft as new note`
- `Keep editing`
- `Overwrite disk with draft`
- `Discard draft and load disk`

Every action first hashes the current disk again:

- Compare is against the pinned base/draft and latest verified disk side.
- Save-as-new uses a separately journaled create with destination absence
  enforced at final publish, then completes the original Attention as
  `saved_as_new`.
- Keep-editing leaves Attention active and durably replaces its draft revision
  as the buffer changes.
- Overwrite requires confirmation, composes the draft body with the freshly
  observed opaque frontmatter/BOM/body-byte style, round-trip verifies recovery
  protection, and uses the displaced-target save protocol. If that fresh
  prefix/style or platform replacement capability is unsupported or ambiguous,
  overwrite fails closed and offers Save-as-new/export.
- Discard requires confirmation, retains the latest draft as
  `recovered_draft`, loads the current disk body, and completes Attention.

Neither action can erase the only copy of displaced content. There is no
automatic merge, timestamp winner, or navigation lock; unresolved items remain
in `Needs attention`.

Target observation for a settled local edit is projection and FTS visibility
within 2 seconds p95 on the fixed benchmark runner. Optional RAG may lag.

## Recovery Replica, Retention, and Capacity

### Protection selection

Protection defaults off. The configuration model is intentionally small:

- folder-prefix include/exclude rules relative to a root;
- per-file include/exclude overrides;
- most-specific file override, then deepest folder prefix, then default off.

Enabling protection captures and verifies the current file before claiming
coverage. Until then the selected note is `Protection pending`. A pending
protected note is not writable through Chatbook. Unprotected notes in the same
healthy root may still be writable because every mutation creates its own
mandatory safety snapshot.

For every protected note, `notes_recovery.db` retains one exact current replica.
The central invariant is testable without opening ChaChaNotes or
`file_notes.db`:

> After every completed protected Chatbook save, the recovery database contains
> the exact current file bytes and can verify/list them without opening
> ChaChaNotes or `file_notes.db`.

If an external edit occurs while recovery is unavailable or behind, Chatbook
cannot block that external write. It marks the note `Recovery copy behind`,
makes Chatbook file commands read-only, and attempts reconciliation without
claiming protection.

Changing protection previews affected file count, current logical bytes, and
estimated capture/pruning impact. Disabling protection requires confirmation,
stops future current-replica maintenance only after the operation barrier, and
seals the latest verified current replica as an ordinary checkpoint with the
normal 30-day/50-checkpoint limits. It is not silently deleted to free capacity.

### Retention

- Current replica: retained while protection is enabled.
- Sealed checkpoints: retain at most 50 per note and none older than 30 days;
  both limits apply.
- Confirmed deletion bytes: pinned for at least 30 days.
- Tombstone metadata: retained beyond payload expiry until root-forget policy
  permits removal or a future explicit purge capability is approved.
- Unresolved drafts/conflicts and incomplete operations: pinned until resolved
  or explicitly discarded.
- After resolution, `conflict_disk`, `recovered_draft`, and bytes displaced by a
  confirmed conflict/restore overwrite remain recoverable for 30 days. The core
  release offers no early guarantee waiver. Their expiry is shown before
  resolution completes.
- Temporary safety revisions: eligible for pruning after the operation completes
  and all stronger retention needs are satisfied.
- Completed, non-Attention unprotected autosaves coalesce their payloads to at
  most one latest verified disk/base revision for each still-dirty open note.
  Each later completed save supersedes/prunes the prior one. That final base
  becomes immediately garbage-collection eligible when the editor becomes
  clean/closes unless an incomplete operation or Attention references it. A
  protected equivalent is eligible only after its current replica/checkpoint
  satisfies the stronger retention rule.

Completed operations are removed after their retained revisions own every
remaining retention obligation and the in-memory session-change map records the
result. Startup garbage collection removes crash-left completed rows after the
same proof. Pending and Attention rows are never removed by ordinary garbage
collection.

### Fixed capacity policy

The first release uses a fixed 1 GiB cap for live retained recovery data
(compressed content plus encoded metadata manifests), a fixed 64 KiB maximum
encoded metadata manifest per retained revision or live operation, and at least
256 MiB of filesystem free space after the next reserved operation. A larger or
unsupported manifest makes the source read-only. These are release constants,
not user-facing settings. A configurable quota is deferred until real usage
demonstrates a need. The UI shows logical retained content and manifest bytes
separately from physical `notes_recovery.db`, WAL, and SHM sizes, because
pruning does not promise immediate file shrinkage.

Before a Chatbook mutation or editable-draft admission, the coordinator
compresses or conservatively reserves the required content, encoded metadata
manifests in revisions or live operations, conflict headroom, and a free-space
margin. Current replicas, guaranteed deletion/conflict retention, and unresolved
operations are never silently evicted to make room.
The protection preview refuses a selection that cannot fit. Reaching the cap or
free-space floor later enters `Recovery capacity reached` and makes file-backed
commands read-only until automatic or explicit eligible-history pruning
succeeds. Stopping protection only makes its sealed checkpoint eligible for a
separate pruning decision; it does not itself free capacity. Guaranteed
deletion/conflict retention and unresolved drafts are never part of ordinary
pruning.

Every decompression is bounded by recorded codec/version, compressed length,
expected raw length, and maximum supported file size, then verified by raw hash.
Every metadata manifest is bounded by the 64 KiB encoded limit, parsed only at a
supported version, canonicalized, and verified by fingerprint. Recovery-store
corruption or failed validation preserves the suspect database for diagnosis
and makes every linked root read-only until the exact paired store is repaired
or restored. Destructive recovery-store reset is not part of the core rollout.
Database notes remain usable.

### Restore

All restore paths require `read_write`, a healthy/capacious recovery store, a
verified payload, a matching root identity, safe containment, no symlink
substitution, an existing parent directory, and a destination confirmation
bound to an exact hash or expected absence. The coordinator creates a pending
restore operation before filesystem mutation and repeats the destination check
at final publish. Parent directories are never recreated implicitly.

B2 ships only deletion restore to the absent original path. It writes a
same-directory temporary, applies the verified supported-metadata manifest,
verifies that temporary's bytes and fingerprint, then uses no-replace
publication. It flushes the restored file and destination directory, re-reads
and verifies the exact bytes plus supported security facts, reuses the
tombstoned UUID, commits the projection/binding transition, and completes the
recovery operation last. If the original path is occupied or its parent is
absent, B2 offers verified exact export and does not offer overwrite,
alternate-path restore, or implicit directory creation.

B3 may add explicit alternate-path restore, `Restore as copy`, and overwrite:

- alternate-path restore reuses the tombstoned UUID only when no live binding
  owns it; `Restore as copy` creates a new UUID;
- overwrite is a separate exact-destination confirmation, round-trip verifies
  safety bytes, and uses atomic replace-with-displaced-target preservation;
- displaced bytes receive the 30-day resolved-conflict retention and verify
  before operation-owned artifact cleanup;
- ordinary reuse of a tombstoned path without the restore command remains a new
  note.

Minimal restore for confirmed deletion ships before destructive Delete is
enabled. The product never promises `recoverable until <date>` without the B2
restore/export path.

### Independent recovery access

Beginning in B1, a minimal recovery-only screen opens `notes_recovery.db`
without joining `file_notes.db` or the Database-note store. It can enumerate,
verify, and exact-export every retained safety snapshot, draft, conflict side,
and later deletion/protection revision immediately. B3 expands the same screen
with selected-protection checkpoints and history controls; it does not introduce
the independent access boundary.

The recovery-only path performs no in-place file mutation, does not create a
deferred projection state, and operates without opening ChaChaNotes or
`file_notes.db`. Recovery-only in-place restore is deferred until there is a
concrete need and a separately reviewed destination-ownership design.

## Library Notes User Experience

### Information architecture

File-backed notes stay in the existing Local Notes owner inside Library. They do
not become a new top-level route or scope.

With zero active linked roots, the current Database Notes canvas remains the
default without an empty Files group, file-only modes, split workbench, watcher,
or file worker. A pristine installation adds only an unobtrusive `Link notes
folder…` action. If unlinking leaves retained recovery, tombstones, drafts, or
conflicts, the same canvas also shows `Recovery items (N)` for minimal
relink/verify/export and required Attention resolution. Full detached-root
management and general purge are deferred. The action does not mount the file
workbench.
This is the zero-configuration non-degradation contract, not merely a performance
optimization.

Browse mode has two explicit groups:

- `Files on disk`
- `Database notes`

The Database group includes `Open database notes`, which switches the whole
Notes content surface to the existing Database list/create/import/export/Sync
canvas. The combined navigator never reimplements those modes inside the File
workbench. Host chrome keeps `Back to linked notes` visible from the delegated
Database surface, including list, editor, create/Sync, loading, and failure
states. A note opened from that delegated list retains its existing `Back to
list`; host chrome is the cross-source route. Only a Database result opened
directly from the combined navigator labels editor Back as `Back to linked
notes` and returns to its tagged File origin. All other Database handlers remain
unchanged.

The existing local folder sync remains Database-note-only and is labeled
`Legacy folder sync (database notes)`. Milestone A read-only activation rejects
a configured overlap but acquires no mutation lease and does not pause any
non-overlapping legacy pass. From B1 onward, upgrading a root to read/write
drains cooperative shared holders and acquires the exclusive mutation lease;
only then do passive secondary processes show legacy filesystem sync as paused.
The exclusive owner may run a non-overlapping pass through the in-process gate.
Existing generic `New note` creates a Database note. File creation is a separate
explicit command at the current disk-tree location.

With a linked root, the Notes host always mounts even when one source is loading
or failed. File projection/index failures appear only in the Files group or file
workbench. Database snapshot/search failures appear only in the Database group
or delegated canvas. Neither source may replace the whole Notes surface with a
shared loading/error placeholder. With zero roots, the existing Database-only
loading/error behavior remains unchanged.

The navigator has three modes:

- `Browse`
- `Changed this session (N)`
- `Needs attention (N)`

`Changed this session` contains successful working-tree-changing Chatbook
operations for the current process-lifetime session: create, content save,
rename, move, delete, and restore. No-op saves and externally initiated changes
are excluded. Its help text says `Chatbook changes since launch, not Git status;
external changes excluded`. `Needs attention` is durable across restarts.

### Linking and root management

`Link notes folder…` opens the existing directory-selection vocabulary, then the
memory-only preview. Confirmation shows the exact canonical root, supported/
ignored/read-only counts, collisions, legacy overlap, write-safety capability,
and estimated projection/index bytes. It explicitly says that every readable
body will be cached and token-indexed as owner-only plaintext in `file_notes.db`;
this derived cache is separate from exact recovery copies. Milestone A offers
`Link read-only` only. From the B1 writable milestone onward, a fresh preview may
offer `Link read/write` when every safety gate passes; an existing read-only
root uses the same preview as an explicit `Upgrade to read/write` action. The B3
protection preview separately estimates selected exact-recovery bytes and names
mandatory temporary safety/conflict/deletion payloads that may exist for
otherwise unprotected notes. Activation shows cancellable/resumable progress
without blocking Database Notes.

Each linked root has labeled `Refresh`, `Manage`, and `Unlink` actions. `Manage`
shows canonical path/identity, runtime mode, monitoring health, protection
summary and retained-capacity contribution when those capabilities exist, and
the concrete reason/remedy for any read-only state. It never exposes a vague
disabled control.

The B3 selective-recovery milestone adds `Keep a Chatbook recovery copy` and `Stop keeping a recovery copy`
to file/folder actions. The navigator and editor distinguish inherited folder
protection from an explicit file override. A change previews affected count/
bytes and resulting `Protected`, `Excluded`, or `Protection pending` states
before confirmation.

### Capability presentation

Delivery and runtime capability are explicit:

| Capability | Read-only source (A) | Journaled writing (B1/B2) | Selective recovery (B3) |
| --- | --- | --- | --- |
| Browse, search, preview, copy/export, Console handoff | Enabled | Enabled | Enabled |
| Create/save/move/rename file in existing folders | Not shipped | Enabled in B1 when root safety gates pass | Enabled |
| Confirmed file delete plus minimal deletion restore | Not shipped | Enabled together in B2 after its recovery gate | Enabled |
| Independent retained-content list/verify/exact-export | Not shipped | Enabled in B1 | Enabled |
| Protection rules and per-note checkpoint/history restore | Not shipped | Not shipped except B2 deletion restore | Enabled |

Before a capability ships, its controls are absent rather than decorative.
After it ships, a root/note that cannot use it shows the stable disabled action
with its exact blocking state and recovery path. Database-note actions retain
their existing availability independently.

In Milestone A, opening a supported file shows its decoded body in a
keyboard-scrollable, selection-capable read-only text control. It is a reader,
not an editable but unsavable buffer: it emits no dirty state, creates no draft,
starts no autosave, and shows no Save, Edit, Create, Rename, Move, or Delete
control. Its actions are Preview, copy body/exact source, export exact source,
and Use in Console. A body-ineligible file shows only its path/title diagnostic
and never presents a cached body as current.

The same reader is used after B1 whenever a clean file is runtime-read-only. A
dirty buffer that later becomes offline or read-only is not converted to the
ordinary reader until its draft is durably retained, explicitly exported,
successfully saved, or explicitly discarded.

### Wide layout

When the Notes canvas itself has at least 110 columns after shell chrome, it
shows a split workbench:

The diagram shows B1 writable capability. In A, `File editor`, save state, and
Save/Actions are replaced by the read-only File body reader and its
Preview/Copy/Export/Console actions.

```text
┌ Navigator ──────────────────┬ File editor ──────────────────────────────┐
│ Browse | Changed | Attention│ Root / folder / note.md   Saved to disk │
│ Search…                     │                                           │
│ ▼ Files on disk             │ body editor                               │
│   ▼ research                │                                           │
│     note.md                 │                                           │
│ ▶ Database notes            │ Save now  Preview  Use in Console  Actions│
└─────────────────────────────┴───────────────────────────────────────────┘
```

The navigator is collapsible. The editor is the dominant surface and has one
scroll owner. Existing Library chrome/rails remain governed by ADR-011 and
ADR-015; this design does not add a permanently competing third Notes pane.

### Narrow layout

Below the split threshold, Notes has `Navigator` and `Editor` views. Opening a
note switches to Editor; Back/Escape returns to Navigator when no closer modal
or conflict action owns Escape. The views do not remount the logical editor
state. Draft, cursor/selection, focused control, search text/results, tree
expansion, current mode, and scroll position survive view switches and terminal
resizes.

### Search

Typing in navigator search replaces the tree/mode contents with results. Clearing
it restores the prior mode and tree expansion.

Search uses the dedicated navigator repositories, not the capped existing
`search_notes` call:

- input is debounced by 150 ms and tags independent File and Database workers
  with the same query generation;
- each source publishes ready results without waiting for the other and returns
  status, at most 100 metadata rows/snippets, a source-local continuation
  cursor, exhaustion state, and an optional user-facing error;
- results always render as `Files on disk`, then `Database notes`; scores are
  never compared across their different FTS indexes;
- Files rank exact filename, filename prefix, path substring, then file FTS
  rank, with normalized path/UUID tie-breakers; Database notes rank exact title,
  title prefix, then Database FTS rank, with normalized title/UUID tie-breakers;
- duplicate path/FTS matches collapse within the File group;
- each group owns `Load more files` or `Load more database notes`; PageDown loads
  only the selected group's next page;
- no full note body is loaded merely to render the navigator;
- every result text-labels its source; File rows include root/relative path plus
  freshness/read-only state, never only a bare title;
- one source's loading, failure, cancellation, or unavailable state preserves
  the other source's rows and shows an inline source-specific Retry;
- a failed later page also preserves that source's already loaded rows and
  cursor and shows an inline tail Retry for the same continuation cursor;
- filename/path results remain available while body indexing is incomplete or
  a file is body-unreadable, but stale body matches/snippets are suppressed;
- selection is keyed by source kind, note UUID, and query generation;
- clearing search cancels both publications and restores the prior mode, tree
  expansion, selection, and scroll anchor;
- there is no fuzzy, regex, or semantic navigator search.

Initial body indexing is incremental, resumable, and visibly labeled. Frontmatter
is never indexed. Tree children are loaded in deterministic batches of at most
200, including a single directory with thousands of sibling files.

### Editor and status

The editor header has a two-line maximum. It always preserves the exact
root/path identity through middle truncation plus focus/help disclosure, shows
source authority, and shows only the highest-priority actionable state. Status
priority is:

1. user bytes at risk or not on disk: unrecoverable/retained draft, save failure
   with a draft, or disk-changed conflict;
2. other durable Attention;
3. offline/read-only capability;
4. ordinary saving/saved state;
5. monitoring, indexing, protection, and recovery information.

Root state never hides an unsafe draft. Combined copy is allowed when necessary,
for example `Draft retained locally • root offline • not on disk`, and names an
immediate Retry, Export, Compare, or Resolve action. Navigation that would
destroy the only in-memory copy is vetoed. Lower-priority details live in
`Status` or `Manage`; they do not become a badge wall. File-backed notes do not
show an independently editable title field; rename is a file action.

When opaque frontmatter exists, the header shows `Frontmatter preserved
(hidden)`. Its action opens a read-only exact-source view and offers the
binding-local `Treat entire file as plain body` choice. While that override is
active, a distinct `Frontmatter-like block treated as body` badge offers
`Re-detect frontmatter`; the UI never falsely labels those bytes hidden. Hidden
frontmatter is never mistaken for absent content.

When their rollout/runtime capability exists, primary visible actions are:

- `Save now`
- `Preview` / `Edit`
- `Use in Console`

The capability-filtered secondary menu is labeled `Actions`:

- Export
- Copy
- Move
- Rename
- History

Delete remains visually separate.

Save status vocabulary is exact:

- `Draft • not on disk`
- `Draft retained locally • not on disk`
- `Draft not yet recoverable`
- `Saving`
- `Saved to disk`
- `Saved to disk • search index updating`
- `Saved to disk • projection needs attention`
- `Saved to disk • recovery needs attention`
- `Disk changed while editing`
- `Save failed • draft retained`

Disk-save state is separate from capability/recovery state. The latter includes
`Disk • Read/write`, `Disk • Read-only: <reason>`, `Root offline`,
`Protection pending`, `Recovery copy current`, `Recovery copy behind`,
`Recovery capacity reached`, `Monitoring degraded`, and `Mixed newlines • autosave
paused`. Every blocking badge links to or names the remedy; color is never the
only carrier.

Recovery UI is called `Chatbook recovery history`, not `backup` or `protected
history`.

The existing word count remains visible. Database-note keyword and local-link
editing remains unchanged. File-specific keywords/links are deferred rather
than coupling file projections to Database-note sync relations. Database-note
sort options keep their current behavior, while file browsing is path-ordered
and combined search is source-grouped and relevance-ordered only within each
source.

### Keyboard and focus contract

- `F6` / `Shift+F6` cycle major available Notes regions per the existing
  Workbench convention.
- `Ctrl+S` invokes `Save now` for the open editable note.
- `/` focuses navigator search only when a text editor/input is not focused.
- `Enter` opens the selected navigator item.
- `Left`/`Right` collapse/expand tree nodes using the standard Textual tree
  contract; PageUp/PageDown move through loaded rows and fetch the next search or
  high-fanout batch at the end.
- Escape from navigator search clears it first and restores the prior tree/mode.
- Narrow-mode Escape returns from Editor to Navigator only after modal,
  comparison, and focused-widget Escape handling decline it.
- Single-letter Library/file shortcuts, including `u`, never fire while the
  body editor or another text input is focused. The visible `Use in Console`
  button remains available.

All destructive and conflict actions are reachable without memorized shortcuts.

Focus transitions are deterministic:

| Transition | Resulting focus after its leave/save guard succeeds |
| --- | --- |
| Open File row in wide mode | File reader/editor; selected row remains the navigator anchor |
| Open File row in narrow mode | Editor view and File reader/editor |
| File Editor to Navigator | Original composite row and scroll anchor |
| Open Database result directly from combined navigator | Database title input; tagged editor Back target retains the File origin |
| Open Database note from delegated list | Database title input; editor Back target retains that Database list/selection |
| Database editor Back to delegated list | Existing Database list and selected row; host File target remains |
| Direct-result Database editor Back | Exact originating File/search row |
| Host `Back to linked notes` from any delegated Database state | Exact originating File/search row |
| Database Back without a host/File target | Existing Database behavior |
| Wide to narrow | Navigator if focus was there or no note is open; otherwise Editor |
| Narrow to wide | Same logical control in its now-visible region |
| Clear search with Escape | Restored prior mode while focus remains in search |
| Close modal/comparison | The control that opened it |
| Focus target disappeared | Same-source selected row, then source header, search input, canvas host |

`F6`/`Shift+F6` visit only mounted, visible, enabled regions. Wide File order is
navigator search, navigator rows, body reader/editor, primary actions, then
Status/Manage. Narrow mode cycles only the active view. Focus restoration uses
logical IDs, never stale widget references, and never targets a hidden,
unmounted, or disabled control.

Delegated Database Escape keeps its existing focused-widget behavior and does
not trigger host return. `Back to linked notes` is the explicit cross-source
route.

## Existing Workflow Parity

### Preview and copy/export

Preview renders the current visible body and never parses frontmatter. In A the
visible body is the disk reader; after B1 it may be an unsaved draft.

File-backed actions distinguish body/draft from exact source:

- `Copy body` in A or `Copy draft body` after B1 when dirty
- `Copy exact saved source` after validating the current disk raw hash
- `Export body as .md/.txt` in A or `Export draft as .md/.txt` after B1
- `Export exact saved source` after validating current disk raw hash

When a conflict is active, draft export names include a `draft` prefix and disk
export names include a `disk` prefix so they cannot be mistaken for resolution.

Database-note exports retain current normalized/frontmatter-injecting behavior.
File-backed exact export never injects `note_id`, frontmatter, or metadata.

Mixed Database/file bulk export is deferred. Existing Database-note bulk export
must not silently include file projections. File-source bulk export can be added
later through the explicit combined read repository with exact-byte and
missing/conflict reporting.

### Templates and import

Existing template actions continue to create Database notes only. File creation
in B1 starts as a blank file with an explicit safe filename in an existing
directory. File-template creation is deferred; if added, it must use the same
journaled no-replace create command and may not inject a UUID or frontmatter.

The generic import action is labeled `Import into Database notes`. Selecting an
already-supported file within a linked root opens its existing binding rather
than duplicating it into SQLite.

`Select all shown` applies only to loaded visible leaves, never undisplayed
pages or collapsed descendants.

### Console handoff

`Use in Console` stages a versioned visible-body snapshot up to 80,000
characters, and that snapshot, not a 4,000-character UI preview, must reach the
provider context path. A body over the cap is never silently truncated. The
handoff asks `Stage current selection` when a nonempty selection fits,
`Stage first 80,000 characters`, or `Cancel`, and shows `80,000 of N characters
staged`. Any later provider-budget truncation is reported separately before
send and cannot imply that the entire note reached the model.

Handoff metadata contains:

- opaque `note:<uuid>` identity;
- root label and relative path, never the absolute path;
- `base_disk_raw_hash`;
- `snapshot_semantic_hash`;
- `disk_raw_hash` only when the staged snapshot is the current disk body.

Logs and status text never contain note bodies.

For an unresolved conflict, handoff asks:

- `Stage draft`
- `Stage disk`
- `Cancel`

Handoff does not resolve the conflict.

## Search, RAG, and MCP Boundaries

### FTS and RAG

Dedicated file FTS projection is required and eventual within the near-real-time
target. File RAG is outside A through B3 completion. A later design must preserve
disk authority, suppress missing/deleted projections immediately, and ensure
RAG lag or failure never blocks disk save or FTS; no RAG queue or schema is
introduced by this rollout.

### MCP

Existing MCP note resources, search, create, update, and delete remain
Database-note-only. File projections do not leak into them through the `notes`
table. A future explicit MCP read adapter may use the combined read repository
and return file freshness/binding state. MCP filesystem writes remain outside
this design.

## Sync and Association Isolation

File-backed notes are local-only projections:

- existing notes Sync v2, sync-log, FTS, keyword, and relation triggers cannot
  see `file_note_projections`;
- `NotesScopeService` remains Database-note-only and never enqueues file
  projections;
- remote inbound operations have no table or service route to a file projection
  or disk;
- file-specific associations are not shipped in the core rollout;
- legacy folder sync is rejected for an overlapping active file root.

Legacy sync configuration detection includes its database roots, both TOML key
aliases, timer state, `sync_profiles.json`, and per-note bindings. Disabling
legacy overlap does not touch its files. Version 1 performs no automatic legacy
conversion and never reuses a legacy UUID that has pending or acknowledged Sync
v2 identity.

Beginning with B1, a process-wide `LegacyRootOwnershipGate` is injected into
`NotesSyncEngine.sync()` and every direct legacy mutation entry point. Library
timers, `NotesSyncService`, and `AutoSyncManager` must route through that
engine-level gate.

The B1 gate is backed by the same per-user cross-process root-mutation lease used
by the file coordinator for writable ownership. Every cooperative B1-capable
legacy pass acquires shared OS ownership before scanning and holds it for its
complete mutation lifetime. Upgrading a linked root from `read_only` to
`read_write` atomically closes new shared admission, waits for all
current-process and other-process shared holders, and acquires exclusive OS
ownership for the writable root's lifetime. While that exclusive lease is held,
secondary processes run no legacy filesystem mutation. The lease-owning process
may admit a non-overlapping legacy pass under its stronger exclusive ownership
through the in-process canonical-root gate; an overlapping pass is always
rejected. A UI-only or process-local overlap check is insufficient. Milestone A
uses neither this gate nor the mutation lease; it rejects configured overlap and
treats legacy/external changes as reconciliation input. Older tools that do not
honor the lease remain external writers under the stated support boundary.

## Runtime Modes and Concurrency

The file coordinator has three same-version modes:

| Mode | Behavior |
| --- | --- |
| `disabled` | Kill switch for activation, monitoring, and file commands. Existing projections are diagnostic and make no freshness claim. Startup still inspects incomplete operations without replaying them. |
| `read_only` | Link, inventory, tree/search, monitor, reconcile external changes, and inspect/export recovery when available. No file mutation or restore into a linked root. |
| `read_write` | Enables explicit coordinator mutations after all health, lease, and recovery gates pass. |

`disabled` is not a schema downgrade. An incompatible `file_notes.db` version
disables only the Files source; it does not alter or gate the existing
Database-note store.

A mode-transition barrier rejects queued new saves, lets an already-started
operation finish or enter recovery, preserves newer editor buffers, and only
then changes mode. After `disabled`, each affected root completes identity and
hash reconciliation before it may claim freshness or return to `read_write`.
Applying any retained draft still requires a final fresh disk hash.

An owner-only per-user File Notes coordinator-election lease, stored in a fixed
application runtime namespace independent of configurable user-data,
main-database, and repository paths, permits one active root and elects the
process allowed to activate, monitor, and reconcile/project it. Read-only A
holds only this election lease. It has no legacy shared/exclusive semantics.

From B1 onward a separate coarse root-mutation lease permits shared cooperative
legacy passes only when no read/write File Notes root owns it exclusively. A
root must hold both coordinator election and exclusive mutation ownership before
any file command is admitted. This intentionally replaces a multi-profile
durable root registry in the first release. Lease diagnostics record a random
owner nonce, process-start identity, canonical root, and capability; PID text is
never ownership proof. Kernel release plus root-identity and
incomplete-operation checks govern stale recovery.

A second Chatbook process is passive read-only: it may inspect cached projection/
recovery and exact-export a freshly hashed file, but it starts no second watcher/
reconciler and issues no file commands. It may continue non-overlapping legacy
sync during A; from B1 it runs no legacy filesystem sync while another process
holds the exclusive mutation lease. The core pilot permits one linked active
root total. The schema remains multi-root capable, but additional roots are a
separately gated expansion. Multiple hosts and network mounts are outside
write-safety guarantees.

### Upgrade and process support boundary

The core support contract is one current Chatbook installation, one configured
storage profile, one active file root, and one cooperative file coordinator.
Read-only activation uses coordinator election and does not require draining
legacy mutation. B1 read/write upgrade closes admission of new cooperative
legacy passes, waits a bounded interval for current shared holders to drain, and
then acquires exclusive root-mutation ownership. It fails without changing mode
only when that drain times out or another exclusive owner remains; the remedy
identifies the holder and tells the user which Chatbook process or legacy pass
must finish.

The dedicated File Notes databases do not rewrite existing Database-note
triggers, rows, or schema. An already-running pre-feature or older process has
no API knowledge with which to mutate a file projection and simply ignores
`file_notes.db`. An old copy, different profile, editor, or Git tool that writes
the root is treated as an external writer and is contained by the same raw-hash,
displaced-target, and reconciliation protocol.

Chatbook does not claim that it can enumerate or launcher-disable every arbitrary
source checkout or virtual environment. Durable multi-profile registration,
managed-launcher version floors, and simultaneous writable storage instances
are deferred product capabilities, not hidden prerequisites for Notes.

### Startup operation recovery

After `file_notes_storage` records a paired recovery store, every runtime mode,
including `disabled`, opens and identity-checks that recovery journal far enough
to inspect incomplete operations. Read-only A does not create or require a
recovery journal before that pairing exists. A startup process may classify,
repair projection state, or clean an operation artifact only while it holds both
the File Notes coordinator election and exclusive root-mutation ownership.
Passive processes may inspect and report incomplete operations but perform no
filesystem cleanup, journal transition, or projection reconciliation. The
elected owner examines current root identity/containment plus
source/destination paths, hashes, and metadata fingerprints and never blindly
replays a filesystem mutation.

Classification is action-specific:

- save/create/restore: intended target plus the expected displaced/absent state
  means finish recovery metadata and projection; old target/expected absence
  means `not_applied` with draft retained; any unexpected target or displaced
  artifact or metadata fingerprint means Attention with every stable side
  captured;
- delete: verified target absence plus absent quarantine and a valid deletion
  snapshot means complete the tombstone; the unchanged expected file means
  `not_applied` without unlinking; an operation-owned quarantine is classified
  and either safely finalized/restored or retained in Attention; a recreated/
  different target or unexpected metadata fingerprint also means Attention;
- rename/move: absent source plus intended destination means complete the
  binding move; intended source plus absent destination means `not_applied`;
  both paths present, both absent, unexpected hashes, or unexpected metadata
  fingerprints mean Attention. No operation artifact is cleaned in any
  metadata-divergent state.

Recovery metadata is repaired before the File Notes projection, which is then
rebuilt idempotently from the observed disk result. If the expected
`notes_recovery.db` is absent, mismatched, or invalid, every linked root is
read-only, no restore is allowed, and no empty replacement is auto-created;
Database notes continue to operate.

## Migration and Activation

### File-store bootstrap and orphan safety

This feature performs no ChaChaNotes migration, advances no existing main schema
version, wraps no existing constructor, and introduces no Database-only
compatibility token. A File Notes bootstrap failure cannot change Database Notes
startup, CRUD, search, Sync, backup/restore behavior, or availability.

`FileNotesStoreBootstrap` runs only after a user confirms the first read-only
link. Under its storage-instance creation lock it:

1. verifies the owner-only storage directory without following links;
2. applies the pairing/orphan matrix before opening or creating a database;
3. exclusively creates a missing `file_notes.db` and its initial schema in one
   transaction only when no recovery database, sidecar, or prior-pair marker
   exists;
4. verifies schema version and integrity before publishing any root state.

An existing corrupt, incompatible, partial, or orphan-indicating database or
marker is preserved for diagnosis. File Notes remains disabled or recovery-only;
Chatbook never renames it aside and silently starts empty. Future
`file_notes.db` schema changes use independently versioned transactional
migrations and must fail within the File Notes boundary. Speculative migration,
downgrade, and pair-swap machinery is not part of the first schema.

Before B1 enables a write command, `RecoveryStoreBootstrap` runs while holding
both coordinator election and exclusive root-mutation ownership and applies the
pairing matrix. It may begin first pairing only from a healthy unpaired
`file_notes.db` with complete absence of recovery evidence. It persists and
directory-flushes the exact S/X/G bootstrap marker, commits
`recovery_store_identity` S/X/G, commits expected S/X/G in
`file_notes_storage`, verifies both, then durably removes the marker. Startup
resumes only an exact matching intermediate state. After pairing,
missing/mismatched storage is
Attention/read-only, never permission to initialize a fresh store.

Existing Settings database backup/restore controls remain Database-note-store
controls and must be labeled as excluding linked source files, `file_notes.db`,
and `notes_recovery.db`. They never live-copy or overwrite either File Notes
database. A future quiescent File Notes store backup/relocation design is
separate; ordinary recovery-only exact export is available beginning in B1.

### Preview

Root preview is ephemeral and memory-only:

- no root/projection rows;
- no watcher or lease;
- no recovery payload;
- no index publication;
- no file writes.

It reports supported, ignored, read-only, inaccessible, and special counts plus
estimated initial indexing bytes. B3 adds a separate protection preview with
selected file/folder counts and estimated recovery bytes.

### Confirmed activation

Read-only A activation is crash-resumable:

1. Acquire the File Notes coordinator-election lease; acquire no mutation lease.
2. Canonicalize/fingerprint the root, reject a second active root and configured
   legacy-sync overlap, and establish `read_only`.
3. Persist a durable root record in `activating` while retaining election.
4. Start bounded watcher/event capture.
5. Inventory path metadata and classification.
6. Drain captured events.
7. If capture overflowed or root identity changed, discard the inventory and
   rescan.
8. Atomically publish the root and file projections.

A crash leaves an explicit resume/discard activation state. Path navigation
becomes available from the published metadata while body FTS proceeds
incrementally. Source files remain unchanged.

B1 upgrade to `read_write` is a separate capability transition:

1. Repeat preview and verify the checked-in B0 platform/version capability
   result against the actual root volume and current metadata.
2. Close new cooperative legacy-pass admission, await current shared holders,
   and acquire the root-mutation lease exclusively while retaining coordinator
   election.
3. Revalidate root identity, configured overlap, native write primitives,
   metadata round-trip eligibility, and the in-process legacy ownership gate.
4. Bootstrap or verify the exact recovery pairing and operation capacity.
5. Pass the mode-transition barrier and persist `read_write`.

Failure at any upgrade step leaves the linked root usable read-only and does not
pause non-overlapping legacy sync after the failed exclusive transition is
released.

From B3 onward, selected protected items show `Protection pending`; each becomes
writable after its baseline replica verifies. In B1/B2, and for unprotected
items in B3, items become writable after root activation and per-operation
recovery health gates pass.

## Unlink, Forget, and Retained History

### Unlink

Unlink crosses an operation barrier, stops monitoring, and releases coordinator
election plus any B1 mutation ownership. It never touches source files or Git.
The root becomes detached while retaining UUIDs, metadata, drafts/conflicts, and
recovery bytes. Cached editable bodies and their plaintext FTS tokens also
remain until Forget; recovery payloads still count toward the fixed capacity. A
detached root does not reserve a global path after both leases are released.

Relink requires explicit confirmation plus root identity and inventory checks.
A recreated directory at the same path does not silently inherit old UUIDs.

### Forget

Forget removes the root and file projections. Because file FTS has no projection
triggers, the same `file_notes.db` transaction explicitly removes every FTS row
for that root before deleting projections; completion verifies that no body,
snippet, or token row remains. It does not touch source files. Forget is blocked
while pending/Attention operations or unresolved drafts exist; those must be
resolved or separately discarded first. Self-contained deletion snapshots
remain through their guaranteed expiry. Ordinary current replicas and
checkpoints lose their protection pin and become eligible for capacity pruning;
the confirmation reports those logical bytes and recommends Unlink when the user
wants to retain them.

### Minimal history controls

Automatic garbage collection removes expired, superseded, and otherwise
eligible revisions. A scoped explicit action may prune eligible ordinary
checkpoints after showing item count and logical bytes. It cannot remove pending
operations, unresolved drafts/conflicts, current protected replicas, or
unexpired guaranteed deletion/conflict payloads. The core release does not
offer guarantee waiver, general purge, pair backup, or pair restore.

Application-consistent backup of the paired File Notes databases is a later
optional feature. Source files and Git would remain outside such a backup, so it
would not replace the user's repository or an off-device backup.

## Security and Privacy

- `file_notes.db`, `notes_recovery.db`, their WAL/SHM files, lock metadata, and
  temporary recovery artifacts use owner-only app-directory permissions (POSIX
  0700 directories and 0600 files where supported, equivalent owner-only policy
  elsewhere).
- Writable activation is refused if that protection or symlink-substitution
  safety cannot be enforced.
- Every writable path is reached from a pinned root handle through no-follow,
  descriptor/handle-relative traversal. Newly created Chatbook files start
  owner-only. Displaced/quarantined originals retain their verified source
  security facts until Chatbook can safely narrow them, and those facts are
  available for rollback/restore. Writable activation is refused where
  equivalent beneath-root and atomic displaced-target primitives are
  unavailable.
- Linked-root file permissions are not broadened. Replacement uses freshly
  observed supported security facts and verifies the result; a platform where
  non-broadening cannot be proved remains read-only.
- Absolute paths, bodies, raw hashes, and recovery payloads are excluded from
  routine logs. Exceptions are sanitized before user display.
- Both File Notes databases are unencrypted plaintext SQLite. `file_notes.db`
  contains root paths, filenames, every readable decoded editable body, and a
  token index even when protection is off. `notes_recovery.db` contains exact
  bytes only when mandatory mutation/conflict/delete retention or selected
  protection/history requires them. Protection selection controls the recovery
  guarantee, not whether a readable body is cached/indexed.
- Same-user processes, device backups, and anyone with access to an unencrypted
  disk may read either store.
- Full-disk encryption and an independent off-device backup are recommended.
- The Git remote protects only content that the user has committed and pushed.

## Performance and Non-Degradation Gates

- A pristine profile with no `file_notes.db` and no recovery evidence creates
  or opens no File Notes database. An existing `file_notes.db` may receive one
  short read-only bootstrap query to discover linked, detached, or retained-
  recovery state, then closes when no root is active. In either zero-root case
  Chatbook starts no long-lived File Notes connection, watcher,
  reconciliation/index worker, recovery writer, coordinator election, mutation
  lease, or file scan; genuine retained evidence may expose only the
  `Recovery items` action.
- Existing Database Notes schema, constructors, startup, CRUD/search, Sync, and
  backup/restore controls remain unchanged. `file_notes.db` absence, corruption,
  incompatibility, or indexing failure cannot replace the Database canvas with
  a shared error state.
- A cached 5,000-file root reaches interactive tree navigation without reading
  file bodies before first paint.
- Tree rows carry metadata only and load children lazily in bounded batches,
  including a 5,000-sibling directory.
- Navigator search returns its first 100-result page within 200 ms p95 on the
  fixed benchmark fixture, after warm-up and over at least 30 samples; canceled
  stale queries never publish over a newer generation.
- A settled local edit reaches projection and eligible FTS visibility within 2
  seconds p95 on the healthy fixed benchmark runner.
- FTS retry consumes no recovery-journal capacity and never blocks a later file
  command after the projection operation has completed.
- No individual file-notes UI-thread callback measures 100 ms or more in the
  benchmark trace.
- Polling fallback over an unchanged 5,000-file fixture averages at most 5% of
  one CPU core over 60 seconds, with a bounded, non-growing event queue.
- Full root scans, RAG, initial body indexing, compression, hashing, and File
  Notes store maintenance are off the startup/UI critical paths.
- Resize preserves state and never creates two nested editor scroll owners.

The benchmark manifest pins runner class, OS/filesystem, Python/SQLite/Textual
versions, both deep-tree and 5,000-sibling generated fixtures, warm-up, sample
count, and timing boundaries. Shared CI verifies deterministic behavior; the
pinned runner owns hard timing gates.

## Verification Strategy

### Focused per-PR tests

- Structural isolation: the existing Database-note schema version, tables,
  triggers, constructors, and CRUD behavior remain byte-for-byte unchanged;
  `file_notes.db` failure cannot impair them; and file projections cannot leak
  into Database-note counts, search, export, RAG, MCP, keyword, relation, or
  Sync paths.
- Projection idempotency by root, normalized path, raw hash, and presence state.
- Storage-instance namespacing, every row of the pairing/orphan matrix, no
  silent rebootstrap/adoption after loss or relocation, coordinator-election
  contention, and one active root.
- Read-only capability on macOS/Linux/Windows plus APFS-specific writable
  capability probing, the checked-in B0 version-matrix artifact, and fail-closed
  downgrade elsewhere.
- Exact frontmatter/BOM/newline/final-newline/mode preservation, raw-hash-bound
  normalization acknowledgement, and deterministic new-file byte defaults.
- Strict encoding, size, malformed-frontmatter, hardlink, symlink, unreadable,
  unsafe-name/control rendering, nested-mount/cross-device mutation refusal, and
  special-file classification.
- Unsupported ACL/xattr/flag/ownership/alternate-stream admission plus metadata
  introduced during the publication race, proving no mutation silently drops it.
- Versioned hash vectors plus rejection of unknown hash versions.
- Save no-op behavior and editor-generation races.
- Every mutation outcome at each journal checkpoint, destination-path locking,
  no-replace publication, displaced-target/exchange classification,
  delete-quarantine recovery, and operation-linked artifact cleanup.
- Recovery/full-synchronous commit barriers, post-commit safety-blob
  round-trip verification, per-action file/directory fsync, normal complete-last
  ordering, and completed-operation pruning.
- Projection-commit failure keeps an operation pending/fail-closed; FTS failure
  after projection commit completes the operation, leaves indexing pending,
  permits later writes, and never surfaces a stale index generation.
- Expected-hash conflict, conflict-headroom failure, overwrite/discard safety,
  resolved-side retention, exact-export fallback, and durable Attention.
- Missing/root-Offline/tombstone transitions, atomic-save versus genuine move,
  manual reassociation, and path-reuse identity.
- Delete snapshot verification, 30-day pin, stale confirmation, quarantine, B2
  absent-original restore, and occupied/missing-parent exact-export fallback.
- Protection inheritance/overrides, current replica invariant, fixed capacity,
  free-space floor, corrupt
  revision, bounded decompression, payload/operation garbage collection, and
  recovery failure.
- Independent File/Database paging, source-grouped rank/dedup, partial-result
  failures, latest-query cancellation, high-fanout batching, stale-body
  suppression/path-only fallback, and durable incremental file-FTS progress.
- A coordinator-election contention without legacy-sync pause; B1
  shared/exclusive mutation ownership and engine-level canonical-root tokens;
  Database/file repository boundaries.
- B1 recovery-only enumerate/verify/exact-export without either application
  database.
- Every focus-transition-matrix row, narrow/wide resize, autosave, unsafe-draft
  status priority, source leave guards/return targets, root/protection flows,
  conflict, preview, exact export, and explicit Console handoff truncation.
- Exact zero-root Database Notes UI/runtime non-degradation apart from the
  specified Link/Recovery actions, plus active-root Database-source routing
  parity for create, edit/autosave/conflict, keywords/links, template/import/
  export, and legacy sync.

Minimal test seams are:

- one coordinator fault-checkpoint callback;
- an injectable clock;
- a fake watcher event source;
- a coordinator command port for deterministic controller tests;
- a navigator query repository for deterministic paging/cancellation tests.

Integration tests continue to use real `pathlib`/file descriptors, SQLite, and
temporary files rather than a broad fake filesystem abstraction.

### Nightly and release-blocking suites

- Process termination at every journal and activation boundary, proving complete
  bytes and recoverable intent after process crashes.
- Recovery COMMIT/fsync failures before file mutation, projection-commit/recovery-
  complete ordering, deletion retention, and revision-GC reference swap.
- Real watcher timing, duplicate/move/overflow storms, polling fallback, and Git
  checkout/bulk/conflict-marker reconciliation.
- Deep and 5,000-sibling scale plus fixed-runner p95 benchmarks.
- Two-process shared-legacy/exclusive-file lease contention, secondary-worker
  refusal, PID reuse diagnostics, and stale-owner recovery.
- Running legacy-sync shared-token drain and overlap fencing at every engine
  entry point and across cooperative processes.
- Root identity change and same-path directory recreation.
- File-store exclusive creation, corruption/incompatible-schema isolation, every
  recovery bootstrap crash/mismatch/orphan boundary, and proof that neither
  store silently creates, adopts, or replaces the other.
- Mode change during mutation, recovery corruption/full disk/capacity, unlink,
  relink, forget, eligible-history pruning, and occupied-destination restore.
- B0 packaged APFS native exchange/exclusive/no-follow, metadata, and durability
  matrix on every supported macOS release; packaged Linux/Windows tests prove
  the same roots remain explicitly read-only.

Process crash tests do not claim to simulate power loss. Commit and directory-
fsync ordering is fault-injected, while each supported writable
platform/filesystem durability profile also receives the named B0 release
power-cut/reboot validation artifact. Network/cloud mounts remain outside
release guarantees.

The Git smoke matrix is intentionally small: status/diff, one local commit,
checkout/bulk reconciliation, and conflict markers. Push remains ordinary Git
behavior outside Chatbook and does not justify a remote test matrix here.

## Rollout

Delivery is split into atomic PR-sized children:

1. **A0 storage/isolation:** create isolated `file_notes.db`; leave ChaChaNotes
   untouched and contain File-store failures.
2. **A1 discovery/preview:** preview and link one read-only root using only
   coordinator election, with explicit plaintext-cache disclosure.
3. **A2 projection/search/reconciliation:** scalable metadata/body projection,
   triggerless retryable FTS, watcher/polling reconciliation.
4. **A3 workbench:** tree/search-replacement UI, metadata-only pageable
   Database read adapter, read-only file reader, source routing/return targets,
   partial-result handling, responsive focus.
5. **A4 workflow/parity:** exact copy/export, Console handoff, more-than-100-note
   Database paging verification, read-only Unlink/Relink/Forget, Settings
   exclusion labels, and active-root/zero-root Database-note parity.
6. **B0 APFS proof:** executable packaged probe and checked-in supported-macOS
   capability matrix. It may proceed after A1 but must pass before B1.
7. **B1a gated write foundation:** recovery pairing, journaled
   create/save/autosave internals, mandatory safety, create/save startup
   classification and owned-artifact cleanup, and independent recovery-only
   enumerate/verify/export. It exposes no writable control or mode transition.
8. **B1b writable completion/release gate:** rename/move, complete conflict
   actions, mode transitions, writable Unlink/Forget barriers, and rename/move
   startup classification using the shared classifier. B1 controls appear only
   after this whole gate passes.
9. **B2 delete/minimal restore:** deletion snapshot, quarantine, 30-day
   guarantee, absent-original restore, and exact-export fallback.
10. **B3a selective protection:** file/folder rules, verified current replicas,
    fixed-capacity admission, and behind/pending state.
11. **B3b history/expanded restore:** coalesced checkpoints, retention, history,
    alternate-path/copy/overwrite restore, and protected-history Forget
    consequences.
12. **Deferred expansions:**
   - Linux/Windows writable adapters, additional roots/storage instances,
     multi-profile durable registry, recovery-store relocation/clone, folder
     mutation, file templates,
     file-specific keywords/links, file MCP/RAG, configurable quotas, guarantee
     waiver/general purge, paired-store backup/restore, recovery-only in-place
     restore, Git controls, and remote/portable file identity or synchronization.

TASK-399 remains a roll-up tracker and is never implemented as one PR. B2 and
B3a are sibling branches after B1b; B3b depends on both the restore primitive
and selective-replica foundation. Every child owns scoped acceptance/performance
tests and links ADR-021. No milestone exposes an action whose recovery and
fail-closed prerequisites belong only to a later milestone.

## Alternatives Considered

### Keep legacy equal-peer bidirectional sync

Rejected. It retains ambiguous ownership, timestamp winners, weak move/delete
identity, and avoidable write noise. It remains only as a compatibility feature
for Database notes.

### Keep SQLite canonical and export files

Rejected. Chatbook edits would not immediately appear as ordinary Git changes,
and external editors/Git would require a second import/export authority loop.

### Use disk only, without a projection

Rejected. It would degrade Library FTS, local identity, Console handoff, and
large-tree performance.

### Put file projections in `notes` with a `storage_kind` discriminator

Rejected after repository integration review. Existing Database-note CRUD, FTS,
Sync, MCP, RAG, export, keyword, and relation callers assume every active
`notes` row is Database-owned, and several bypass `NotesScopeService`. Gating
every path would make zero-root non-degradation depend on a permanent,
error-prone audit. Dedicated file projection/FTS tables provide isolation by
construction; intentional combined reads happen above both repositories.

### Put dedicated projection tables in the ChaChaNotes database

Rejected after constructor and backup/restore integration review. Separate
tables would avoid `notes` write-path leakage, but would still expand the
existing schema, migration, constructor, backup/restore, corruption, and
availability boundary without making ChaChaNotes a content authority. A sibling
`file_notes.db` provides the same query model while leaving Database Notes
structurally and operationally unchanged.

### Store UUIDs in frontmatter or a repository manifest

Rejected. It would alter user files solely for Chatbook, create Git noise, and
make Chatbook metadata part of repositories that currently contain only notes.

### Store recovery bytes in an application database

Rejected for both ChaChaNotes and `file_notes.db`. A separate recovery database
supports independent enumeration without opening ChaChaNotes or `file_notes.db`
and isolates recovery capacity/corruption from Database-note and projection
availability.

### Ship writable macOS, Linux, and Windows support together

Rejected for the first writable release. Their beneath-root, atomic replacement,
permission, and durability primitives are not equivalent through Python's
portable APIs. APFS is the first tested write target; the same feature remains
useful read-only elsewhere while later native adapters prove equal safety.

### Stage and atomically swap the File Notes database pair

Rejected for first creation. `file_notes.db` and `notes_recovery.db` have
different authority and retention roles, and B1 pairs them through durable
identity plus fail-closed orphan detection rather than pretending their
transactions are atomic. Renaming live database families adds handle, sidecar,
cross-volume, and launcher coordination without making filesystem publication
transactional. Future paired-store relocation/backup requires its own quiescent
design.

### Add Git controls with the first file UI

Rejected for this tranche. Direct file correctness, conflicts, deletion, and
recovery must be proven before Chatbook adds another stateful workflow around
staging or remotes.

## Governance

ADR required: yes
ADR path: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`
Reason: This design establishes long-lived content authority, isolated schemas,
bootstrap, recovery, conflict, sync-isolation, filesystem, and Library UX
boundaries.

Related canonical decisions:

- [ADR-001: canonical ADR workflow](../../../backlog/decisions/001-adopt-backlog-decisions-as-canonical-adrs.md)
- [ADR-003: Settings/Library/RAG ownership](../../../backlog/decisions/003-settings-library-rag-defaults.md)
- [ADR-004: storage restart boundary](../../../backlog/decisions/004-settings-storage-defaults-restart-boundary.md)
- [ADR-008: remote Sync v2 contract](../../../backlog/decisions/008-sync-v2-client-m1-contract-alignment.md)
- [ADR-011: Workbench UI system](../../../backlog/decisions/011-chatbook-workbench-ui-system.md)
- [ADR-015: Library owns Notes navigation](../../../backlog/decisions/015-shell-destination-ia.md)

Legacy `Docs/Features/notes_bidirectional_sync.md` remains historical behavior
for Database notes only. It is not the authority model for linked file roots.
