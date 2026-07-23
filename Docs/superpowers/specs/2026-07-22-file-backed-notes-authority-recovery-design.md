# File-Backed Notes Authority and Recovery Design

Date: 2026-07-22
Status: User-approved revised design; pending final written-spec review
Backlog: [TASK-399](../../../backlog/tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
ADR: [ADR-021](../../../backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md)

## Summary

Chatbook will make existing Git-managed Markdown and text folders first-class
Notes sources inside Library. The bytes on disk remain authoritative. Editing a
file-backed note in Chatbook changes the actual file, so the user's existing
`git status` / `git add` / `git commit` / `git push` workflow continues without
translation, export, or a second synchronization step.

SQLite has two supporting roles, neither of which competes with the files:

1. Dedicated file-note tables in the main Chatbook database hold a searchable,
   derived projection and durable local metadata. Existing Database-note tables
   and triggers remain structurally unchanged.
2. A separate `notes_recovery.db` holds the mutation journal, mandatory
   operation-safety copies, and opt-in recovery replicas/history for selected
   files and folders.

The recovery database is independent of the main notes projection and contains
enough information to list, verify, and exact-export retained notes if the main
database is unavailable. It is still an unencrypted, same-device recovery copy.
It is not an off-device or disaster backup.

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
- the main database is not an independent recovery copy of selected source
  files.

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
| **File-backed note** | A supported regular file under a linked root; disk owns its path, title, and bytes. It is represented only in dedicated file-note projection tables. |
| **Projection** | The dedicated main-database title/body/search representation derived from a file. It is not a Database-note row. |
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

The main database may temporarily lag disk, but it may never overwrite disk merely
because its projection is newer by timestamp.

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
will not broaden access. ACLs, extended attributes, ownership, Windows alternate
streams, and network-filesystem atomicity are otherwise not promised.

## Data Model

The main database gains isolated file-source tables. Existing `notes`,
`notes_fts`, keyword, relation, and sync tables and their triggers are not
modified for file ownership. Exact DDL is assigned in the implementation plan,
but the following state is required.

### Storage-instance identity

The canonical configured main-database path is hashed with a fixed domain tag
to form a stable, non-secret storage-instance ID. The owner-only
`<user-data>/file_notes/<storage-instance-id>/` directory contains that
instance's `notes_recovery.db`, pre-migration backup, diagnostics, and recovery
bootstrap markers. Absolute main-database paths are not logged or used as
filenames.

The core rollout activates only one storage instance/root at a time under the
coarse per-user root-mutation lease. The namespace prevents a custom main DB
from silently sharing another DB's recovery history without introducing
multi-profile coordination machinery.

Path-derived placement is not recovery identity. At first B1 bootstrap Chatbook
generates a random recovery-instance UUID and persists it in both the main
file-notes storage row and a self-contained recovery-store identity row. A
recoverable bootstrap marker handles a crash between those commits. After a
pairing exists, an absent or mismatched recovery store fails closed: Chatbook
never silently creates an empty replacement over evidence of prior B1/B2/B3
state.

Changing the configured user-data directory or moving/copying the main database
can therefore require the corresponding recovery directory to be restored. The
core rollout diagnoses the expected recovery-instance UUID and keeps file
commands read-only. Explicit quiescent recovery-store relocation/clone is a
later administrative capability, not an automatic path heuristic. Before B1
has ever paired a store, the bootstrap may update A's recovery-unpaired storage
binding to the newly derived namespace. After pairing, either a derived/stored
storage-instance mismatch or a recovery UUID mismatch fails closed.

### Main Chatbook database

#### `file_notes_storage`

One file-feature storage row records the derived storage-instance ID, nullable
expected recovery-instance UUID, feature/migration state, and pairing/bootstrap
generation. It contains no note content. Only the explicit
`RecoveryStoreBootstrap` may establish first pairing; changing an established
pair is deferred to the quiescent relocation/clone design.

#### Existing Database-note tables

The `notes` table remains Database-note-only. Existing list, count, search,
create, update, delete, export, RAG, MCP, keyword, relation, and Sync behavior
continues without a new discriminator column or file-row branch. This is the
primary zero-root and non-degradation boundary.

No file projection UUID is a `notes.id`, and no generic Database-note method can
address a file projection. Any future shared keyword/link experience requires
file-specific association tables and an explicit combined read model; it is not
part of the core rollout.

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

Projection presence state is authoritative:

| State | Meaning and projection behavior |
| --- | --- |
| `present` | Path is observed. A readable supported body is eligible for normal projection/search; a body-ineligible diagnostic remains path/title-visible through the navigator only. |
| `missing` | This path is absent while the root is online. Hide it from Browse and search results, but retain the last projected body for diagnosis/recovery. |
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
diagnosis/recovery. The navigator continues to expose a path/title-only row with
the exact read-only reason. It may not show a stale body snippet as current.

#### `file_notes_fts`

A dedicated external-content FTS5 table indexes the filename-derived title and
editable body of present, body-eligible `file_note_projections`. Its triggers
touch only file projection rows. Existing `notes_fts` and its triggers remain
unchanged.

Path search comes from indexed comparison/display-path columns on
`file_note_projections` and is combined with file FTS by the file navigator
repository. Intentional cross-source search unions file results with existing
Database-note results above the two repositories; it never merges their write
paths.

### Independent `notes_recovery.db`

The recovery database has its own schema version, integrity checks, and
owner-only files. It does not depend on joining the main database to enumerate
or verify retained content. Its `recovery_store_identity` row records the bound
storage-instance ID, random recovery-instance UUID, and bootstrap generation
expected by `file_notes_storage`.

#### `note_file_revisions`

Every retained row is self-contained enough for recovery and includes:

- note UUID;
- root label and identity fingerprint;
- exact relative path;
- revision kind;
- versioned raw-hash algorithm/value and raw byte length;
- codec/version, compressed length, and compressed bytes;
- recorded mode/BOM/newline facts;
- creation, expiry, verification, and pin state.

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
- referenced safety/draft/replacement revisions;
- editor buffer generation;
- state, filesystem outcome, main-projection state, outcome, and timestamps.

Its minimal state lifecycle is:

```text
pending -> complete
pending -> attention -> complete
```

`main_projection_state` is independently `pending` or `applied`.

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

`LibraryNotesReadRepository` intentionally combines Database-note reads from
`NotesScopeService` with file reads from `FileNotesRepository` for the Library
navigator, explicit Console handoff, and other approved read-only consumers.
Every result carries a source kind. It is a composition boundary, not a common
write repository.

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
kind. A File result opens the file workbench. A Database result delegates to the
existing `LibraryNotesCanvas` and current Database-note handlers, including
create, edit/autosave/conflict, keywords/links, template/import/export, and
legacy sync. `LibraryScreen` coordinates that source-level switch and forwards
high-level navigation events; it does not absorb the file coordinator or either
editor state machine. With zero roots it mounts the existing Database canvas
directly, apart from the specified Link and retained-Recovery actions.

The workbench keeps the logical editor mounted through Navigator/Editor layout
changes where Textual permits it and uses targeted widget updates rather than
whole-screen recomposition. A dedicated navigator widget lazily mounts bounded
child batches; it never renders a 5,000-row directory as 5,000 eager buttons.

## Mutation Protocol

SQLite and the filesystem cannot form one transaction. Every Chatbook mutation
therefore uses a durable intent journal plus hashes.

The file coordinator owns dedicated critical-write SQLite connections.
`notes_recovery.db` and the main-database transactions that form file-operation
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
   bounded-decompress it, and verify codec, length, and raw hash before touching
   the working tree. A reused current replica qualifies only when its verified
   hash/length match the just-read source.
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
   handles, using stable-read rules, and classify all observed hashes before the
   idempotent main binding/projection transition.
8. After expected displaced bytes are represented by a round-trip-verified
   recovery revision with the applicable retention, unlink the operation-owned
   artifact, fsync its parent, and confirm absence. Unexpected displaced/
   quarantined objects remain named and protected until conflict resolution.
   Cleanup failure leaves the operation pending/Attention.
9. Durably commit that main transition while the recovery operation remains
   `pending`; durably mark the recovery operation complete last.

This removes the silent compare-then-replace loss window: an external version
that wins after preflight but before publication is the displaced target, not
discarded evidence. If its hash differs from the expected base, Chatbook retains
it and enters Attention. An uncooperative process that continues writing through
an already-open descriptor after displacement, plus network/cloud filesystems
and multi-host writing, remains outside the guarantee; Chatbook never deletes a
displaced object until its stable observed bytes are durably captured.

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

Main-database projection is repairable but must precede final journal
completion. Recovery payload/current-replica promotion durably commits first;
the idempotent main transition durably commits second; the recovery row is
durably marked complete last. A crash at either boundary leaves a pending row
for startup classification without weakening the exact-byte recovery invariant.

If projection/FTS update fails after disk and recovery payload commit, the
operation remains pending and the UI reports
`Saved to disk • search index updating` and queues reconciliation. If disk has
the intended bytes but the recovery completion transaction fails, the
pre-written intended revision still holds those exact bytes, the operation
remains pending/Attention, all further file commands fail closed, and the UI
reports `Saved to disk • recovery needs attention`. `Save failed` is reserved
for a result in which the intended bytes did not become canonical on disk.

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
- the idempotent main binding move commits while the recovery operation remains
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
3. decompress that snapshot and verify expected length and raw hash;
4. atomically rename the exact target to an operation-owned same-directory
   quarantine and durably flush that directory;
5. stable-read and hash the quarantined object:
   - if it matches the confirmed hash, verify the deletion revision again,
     unlink the quarantine, durably flush the directory, and confirm target and
     quarantine absence;
   - if it differs, retain those exact bytes as `conflict_disk` and attempt a
     no-replace restoration only while the original path remains absent; enter
     Attention regardless and never report deletion complete;
6. durably commit the main binding tombstone/projection transition;
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
The central invariant is testable without the main database:

> After every completed protected Chatbook save, the recovery database contains
> the exact current file bytes and can verify/list them using only its own rows.

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

The first release uses a fixed 1 GiB cap for live retained compressed payload
and requires at least 256 MiB of filesystem free space after the next reserved
operation. These are release constants, not user-facing settings. A configurable
quota is deferred until real usage demonstrates a need. The UI shows logical
retained bytes separately from physical `notes_recovery.db`, WAL, and SHM sizes,
because pruning does not promise immediate file shrinkage.

Before a Chatbook mutation or editable-draft admission, the coordinator
compresses or conservatively reserves the required payload, conflict headroom,
and a free-space margin. Current replicas, guaranteed deletion/conflict
retention, and unresolved operations are never silently evicted to make room.
The protection preview refuses a selection that cannot fit. Reaching the cap or
free-space floor later enters `Recovery capacity reached` and makes file-backed
commands read-only until automatic or explicit eligible-history pruning
succeeds. Stopping protection only makes its sealed checkpoint eligible for a
separate pruning decision; it does not itself free capacity. Guaranteed
deletion/conflict retention and unresolved drafts are never part of ordinary
pruning.

Every decompression is bounded by recorded codec/version, compressed length,
expected raw length, and maximum supported file size, then verified by raw hash.
Recovery-store corruption or failed validation preserves the suspect database
for diagnosis and makes every linked root read-only until the exact paired store
is repaired or restored. Destructive recovery-store reset is not part of the
core rollout. Database notes remain usable.

### Restore

Restore requires `read_write`, a healthy/capacious recovery store, a verified
payload, a matching root identity, safe containment, no symlink substitution,
an existing parent directory, and a destination confirmation bound to an exact
hash or expected absence. The coordinator creates a pending restore operation
before filesystem mutation and repeats the destination check at final publish.

- A missing destination may be restored in place.
- A differing existing destination defaults to a user-selected new path.
- Overwrite is a separate confirmation, round-trip verifies safety bytes, and
  uses atomic replace-with-displaced-target preservation; displaced bytes receive
  the 30-day resolved-conflict retention.
- Parent directories are never recreated implicitly.
- Restore-to-empty uses no-replace publication; all restore paths flush the
  written file and destination directory. Restored bytes and supported mode/
  security facts are re-read and verified before the main projection/binding
  transition. Restore-overwrite then performs common displaced-artifact cleanup
  after the 30-day recovery copy verifies and before the main transition; the
  recovery operation completes last.
- Explicit restore of a tombstoned note reuses its UUID, including when the user
  chooses another safe path and no live binding owns that UUID. `Restore as
  copy` creates a new UUID. Ordinary reuse of the old path without the restore
  command remains a new note.

Minimal restore for confirmed deletion ships before destructive Delete is
enabled. A broader History/Recovery browser can arrive later, but the product
must never promise `recoverable until <date>` without a working restore path.

### Independent recovery access

If the main database is unavailable, a minimal recovery-only screen opens
`notes_recovery.db` without joining the main projection. It can enumerate,
verify, and exact-export retained notes immediately. That read/export path is
part of the selective-recovery milestone. It performs no in-place file mutation,
does not create a deferred projection state, and does not require the main
database. Recovery-only in-place restore is deferred until there is a concrete
need and a separately reviewed destination-ownership design.

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

The existing local folder sync remains Database-note-only and is labeled
`Legacy folder sync (database notes)`. While a file root is active, an
overlapping pass is rejected and passive secondary processes show legacy
filesystem sync as paused; the exclusive lease-owning process may run a
non-overlapping pass through the in-process gate. Existing generic `New note`
creates a Database note. File creation is a separate explicit command at the
current disk-tree location.

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
and estimated indexing bytes. Milestone A offers `Link read-only` only. From the
B1 writable milestone onward, a fresh preview may offer `Link read/write` when
every safety gate passes; an existing read-only root uses the same preview as an
explicit `Upgrade to read/write` action. The B3 protection preview separately
estimates selected recovery bytes. Activation shows cancellable/resumable
progress without blocking Database Notes.

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
| Protection rules, per-note history/restore, independent list/verify/export | Not shipped | Not shipped except deletion restore | Enabled |

Before a capability ships, its controls are absent rather than decorative.
After it ships, a root/note that cannot use it shows the stable disabled action
with its exact blocking state and recovery path. Database-note actions retain
their existing availability independently.

### Wide layout

When the Notes canvas itself has at least 110 columns after shell chrome, it
shows a split workbench:

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

Search uses a dedicated navigator-search seam, not the capped existing
`search_notes` call:

- pages contain 100 metadata rows/snippets;
- input is debounced by 150 ms and tagged with a query generation so stale
  workers cannot replace newer results;
- `Load more` and PageDown fetch the next page; clearing search cancels pending
  publication and restores the prior tree/mode;
- no note bodies are loaded merely to render the tree;
- literal filename/path substring matching is combined with existing
  title/body FTS;
- exact filename ranks first, then filename prefix, path substring, FTS rank,
  then deterministic path/UUID tie-breakers;
- duplicate path/FTS matches collapse to one result;
- every result text-labels `File` or `Database`; file rows include root/relative
  path plus freshness/read-only state, never only a bare title;
- filename/path results remain available while body indexing is incomplete or
  a file is body-unreadable, but stale body matches/snippets are suppressed;
- selection is keyed by note UUID/result generation and remains stable while
  reconciliation updates unrelated rows;
- there is no fuzzy, regex, or semantic navigator search.

Initial body indexing is incremental, resumable, and visibly labeled. Frontmatter
is never indexed. Tree children are loaded in deterministic batches of at most
200, including a single directory with thousands of sibling files.

### Editor and status

The editor header has a two-line maximum. It always preserves the exact
root/path identity through middle truncation plus focus/help disclosure, shows
source authority, and shows only the highest-priority actionable state. Status
priority is Attention, then offline/read-only, then save state, then monitoring/
recovery. Lower-priority monitoring, protection, and capacity details live in
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
and combined search is relevance-ordered.

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

## Existing Workflow Parity

### Preview and copy/export

Preview renders the current body draft and never parses frontmatter.

File-backed actions distinguish exact source from draft:

- `Copy draft body`
- `Copy exact saved source` after validating the current disk raw hash
- `Export draft as .md/.txt`
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

A process-wide `LegacyRootOwnershipGate` is injected into
`NotesSyncEngine.sync()` and every direct legacy mutation entry point.
Library timers, `NotesSyncService`, and `AutoSyncManager` must route through that
engine-level gate.

The gate is backed by the same per-user cross-process root-mutation lease used by
the file coordinator. Before a file root owns the lease, every cooperative
current-version legacy pass acquires shared OS ownership before scanning and
holds it for its complete mutation lifetime. File-root activation atomically
closes new shared admission, waits for all current-process and other-process
shared holders, and acquires exclusive OS ownership for the active root's
lifetime. While that exclusive lease is held, secondary processes run no legacy
filesystem mutation. The lease-owning process may admit a non-overlapping legacy
pass under its already stronger exclusive ownership through the in-process
canonical-root gate; an overlapping pass is always rejected. A UI-only or
process-local overlap check is insufficient. Older tools that do not honor the
lease remain external writers under the stated support boundary.

## Runtime Modes and Concurrency

The file coordinator has three same-version modes:

| Mode | Behavior |
| --- | --- |
| `disabled` | Kill switch for activation, monitoring, and file commands. Existing projections are diagnostic and make no freshness claim. Startup still inspects incomplete operations without replaying them. |
| `read_only` | Link, inventory, tree/search, monitor, reconcile external changes, and inspect/export recovery when available. No file mutation or restore into a linked root. |
| `read_write` | Enables explicit coordinator mutations after all health, lease, and recovery gates pass. |

`disabled` is not a schema downgrade. A newly launched older binary still
rejects the migrated main-database schema version.

A mode-transition barrier rejects queued new saves, lets an already-started
operation finish or enter recovery, preserves newer editor buffers, and only
then changes mode. After `disabled`, each affected root completes identity and
hash reconciliation before it may claim freshness or return to `read_write`.
Applying any retained draft still requires a final fresh disk hash.

One coarse owner-only per-user OS root-mutation lease, stored in a fixed
application runtime namespace that is not derived from configurable user-data,
main-database, or repository paths, permits shared legacy passes only when no
file root owns it exclusively and permits one active file-notes root across
cooperative current Chatbook processes and storage instances. This intentionally
replaces a multi-profile durable root registry in the first release. Exclusive
ownership elects the process allowed to activate, monitor, reconcile/project,
or write the one root. Its diagnostic contents record a random owner nonce,
process-start identity, canonical root, and current `read_only` or `read_write`
capability; PID text is never ownership proof. Kernel release plus root-identity
and incomplete-operation checks governs stale recovery. Write permission is
gated separately.

A second Chatbook process is passive read-only: it may inspect cached projection/
recovery and exact-export a freshly hashed file, but it starts no second watcher/
reconciler, issues no file commands, and runs no legacy filesystem sync while
the exclusive lease is held. The core pilot permits one linked active root
total. The schema remains multi-root capable, but additional roots are a
separately gated expansion. Multiple hosts and network mounts are outside
write-safety guarantees.

### Upgrade and process support boundary

The core support contract is one current Chatbook installation, one configured
main-database storage instance, one active file root, and one cooperative file
coordinator. First activation tells the user to close other Chatbook processes
and refuses while a detected current legacy pass or exclusive root-mutation
lease is active.

The additive migration introduces only isolated file tables and does not rewrite
existing Database-note triggers or rows. An already-running pre-feature process
therefore has no API or table knowledge with which to mutate a file projection.
A newly launched older binary rejects the newer main schema. An old copy,
different profile, editor, or Git tool that writes the root is treated as an
external writer and is contained by the same raw-hash, displaced-target, and
reconciliation protocol.

Chatbook does not claim that it can enumerate or launcher-disable every arbitrary
source checkout or virtual environment. Durable multi-profile registration,
managed-launcher version floors, and simultaneous writable storage instances
are deferred product capabilities, not hidden prerequisites for Notes.

### Startup operation recovery

After `file_notes_storage` records a paired recovery store, every runtime mode,
including `disabled`, opens and identity-checks that recovery journal far enough
to inspect incomplete operations. Read-only A does not create or require a
recovery journal before that pairing exists. Startup takes the exclusive
root-mutation lease when available, examines current root identity/containment
plus source/destination paths and hashes, and never blindly replays a filesystem
mutation.

Classification is action-specific:

- save/create/restore: intended target plus the expected displaced/absent state
  means finish recovery metadata and projection; old target/expected absence
  means `not_applied` with draft retained; any unexpected target or displaced
  artifact means Attention with every stable side captured;
- delete: verified target absence plus absent quarantine and a valid deletion
  snapshot means complete the tombstone; the unchanged expected file means
  `not_applied` without unlinking; an operation-owned quarantine is classified
  and either safely finalized/restored or retained in Attention; a recreated/
  different target also means Attention;
- rename/move: absent source plus intended destination means complete the
  binding move; intended source plus absent destination means `not_applied`;
  both paths present, both absent, or unexpected hashes mean Attention.

Recovery metadata is repaired before main projection, and main projection is
then rebuilt idempotently from the observed disk result. If the expected
`notes_recovery.db` is absent, mismatched, or invalid, every linked root is
read-only, no restore is allowed, and no empty replacement is auto-created;
Database notes continue to operate.

## Migration and Activation

### Pre-migration safety

The first migration is additive and affects only the existing main database.
`notes_recovery.db` is new and independent, so there is no database pair to
stage, swap, or downgrade atomically.

A `ConfiguredMainDatabaseBootstrap` becomes the mandatory factory gate before
every normal or lazy constructor of the configured disk-backed main database.
Direct constructors require its schema-ready or explicit compatibility token;
in-memory/test databases may use a separate test path. The bootstrap holds a
storage-instance interprocess schema-maintenance lock in the same fixed per-user
runtime namespace, keyed by the canonical main-database path, so alternate
user-data settings cannot construct and auto-migrate the same database first.

1. Resolve the stable storage-instance namespace from the configured main
   database path and acquire the schema-maintenance lock.
2. Open a minimal SQLite bootstrap connection that cannot run migrations, read
   the version, and run the existing main-database integrity check.
3. Use SQLite's online backup API to create a verified pre-migration main
   snapshot in the owner-only storage-instance directory; fsync the snapshot and
   its directory before recording it as usable.
4. Run the transactional schema migration in place. It creates
   `file_notes_storage`, `note_file_roots`, `file_note_projections`, and
   `file_notes_fts`, advances the main schema version, and leaves existing
   `notes`, keyword, relation, FTS, and sync triggers unchanged.
5. Reopen/verify the main schema and integrity, publish a schema-ready token,
   then allow ordinary main-database constructors and file-source activation.

The migration never renames or replaces a live SQLite database file and never
manipulates its WAL/SHM/rollback-journal sidecars. SQLite owns concurrency and
rollback for the in-place transaction.

If integrity is healthy but the required backup/preflight fails, the bootstrap
does not attempt migration. The new binary may open only the exact immediately
preceding schema through an explicit `database_only_compatibility` token that
disables all file-table/workbench access and leaves Database Notes usable.
A rolled-back additive migration may use the same mode after integrity is
rechecked. Other old, malformed, or newer schema versions still fail normally;
compatibility mode is not an automatic downgrade.

The recovery database is created and validated independently in the same
storage-instance namespace before B1 first enables a write command.
`RecoveryStoreBootstrap` runs under the exclusive root-mutation lease and is
permitted only when `file_notes_storage` proves no recovery instance has ever
been paired; a recoverable owner-only marker coordinates creation of
`recovery_store_identity` and commitment of the expected UUID in the main row.
After pairing, missing/mismatched storage is an Attention/read-only condition,
never permission to initialize a fresh database. Recovery bootstrap failure does
not roll back the additive main schema, block Database Notes, or block the
read-only A milestone. It simply keeps all file mutations and protection
disabled until the exact store is healthy. Later recovery-schema migrations and
explicit store relocation require their own backup/design when they exist;
speculative pair-migration machinery is not part of this rollout.

The verified pre-migration main backup is outside the 1 GiB recovery capacity
and retained for manual offline recovery until explicit administrative pruning.
Automatic downgrade is not supported.

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

Activation is crash-resumable:

1. Close new cooperative legacy-pass admission, await current shared holders,
   and acquire the per-user root-mutation lease exclusively.
2. Canonicalize/fingerprint the root, reject a second active root and configured
   legacy-sync overlap, and establish the requested runtime capability.
3. Revalidate root identity and the in-process legacy ownership gate.
4. Persist a durable root record in `activating` while retaining the lease.
5. Start bounded watcher/event capture.
6. Inventory path metadata and classification.
7. Drain captured events.
8. If capture overflowed or root identity changed, discard the inventory and
   rescan.
9. Atomically publish the root and file projections.

A crash leaves an explicit resume/discard activation state. Path navigation
becomes available from the published metadata while body FTS proceeds
incrementally. Source files remain unchanged.

From B3 onward, selected protected items show `Protection pending`; each becomes
writable after its baseline replica verifies. In B1/B2, and for unprotected
items in B3, items become writable after root activation and per-operation
recovery health gates pass.

## Unlink, Forget, and Retained History

### Unlink

Unlink crosses an operation barrier, stops monitoring, and releases the lease.
It never touches source files or Git. The root becomes detached while retaining
UUIDs, metadata, drafts/conflicts, and recovery bytes; those bytes still count
toward the fixed capacity. A detached root does not reserve a global path after
the exclusive root-mutation lease is released.

Relink requires explicit confirmation plus root identity and inventory checks.
A recreated directory at the same path does not silently inherit old UUIDs.

### Forget

Forget removes the root and file projections. It does not touch source files.
Forget is blocked while pending/Attention operations or unresolved drafts exist;
those must be resolved or separately discarded first. Self-contained deletion
snapshots remain through their guaranteed expiry. Ordinary current replicas and
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

Application-consistent backup of the main and recovery databases is a later
optional feature. Source files and Git would remain outside such a backup, so it
would not replace the user's repository or an off-device backup.

## Security and Privacy

- Main/recovery databases, WAL/SHM files, backups, lock metadata, and temporary
  recovery artifacts use owner-only app-directory permissions (POSIX 0700
  directories and 0600 files where supported, equivalent owner-only policy
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
- Recovery content is unencrypted plaintext SQLite. Same-user processes, device
  backups, and anyone with access to an unencrypted disk may read it.
- Full-disk encryption and an independent off-device backup are recommended.
- The Git remote protects only content that the user has committed and pushed.

## Performance and Non-Degradation Gates

- With no linked root, Chatbook starts no file watcher, reconciliation worker,
  recovery writer, lease, or file scan.
- Apart from the one-time additive isolated-table migration and specified
  Link/Recovery actions, existing Database Notes behavior and the zero-root
  steady-state startup path remain unchanged.
- A cached 5,000-file root reaches interactive tree navigation without reading
  file bodies before first paint.
- Tree rows carry metadata only and load children lazily in bounded batches,
  including a 5,000-sibling directory.
- Navigator search returns its first 100-result page within 200 ms p95 on the
  fixed benchmark fixture, after warm-up and over at least 30 samples; canceled
  stale queries never publish over a newer generation.
- A settled local edit reaches projection/FTS within 2 seconds p95 on the fixed
  benchmark runner.
- No individual file-notes UI-thread callback measures 100 ms or more in the
  benchmark trace.
- Polling fallback over an unchanged 5,000-file fixture averages at most 5% of
  one CPU core over 60 seconds, with a bounded, non-growing event queue.
- Full root scans, RAG, initial body indexing, compression, hashing, and SQLite
  backup are off the startup/UI critical paths.
- Resize preserves state and never creates two nested editor scroll owners.

The benchmark manifest pins runner class, OS/filesystem, Python/SQLite/Textual
versions, both deep-tree and 5,000-sibling generated fixtures, warm-up, sample
count, and timing boundaries. Shared CI verifies deterministic behavior; the
pinned runner owns hard timing gates.

## Verification Strategy

### Focused per-PR tests

- Structural isolation: existing Database-note schema/triggers and CRUD behavior
  remain unchanged, and file projections cannot leak into Database-note counts,
  search, export, RAG, MCP, keyword, relation, or Sync paths.
- Projection idempotency by root, normalized path, raw hash, and presence state.
- Storage-instance namespacing, recovery-instance pairing/mismatch, no silent
  rebootstrap after path/config relocation, and one coarse active-root lease.
- Read-only capability on macOS/Linux/Windows plus APFS-specific writable
  capability probing and fail-closed downgrade elsewhere.
- Exact frontmatter/BOM/newline/final-newline/mode preservation, raw-hash-bound
  normalization acknowledgement, and deterministic new-file byte defaults.
- Strict encoding, size, malformed-frontmatter, hardlink, symlink, unreadable,
  unsafe-name/control rendering, nested-mount/cross-device mutation refusal, and
  special-file classification.
- Versioned hash vectors plus rejection of unknown hash versions.
- Save no-op behavior and editor-generation races.
- Every mutation outcome at each journal checkpoint, destination-path locking,
  no-replace publication, displaced-target/exchange classification,
  delete-quarantine recovery, and operation-linked artifact cleanup.
- Recovery/full-synchronous commit barriers, post-commit safety-blob
  round-trip verification, per-action file/directory fsync, normal complete-last
  ordering, and completed-operation pruning.
- Expected-hash conflict, conflict-headroom failure, overwrite/discard safety,
  resolved-side retention, exact-export fallback, and durable Attention.
- Missing/root-Offline/tombstone transitions, atomic-save versus genuine move,
  manual reassociation, and path-reuse identity.
- Delete snapshot verification, 30-day pin, stale confirmation, quarantine, and
  minimal restore.
- Protection inheritance/overrides, current replica invariant, fixed capacity,
  free-space floor, corrupt
  revision, bounded decompression, payload/operation garbage collection, and
  recovery failure.
- Search rank/paging/dedup/latest-query cancellation/high-fanout batching,
  stale-body suppression/path-only fallback, and durable incremental file-FTS
  progress.
- Cross-process shared/exclusive legacy ownership plus engine-level canonical
  root tokens and Database/file repository boundaries.
- Narrow/wide focus, resize, autosave, two-line status priority, root/protection
  flows, conflict, preview, exact export, and explicit Console handoff
  truncation.
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
- Recovery COMMIT/fsync failures before file mutation, main-commit/recovery-
  complete ordering, deletion retention, and revision-GC reference swap.
- Real watcher timing, duplicate/move/overflow storms, polling fallback, and Git
  checkout/bulk/conflict-marker reconciliation.
- Deep and 5,000-sibling scale plus fixed-runner p95 benchmarks.
- Two-process shared-legacy/exclusive-file lease contention, secondary-worker
  refusal, PID reuse diagnostics, and stale-owner recovery.
- Running legacy-sync shared-token drain and overlap fencing at every engine
  entry point and across cooperative processes.
- Root identity change and same-path directory recreation.
- Mandatory pre-constructor ordering across every configured-main-DB entry
  point; main-backup failure and additive-migration rollback enter verified
  Database-only previous-schema compatibility; recovery bootstrap
  crash/mismatch/failure never creates an empty replacement; and older-binary
  rejection remains documented.
- Mode change during mutation, recovery corruption/full disk/capacity, unlink,
  relink, forget, eligible-history pruning, and occupied-destination restore.
- APFS native exchange/exclusive/no-follow and durability tests on macOS;
  packaged Linux/Windows tests prove the same roots remain explicitly read-only.

Process crash tests do not claim to simulate power loss. Commit and directory-
fsync ordering is fault-injected, while each supported writable
platform/filesystem durability profile also receives a release power-cut/reboot
harness or documented equivalent validation. Network/cloud mounts remain
outside release guarantees.

The Git smoke matrix is intentionally small: status/diff, one local commit,
checkout/bulk reconciliation, and conflict markers. Push remains ordinary Git
behavior outside Chatbook and does not justify a remote test matrix here.

## Rollout

Delivery is phased so the realistic read-only A arrives before the desired
writable B:

1. **A: isolated read-only file source**
   - verified main backup and additive isolated projection/FTS migration;
   - declared watcher dependency, preview/activation/resume, one active root,
     external reconciliation, unlink/relink;
   - separately mounted workbench, scalable tree/search, preview, exact
     copy/export, and Console handoff;
   - exact zero-root Database Notes behavior apart from the specified
     Link/Recovery actions;
   - no file mutation and no recovery DB prerequisite.
2. **B1: journaled macOS/APFS editing**
   - independently paired recovery DB bootstrap and fixed-capacity admission;
   - blank create in an existing directory, body save, rename, and move;
   - debounced autosave, mandatory operation safety, atomic displaced-target
     preservation, durable conflicts, and startup classification;
   - no Delete until B2.
3. **B2: confirmed delete and minimal restore**
   - verified deletion snapshot, quarantine protocol, 30-day guarantee, and
     working per-note restore ship together.
4. **B3: selected file/folder recovery**
   - protection rules/current replicas, coalesced checkpoints, per-note
     history/restore, fixed capacity controls, and independent
     enumerate/verify/export.
5. **Deferred expansions**
   - Linux/Windows writable adapters, additional roots/storage instances,
     multi-profile durable registry, recovery-store relocation/clone, folder
     mutation, file templates,
     file-specific keywords/links, file MCP/RAG, configurable quotas, guarantee
     waiver/general purge, database-pair backup/restore, recovery-only in-place
     restore, Git controls, and remote/portable file identity or synchronization.

TASK-399 is a roll-up tracker and is never implemented as one PR. Implementation
planning must create atomic child tasks in this dependency order, each with its
own acceptance criteria and ADR link. No milestone may expose an action whose
recovery and fail-closed prerequisites belong only to a later milestone.

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

### Store UUIDs in frontmatter or a repository manifest

Rejected. It would alter user files solely for Chatbook, create Git noise, and
make Chatbook metadata part of repositories that currently contain only notes.

### Store recovery bytes in the main database

Rejected. A separate recovery database better satisfies independent enumeration
and recovery when the main projection is unavailable, and it isolates recovery
capacity/corruption from Database-note availability.

### Ship writable macOS, Linux, and Windows support together

Rejected for the first writable release. Their beneath-root, atomic replacement,
permission, and durability primitives are not equivalent through Python's
portable APIs. APFS is the first tested write target; the same feature remains
useful read-only elsewhere while later native adapters prove equal safety.

### Stage and atomically swap a main/recovery database pair

Rejected for the first migration. The recovery database is new, file tables are
additive and isolated, and SQLite already provides transactional schema changes
plus a consistent online backup API. Renaming live database families adds handle,
sidecar, cross-volume, downgrade, and launcher coordination without protecting a
state that exists yet.

### Add Git controls with the first file UI

Rejected for this tranche. Direct file correctness, conflicts, deletion, and
recovery must be proven before Chatbook adds another stateful workflow around
staging or remotes.

## Governance

ADR required: yes
ADR path: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`
Reason: This design establishes long-lived content authority, schema,
migration, recovery, conflict, sync-isolation, filesystem, and Library UX
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
