# File-Backed Notes Authority and Recovery Design

Date: 2026-07-22
Status: User-approved design, pending written-spec review
Backlog: [TASK-399](../../../backlog/tasks/task-399%20-%20File-backed-Notes-disk-authoritative-Library-management-and-local-recovery-replica.md)
ADR: [ADR-021](../../../backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md)

## Summary

Chatbook will make existing Git-managed Markdown and text folders first-class
Notes sources inside Library. The bytes on disk remain authoritative. Editing a
file-backed note in Chatbook changes the actual file, so the user's existing
`git status` / `git add` / `git commit` / `git push` workflow continues without
translation, export, or a second synchronization step.

SQLite has two supporting roles, neither of which competes with the files:

1. The main Chatbook database holds a searchable, derived projection and durable
   local metadata.
2. A separate `notes_recovery.db` holds the mutation journal, mandatory
   operation-safety copies, and opt-in recovery replicas/history for selected
   files and folders.

The recovery database is independent of the main notes projection and contains
enough information to list and restore protected notes if the main database is
unavailable. It is still an unencrypted, same-device recovery copy. It is not an
off-device or disaster backup.

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
- Keep stable local UUIDs for Chatbook metadata, links, RAG, Console handoff, and
  recovery continuity across known moves.
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
| **Database note** | Existing `notes` content whose `storage_kind` is `database`; SQLite owns its title and body. |
| **File-backed note** | A supported regular file under a linked root whose `storage_kind` is `file`; disk owns its path, title, and bytes. |
| **Projection** | The main-database title/body/search representation derived from a file. |
| **Binding** | Durable association between a local note UUID and a root-relative path. |
| **Protected note** | A file selected by a folder-prefix rule or file override for a continuously maintained recovery replica. |
| **Current replica** | The one verified, exact current-byte copy retained for a protected note. |
| **Checkpoint** | A sealed, coalesced recovery revision retained by editing session rather than by autosave. |
| **Safety snapshot** | A temporary exact-byte copy required to make one Chatbook mutation recoverable. |
| **Deletion snapshot** | A verified exact-byte copy pinned for the confirmed-deletion guarantee. |
| **Attention** | A durable unresolved operation holding a draft and the observed disk state for user resolution. |
| **App session** | One Chatbook process lifetime, identified by a new random UUID at startup. It is the boundary for `Changed this session`. |
| **Editing session** | One continuous open/edit interval for a note. For protected notes, autosaves within it update the current replica but do not each create a checkpoint. |

## Authority and Identity

### Disk owns content

For a file-backed note:

- the file's exact bytes are canonical;
- the exact root-relative path is its canonical locator;
- the exact basename, including extension, is its displayed and projected title;
- the main database stores the decoded editable body only;
- frontmatter is not included in the projection, FTS, RAG, or Console body;
- filesystem mtime is observation metadata, never a conflict winner or logical
  version.

The main database may temporarily lag disk, but it may never overwrite disk merely
because its projection is newer by timestamp.

### SQLite owns local identity and metadata

Every file-backed note receives a random local UUID in SQLite. The UUID is used
for local keywords, links, Console references, RAG invalidation, operation
history, and recovery. It is not written into frontmatter, a sidecar, or a
repository manifest.

The UUID is preserved automatically only when identity evidence is explicit:

- a Chatbook move or rename completed by the mutation coordinator; or
- a paired move event delivered by the watcher and verified by reconciliation.

A missed or ambiguous external move is represented as one missing binding and
one newly discovered file with a new UUID. Chatbook does not guess identity from
inode reuse, content hash, or filename similarity. A fresh Chatbook database or
clone also assigns new UUIDs.

### Two hashes, two purposes

- `raw_hash` hashes the complete file bytes. It protects mutations, exact export,
  delete, restore, and recovery verification.
- `semantic_hash` hashes the decoded editable body plus the filename-derived
  title. It decides whether FTS and optional RAG work is necessary.

A frontmatter-only or line-ending-only change can increment the local projection
generation while leaving the semantic hash unchanged and avoiding an FTS/RAG
rebuild.

## Supported File Contract

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
a warning before its first Chatbook write. If the user proceeds, the body is
normalized to the count-dominant line ending (the first encountered wins a tie);
the exact prior bytes must already be recoverable. ACLs, extended attributes,
ownership, Windows alternate streams, and network-filesystem atomicity are not
promised.

## Data Model

The design adds two narrow ownership columns and four logical tables. Exact DDL
is assigned in the implementation plan, but these ownership and state fields
are required.

### Main Chatbook database

#### `notes.storage_kind`

`notes.storage_kind` is non-null and has values `database` or `file`. Existing
rows migrate to `database`. The value is immutable after insertion.

A file projection must be inserted with `storage_kind = file` in its first
transaction; no trigger or service may briefly treat it as a Database note.

#### `keywords.sync_eligible`

`keywords.sync_eligible` is non-null boolean metadata. Existing and
Database-note-created keywords default true. A keyword first created from a
file-backed note starts false, so its definition remains local. Explicit reuse
from a Database-note workflow atomically promotes it to true and emits the
required sync upsert once. Promotion is one-way in this tranche.

#### `note_file_roots`

Each root stores:

- root UUID and user-facing label;
- local canonical path and a platform filesystem identity/fingerprint;
- runtime mode and lifecycle state;
- exact-path comparison policy/capability;
- protection-prefix configuration;
- activation, scan, and observation generations;
- timestamps and diagnostic state.

Roots with equal, parent, or child path overlap are rejected. An overlap with a
legacy Database-note sync root is also rejected.

#### `note_file_bindings`

Each binding stores:

- note UUID and root UUID;
- exact relative display path plus a filesystem-aware comparison key;
- raw and semantic hashes;
- presence state;
- observed size, mtime, mode bits, BOM/newline/final-newline facts;
- frontmatter mode and format/read-only diagnostic;
- local projection generation and last observation time;
- body-index state, indexed semantic hash/generation, and path-metadata
  generation.

The filesystem-aware comparison key is derived per root. If Chatbook cannot
determine a safe comparison policy or detects normalized collisions, that root
cannot enter `read_write`. The exact display path is never replaced by the
comparison key.

Binding state, not `notes.deleted`, is authoritative:

| State | Meaning and projection behavior |
| --- | --- |
| `present` | File is observed and eligible for normal projection/search. |
| `missing` | This path is absent while the root is online. Hide it from Browse and search results, but retain the last projected body for diagnosis/recovery. |
| `tombstoned` | Chatbook confirmed deletion. Clear the projected body after the deletion snapshot is committed and unlink is verified. |

`notes.deleted` remains a compatibility visibility flag and follows binding
state; it does not define file authority. Reappearance at a `missing` path
retains the UUID. Reuse of a `tombstoned` path creates a new UUID unless the user
explicitly restores/reuses the tombstone.

Offline is a root lifecycle/effective UI state, not a mass binding transition.
When a root is Offline, its bindings keep their last `present`, `missing`, or
`tombstoned` state and `notes.deleted` value. Browse may show their cached
projection under an Offline banner, but no result claims freshness.

Projection is idempotent on root, filesystem-aware path key, raw hash, and
presence state. `notes.version` is a monotonic local projection generation, not
an mtime-derived value. `notes.last_modified` records the Chatbook mutation or
observation time; the filesystem mtime remains separate binding metadata.
Each root also persists initial-body-index progress so indexing can resume after
restart without making path search wait.

### Independent `notes_recovery.db`

The recovery database has its own schema version, integrity checks, and
owner-only files. It does not depend on joining the main database to enumerate
or verify retained content.

#### `note_file_revisions`

Every retained row is self-contained enough for recovery and includes:

- note UUID;
- root label and identity fingerprint;
- exact relative path;
- revision kind;
- raw hash and raw byte length;
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

`main_projection_state` is independently `pending`, `applied`, or
`deferred_rebuild` (the last is only for the explicit recovery-only emergency
flow).

Failure or `not_applied` is an outcome, not another state taxonomy. A completed
operation may still reference a retained recovered draft. Only successfully
committed working-tree-changing operations from the current app-session UUID
appear in `Changed this session`; no-op saves and external changes do not.

## Service Boundaries

### File mutation coordinator

One process-wide coordinator is the only component allowed to mutate a linked
file or project a file row internally. It provides explicit commands such as:

- save body with expected path/raw hash and editor generation;
- rename/move with expected source and destination absence/hash;
- create at an explicit supported path whose parent already exists;
- confirmed delete with an exact confirmation token;
- restore from a verified recovery revision;
- reconcile one path or one root.

It serializes commands per note and runs at most one reconciliation flight per
root. Filesystem work, hashing, compression, SQLite work, and scans run outside
the Textual UI thread.

### Existing Notes services

`NotesScopeService` remains the public routing boundary:

- Database-note calls retain current behavior.
- File-backed reads can use the projection.
- Generic save/delete calls on file rows fail closed and direct callers to an
  explicit file command.
- Only an internal projection method may update file-backed `notes` rows.

`NotesInteropService` remains a Database-note compatibility adapter and gains no
filesystem authority. `storage_kind` is not auto-converted in either direction.

## Mutation Protocol

SQLite and the filesystem cannot form one transaction. Every Chatbook mutation
therefore uses a durable intent journal plus hashes.

### Common mutation invariant

Before any filesystem mutation, the coordinator must:

1. Acquire the note lock plus normalized source/destination path locks in
   deterministic order. Bulk directory commands serialize at the root mutation
   boundary.
2. Revalidate root identity, mode, lease, containment, parent, file type,
   expected source hash/state, and expected destination hash/absence.
3. Commit one `pending` operation and references to the intended state plus any
   bytes whose only accessible copy could be displaced.
4. Perform the filesystem action using collision-safe no-replace semantics for
   create, move, and restore-to-empty. If the platform cannot provide the
   required no-replace primitive, that command fails closed.
5. Re-read/re-stat all affected paths and classify the observed result before
   applying the idempotent main binding/projection transition.
6. Keep the recovery operation `pending` until that main transition commits,
   then mark the recovery operation complete last.

The final external-writer race cannot be eliminated for replace, move, or
unlink on every supported filesystem. Path locks prevent competing Chatbook
commands, while immediate precondition checks, no-replace primitives where
available, post-action verification, and retained safety bytes bound the
remaining risk. Network/cloud filesystems and uncooperative multi-process
writing remain outside guarantees.

Same-directory temporary names contain the operation UUID and are recorded in
the journal. Startup removes a leftover temporary only after proving it belongs
to that exact operation and is not a current source/destination. Chatbook never
deletes temporaries by wildcard. A crash-left temporary may be Git-visible until
startup reconciliation removes or surfaces it.

### Save

For a non-no-op save:

1. Capture note UUID, path, base hash, body snapshot, and editor buffer
   generation.
2. Acquire the note serialization lock and verify root mode, lease, containment,
   parent, regular-file status, and recovery health/capacity.
3. Read and raw-hash disk immediately before mutation. If it differs from the
   expected base, atomically persist durable Attention as described below
   instead of writing.
4. Construct intended bytes from the preserved BOM/frontmatter/byte-style facts
   plus the editor body.
5. In one recovery-database transaction, persist a `pending` operation, the
   intended draft, and exact old safety bytes (or a verified reference to an
   already sufficient current replica).
6. Write intended bytes to a uniquely created temporary in the same directory,
   preserve supported mode bits, flush/fsync the file where supported, atomically
   replace the target, then fsync the parent directory where supported.
7. Re-read the target and hash it:
   - intended hash: in a recovery transaction, promote/verify the intended
     revision as the protected current replica when applicable and record the
     filesystem outcome while keeping the operation pending; then project the
     observed file and mark the operation complete last;
   - old hash: record `not_applied`, retain the draft, and leave the editor dirty;
   - any other hash: enter Attention as `Disk changed during save`.
8. Clear the editor's dirty state only when its current generation still equals
   the saved generation. Typing that occurred during I/O stays `Draft`.

If the intended raw hash already equals disk, Chatbook performs no file rewrite
and creates no Git noise. It may refresh stale projection metadata, but the
action does not appear as a session change.

Main-database projection is repairable but must precede final journal
completion. Recovery payload/current-replica promotion precedes main projection;
the idempotent main transition follows; the recovery row is marked complete
last. A crash at either boundary leaves a pending row for startup classification
without weakening the exact-byte recovery invariant.

If projection/FTS update fails after disk and recovery payload commit, the
operation remains pending and the UI reports
`Saved to disk • search index updating` and queues reconciliation. If disk has
the intended bytes but the recovery completion transaction fails, the
pre-written intended revision still holds those exact bytes, the operation
remains pending/Attention, all further file commands fail closed, and the UI
reports `Saved to disk • recovery needs attention`. `Save failed` is reserved
for a result in which the intended bytes did not become canonical on disk.

The post-replace re-read detects the remaining external-writer race and retains
both the old safety bytes and intended draft; power-loss durability remains
filesystem-dependent.

Leaving the editor, changing the selected note, switching Library mode, or
navigating away runs the same save/attention flush before the logical draft is
released. A failed flush does not discard the buffer; it becomes a recoverable
draft/Attention item and navigation may continue.

### Autosave and checkpoints

File-backed notes keep the existing two-second debounce. A 30-second maximum
interval from the first dirty keystroke prevents continuous typing from
postponing all persistence indefinitely. `Save now` bypasses both timers.

Autosave writes the canonical file, but it does not seal a checkpoint on every
write:

- for a protected note, all autosaves update the one current replica;
- an identical raw hash creates no revision;
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
is rejected when a changed file is observed at the final precondition check;
the documented external-writer race after that check remains outside the
guarantee.

The coordinator must:

1. revalidate mode, root identity, containment, regular-file status, and hash;
2. commit one `pending` delete operation and its exact deletion snapshot in the
   same recovery transaction;
3. decompress that snapshot and verify expected length and raw hash;
4. unlink the exact file;
5. fsync the containing directory where supported and confirm absence;
6. commit the main binding tombstone/projection transition;
7. complete the recovery operation last.

The exact snapshot is guaranteed for 30 days for Chatbook-initiated deletion of
a supported regular file. This guarantee applies even when continuous
protection is off. Delete is unavailable if Chatbook cannot reserve and verify
that retention. An externally observed deletion can guarantee only the last
bytes Chatbook had already captured.

The completion toast is explicit:

```text
Deleted from disk • recoverable until <date>
```

Folder deletion is a separate bulk preflight. It enumerates supported regular
files and performs the individual protocol for each. Symlinks, special files,
hardlinks, and unsupported files are retained; only directories left empty are
removed. There is no recursive blind delete.

## Reconciliation and External Changes

### Watcher events are hints

`watchdog` is the supported near-real-time event backend in packaged builds. If
the observer cannot run, Chatbook uses a visible one-second metadata-polling
fallback labeled `Monitoring via polling`. Both feed the same thread-safe,
path-keyed bounded accumulator:

- duplicate events coalesce;
- a paired move enqueues source and destination;
- overflow collapses to one root-rescan sentinel;
- `.git/` and Chatbook temporary paths are excluded;
- no event is trusted without inspecting current filesystem state and hashes.

There is no time-window suppression for self-generated events. A watcher event
whose observed raw hash already matches the committed binding is simply a no-op.

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
proves content equality. An externally changing file is accepted only when its
pre/post-read stat facts match; otherwise it is retried as unsettled. Git
checkout/merge/rebase storms are coalesced into bounded root work. They never
populate `Changed this session`.

### Root and path outcomes

- If the root is absent or its filesystem identity changes, mark the root
  Offline and do not mass-mark files missing.
- If one path disappears while the root is online, mark it Missing.
- If a missing path reappears, retain its UUID and reconcile its bytes.
- A reliable paired watcher move preserves identity after verification.
- A missed/ambiguous move produces Missing plus a new binding, without guessing.

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

Creating Attention is one recovery transaction: a pending operation transitions
to `attention` while pinning the exact draft, the editor base (or an already
verified equivalent revision), and the latest stable observed disk bytes.
Returning from the save/reconcile worker is not allowed until that transaction
commits. Later external changes replace the pinned `conflict_disk` side only
after the new bytes verify; the base and draft are not lost.

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
- Overwrite requires confirmation, protects the freshly observed disk bytes,
  and composes the draft body with the freshly observed opaque
  frontmatter/BOM/body-byte style. If that fresh prefix/style is unsupported or
  ambiguous, overwrite fails closed and offers Save-as-new/export.
- Discard requires confirmation, retains the latest draft as
  `recovered_draft`, loads the current disk body, and completes Attention.

Neither action can erase the only copy of displaced content. There is no
automatic merge, timestamp winner, or navigation lock; unresolved items remain
in `Needs attention`.

Target observation for a settled local edit is projection and FTS visibility
within 2 seconds p95 on the fixed benchmark runner. Optional RAG may lag.

## Recovery Replica, Retention, and Quota

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

### Retention

- Current replica: retained while protection is enabled.
- Sealed checkpoints: retain at most 50 per note and none older than 30 days;
  both limits apply.
- Confirmed deletion bytes: pinned for at least 30 days.
- Tombstone metadata: retained beyond payload expiry until explicit purge or
  root-forget policy permits removal.
- Unresolved drafts/conflicts and incomplete operations: pinned until resolved
  or explicitly discarded.
- Completed emergency operations with `main_projection_state =
  deferred_rebuild`, plus their required revisions: pinned until main rebuild
  consumes them and marks the projection applied.
- Temporary safety revisions: eligible for pruning after the operation completes
  and all stronger retention needs are satisfied.
- Completed, non-Attention unprotected autosaves coalesce their payloads to at
  most one latest verified disk/base revision for each still-dirty open note.
  Each later completed save supersedes/prunes the prior one. That final base
  becomes immediately garbage-collection eligible when the editor becomes
  clean/closes unless an incomplete operation or Attention references it. A
  protected equivalent is eligible only after its current replica/checkpoint
  satisfies the stronger retention rule.

Completed operation metadata is bounded independently of payload retention. The
current app session coalesces successful completed operations to at most the
latest row per affected note UUID (or path when no UUID exists), so repeated
autosaves cannot grow it without bound while `Changed this session` stays
complete. Inactive-session completed rows are retained for at most seven days
and capped at 10,000, oldest first. Pending/Attention and `deferred_rebuild`
rows are never removed by this garbage collection.

### Quota

The default quota is 1 GiB of live retained compressed payload. The UI shows
that logical amount separately from physical `notes_recovery.db`, WAL, and SHM
sizes, because pruning does not promise immediate file shrinkage.

Before a Chatbook mutation, the coordinator compresses or conservatively
reserves the required payload plus a free-space margin. Current replicas,
guaranteed deletion snapshots, and unresolved operations are never silently
evicted to make room. Lowering the quota below pinned content enters
`Recovery over quota` and makes file-backed commands read-only until the user
raises the quota, exports/purges eligible history, or explicitly changes
protection.

Every decompression is bounded by recorded codec/version, compressed length,
expected raw length, and maximum supported file size, then verified by raw hash.
Recovery-store corruption or failed validation preserves the suspect database
for diagnosis and makes every linked root read-only until the store is repaired
or replaced. Database notes remain usable.

### Restore

Restore requires `read_write`, a healthy/capacious recovery store, a verified
payload, a matching root identity, safe containment, no symlink substitution,
an existing parent directory, and a destination confirmation bound to an exact
hash or expected absence. The coordinator creates a pending restore operation
before filesystem mutation and repeats the destination check at final publish.

- A missing destination may be restored in place.
- A differing existing destination defaults to a user-selected new path.
- Overwrite is a separate confirmation and first protects displaced bytes.
- Parent directories are never recreated implicitly.
- Restored bytes and supported POSIX mode bits are re-read and verified before
  the main projection/binding transition; the recovery operation completes
  last.
- Explicit restore of a tombstoned note reuses its UUID, including when the user
  chooses another safe path and no live binding owns that UUID. `Restore as
  copy` creates a new UUID. Ordinary reuse of the old path without the restore
  command remains a new note.

Minimal restore for confirmed deletion ships before destructive Delete is
enabled. A broader History/Recovery browser can arrive later, but the product
must never promise `recoverable until <date>` without a working restore path.

### Recovery-only emergency flow

If the main database is unavailable, a dedicated recovery-only screen opens
`notes_recovery.db` without joining the main projection. It can enumerate,
verify, and export retained notes immediately.

An in-place recovery-only restore is never unattended: the user selects the
destination root, Chatbook verifies it against the stored root fingerprint,
acquires its coordinator lease, enters an explicit recovery `read_write` mode,
and runs the same journal/CAS protocol. A nonmatching root is export/restore-as-
new only. The operation remains self-contained in the recovery database; a
repaired or newly created main database later rebuilds projection/bindings from
the restored files. This emergency operation completes with
`main_projection_state = deferred_rebuild`, and main rebuild explicitly consumes
those rows. Absolute root paths are not required in recovery payloads.

## Library Notes User Experience

### Information architecture

File-backed notes stay in the existing Local Notes owner inside Library. They do
not become a new top-level route or scope.

Browse mode has two explicit groups:

- `Files on disk`
- `Database notes`

The existing local folder sync remains available only for Database notes and is
labeled `Legacy folder sync (database notes)`. Existing generic `New note`
creates a Database note. File creation is a separate explicit command at the
current disk-tree location.

The navigator has three modes:

- `Browse`
- `Changed this session (N)`
- `Needs attention (N)`

`Changed this session` contains successful working-tree-changing Chatbook
operations for the current process-lifetime session: create, content save,
rename, move, delete, and restore. No-op saves and externally initiated changes
are excluded. `Needs attention` is durable across restarts.

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
- no note bodies are loaded merely to render the tree;
- literal filename/path substring matching is combined with existing
  title/body FTS;
- exact filename ranks first, then filename prefix, path substring, FTS rank,
  then deterministic path/UUID tie-breakers;
- duplicate path/FTS matches collapse to one result;
- filename/path results remain available while body indexing is incomplete or
  a file is body-unreadable;
- there is no fuzzy, regex, or semantic navigator search.

Initial body indexing is incremental, resumable, and visibly labeled. Frontmatter
is never indexed.

### Editor and status

The editor header shows an exact root/path breadcrumb. File-backed notes do not
show an independently editable title field; rename is a file action.

Primary visible actions:

- `Save now`
- `Preview` / `Edit`
- `Use in Console`

Secondary labeled `Actions`:

- Export
- Copy
- Move
- Rename
- History

Delete remains visually separate.

Save status vocabulary is exact:

- `Draft`
- `Recoverable draft`
- `Saving`
- `Saved to disk`
- `Saved to disk • search index updating`
- `Saved to disk • recovery needs attention`
- `Disk changed while editing`
- `Save failed • draft retained`

Recovery UI is called `Chatbook recovery history`, not `backup` or `protected
history`.

The existing word count remains visible. Existing keyword and local-link
editing remains available for both note kinds; those values are SQLite-owned
metadata and never rewrite a file. Database-note sort options keep their current
behavior, while file browsing is path-ordered and combined search is
relevance-ordered.

### Keyboard and focus contract

- `F6` / `Shift+F6` cycle major available Notes regions per the existing
  Workbench convention.
- `Ctrl+S` invokes `Save now` for the open editable note.
- `/` focuses navigator search only when a text editor/input is not focused.
- `Enter` opens the selected navigator item.
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

- `Copy body`
- `Copy draft as Markdown` (not exact source)
- `Export draft as .md/.txt`
- `Export exact saved source` after validating current disk raw hash

When a conflict is active, draft export names include a `draft` prefix and disk
export names include a `disk` prefix so they cannot be mistaken for resolution.

Database-note exports retain current normalized/frontmatter-injecting behavior.
File-backed exact export never injects `note_id`, frontmatter, or metadata.

Mixed bulk export namespaces file entries by root ID/label, exports exact saved
bytes, exports Database notes in their existing normalized format, and includes
a metadata manifest plus missing/conflict report.

### Templates and import

Creating from a template while browsing `Files on disk` proposes a safe filename
under the current directory and stops on collision. It does not inject a UUID or
frontmatter. The parent must already exist.

The generic import action is labeled `Import into Database notes`. Selecting an
already-supported file within a linked root opens its existing binding rather
than duplicating it into SQLite.

`Select all shown` applies only to loaded visible leaves, never undisplayed
pages or collapsed descendants.

### Console handoff

`Use in Console` stages a versioned visible-body snapshot up to 80,000
characters, and that snapshot—not a 4,000-character UI preview—must reach the
provider context path.

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

## Search/RAG and MCP Boundaries

### FTS and RAG

FTS projection is required and eventual within the near-real-time target.
Optional file-backed RAG is not auto-enabled:

- it indexes body only with filename/path metadata;
- a path/filename generation change refreshes RAG metadata without re-embedding
  an unchanged body;
- the global queue is bounded and retains only the latest pending generation per
  note;
- queue overflow sets a dirty-sweep marker rather than growing without bound;
- missing/deleted UUIDs are suppressed from results immediately even if physical
  cleanup retries;
- failures or lag never block disk save or FTS.

### MCP v1

MCP read/search may return file projections with freshness and binding state.
MCP create remains `Database note` and says so explicitly. File-backed
update/delete returns a structured unsupported response directing the caller to
the interactive Library file commands. MCP never gains implicit filesystem
write authority in this tranche.

## Sync and Association Isolation

File-backed notes are local-only projections:

- notes Sync v2/sync-log triggers run only for `storage_kind = database`;
- Notes FTS triggers continue to index both note kinds, but their update trigger
  rebuilds only when projected title, body, or visibility changes—not when only
  `version`, observation time, raw hash, or frontmatter changes;
- `NotesScopeService` does not enqueue Sync v2 for file rows;
- remote inbound operations cannot update a file row or disk;
- an inbound UUID collision is a sync conflict, not a file mutation;
- legacy folder-sync timers, profiles, direct service calls, and UI actions are
  rejected for any overlapping root;
- file-note memberships and links use internal local-only association methods
  rather than the current generic sync-logging relation helper;
- keyword sync triggers require `sync_eligible = true`;
- a new keyword created only from file-note work starts ineligible and remains
  local unless explicitly promoted/reused through a Database-note workflow.

Legacy sync configuration detection includes its database roots, both TOML key
aliases, timer state, `sync_profiles.json`, and per-note bindings. Disabling
legacy overlap does not touch its files. Version 1 performs no automatic legacy
conversion and never reuses a legacy UUID that has pending or acknowledged Sync
v2 identity.

## Runtime Modes and Concurrency

The file coordinator has three same-version modes:

| Mode | Behavior |
| --- | --- |
| `disabled` | Kill switch for activation, monitoring, and file commands. Existing projections are diagnostic and make no freshness claim. Startup still inspects incomplete operations without replaying them. |
| `read_only` | Link, inventory, tree/search, monitor, inspect/export recovery, and reconcile external changes. No file mutation or restore into a linked root. |
| `read_write` | Enables explicit coordinator mutations after all health, lease, and recovery gates pass. |

`disabled` is not a schema downgrade. An older binary still rejects the migrated
database.

A mode-transition barrier rejects queued new saves, lets an already-started
operation finish or enter recovery, preserves newer editor buffers, and only
then changes mode. After `disabled`, each affected root completes identity and
hash reconciliation before it may claim freshness or return to `read_write`.
Applying any retained draft still requires a final fresh disk hash.

One OS-level coordinator lease per root, stored outside the repository, elects
the process allowed to activate, monitor, reconcile/project, or write that root.
The lease records the owner's current `read_only` or `read_write` capability;
write permission is still gated separately. A second Chatbook process is
passive read-only: it may inspect cached projection/recovery and exact-export a
freshly hashed file, but it starts no second watcher/reconciler and issues no
file commands. Stale owner leases are recovered only after process-liveness,
root-identity, and incomplete-operation checks.

The pilot permits at most one active writable root, although the schema supports
multiple roots and additional non-overlapping roots may be previewed or linked
read-only. Multiple hosts and network mounts are outside write-safety
guarantees.

### Startup operation recovery

Every runtime mode, including `disabled`, opens the recovery journal far enough
to inspect incomplete operations and completed emergency operations marked
`deferred_rebuild`. Startup takes the coordinator lease when available,
examines current root identity/containment plus source/destination paths and
hashes, and never blindly replays a filesystem mutation.

Classification is action-specific:

- save/create/restore: intended bytes present means finish recovery metadata and
  projection; old bytes/expected absence means `not_applied` with draft
  retained; any other bytes mean Attention;
- delete: verified target absence plus a valid deletion snapshot means complete
  the tombstone; the unchanged expected file means `not_applied` without
  unlinking; a recreated/different file means Attention;
- rename/move: absent source plus intended destination means complete the
  binding move; intended source plus absent destination means `not_applied`;
  both paths present, both absent, or unexpected hashes mean Attention.

Recovery metadata is repaired before main projection, and main projection is
then rebuilt idempotently from the observed disk result. If
`notes_recovery.db` cannot be opened and validated, every linked root is
read-only and no restore is allowed; Database notes continue to operate.

## Migration and Activation

### Pre-migration safety

The normal database constructor currently performs migrations, so file-backed
schema migration needs an earlier bootstrap boundary:

1. Acquire the single-instance migration lock.
2. Inspect the on-disk schema before ordinary DB construction.
3. Create a SQLite online backup in the owner-only app data directory.
4. Run integrity checks on source and backup.
5. Write the verified backup manifest last.
6. Only then allow the normal additive migration.

Migration adds the ownership columns/tables/indexes, gates note/keyword
sync-log triggers as specified above, and makes the FTS update trigger
content/visibility-conditional before any file projection can be inserted. It
creates/version-controls `notes_recovery.db` independently. It links no root and
mutates no note file.

The latest verified pre-migration backup is outside the 1 GiB recovery quota and
is retained until explicit pruning. If backup or verification fails, startup
aborts this migration and never continues partially. When the unchanged schema
is the immediately previous supported version, Chatbook may continue in an
explicit compatibility mode with the file coordinator `disabled` and ordinary
Database notes available. Any other unknown/incompatible schema aborts normal
database startup.

Emergency downgrade is an explicit offline restore. Chatbook preserves the
upgraded database pair first and warns that all post-backup Database-note and
metadata changes may be lost.

### Preview

Root preview is ephemeral and memory-only:

- no root/binding rows;
- no watcher or lease;
- no recovery payload;
- no index publication;
- no file writes.

It reports supported, ignored, read-only, inaccessible, special, and potential
protection counts plus estimated initial indexing/recovery bytes.

### Confirmed activation

Activation is crash-resumable:

1. Acquire the exclusive root coordinator lease for activation/monitoring,
   recording whether the requested capability is `read_only` or `read_write`.
2. Persist a durable root record in `activating`.
3. Start bounded watcher/event capture.
4. Inventory path metadata and classification.
5. Drain captured events.
6. If capture overflowed or root identity changed, discard the inventory and
   rescan.
7. Atomically publish roots/bindings/projections.

A crash leaves an explicit resume/discard activation state. Path navigation
becomes available from the published metadata while body FTS proceeds
incrementally. Source files remain unchanged.

Only selected protected items show `Protection pending`; each becomes writable
after its baseline replica verifies. Unprotected items become writable after
root activation and per-operation recovery health gates pass.

## Unlink, Forget, Purge, and Backup

### Unlink

Unlink crosses an operation barrier, stops monitoring, and releases the lease.
It never touches source files or Git. The root becomes detached while retaining
UUIDs, metadata, drafts/conflicts, and recovery bytes; those bytes still count
toward quota. A detached root reserves its path overlap until relink or forget.

Relink requires explicit confirmation plus root identity and inventory checks.
A recreated directory at the same path does not silently inherit old UUIDs.

### Forget

Forget removes root registration, bindings, projections, and note-specific
associations. It does not delete shared keyword definitions or source files.
Forget is blocked while pending/Attention operations or unresolved drafts exist;
those must be resolved or separately discarded first. Self-contained deletion
snapshots remain through their guaranteed expiry. Ordinary current replicas and
checkpoints lose their protection pin and become eligible for quota pruning;
the confirmation reports those logical bytes and recommends Unlink when the user
wants to retain them.

### Purge

Recovery purge is separate and requires typed confirmation showing scope, item
count, and logical compressed bytes. Pending operations, unresolved conflicts,
and drafts are excluded until resolved or explicitly discarded. Unexpired
30-day deletion snapshots are excluded; waiving that guarantee is a distinct
typed confirmation that changes the tombstone from `recoverable until <date>`
to `recovery purged`. Purge is logical deletion; it promises neither secure
erasure nor immediate SQLite/WAL shrinkage.

### Application-consistent DB backup

A later recovery export can back up the two SQLite databases:

1. pause Database-note/metadata writes, file commands, reconciliation projection
   and replica updates, and every other Chatbook database mutation;
2. settle or classify operations while retaining watcher events;
3. assign a shared backup epoch;
4. use SQLite online backup for main and recovery databases into private staging;
5. validate both;
6. write the pair manifest last.

This is an application-consistent epoch, not a cross-database transaction.
External file writers cannot be frozen. Source files and Git are excluded, so
exact unprotected content still depends on the working tree/repository.

Pair restore is offline only. It validates epoch and pair, preserves the current
pair, uses a recoverable staging marker, and never hot-swaps open databases.

## Security and Privacy

- Main/recovery databases, WAL/SHM files, backups, manifests, and temporary
  recovery artifacts use owner-only app-directory permissions (POSIX 0700
  directories and 0600 files where supported, equivalent owner-only policy
  elsewhere).
- Writable activation is refused if that protection or symlink-substitution
  safety cannot be enforced.
- Linked-root file permissions are not broadened.
- Absolute paths, bodies, raw hashes, and recovery payloads are excluded from
  routine logs. Exceptions are sanitized before user display.
- Recovery content is unencrypted plaintext SQLite. Same-user processes, device
  backups, and anyone with access to an unencrypted disk may read it.
- Full-disk encryption and an independent off-device backup are recommended.
- The Git remote protects only content that the user has committed and pushed.

## Performance and Non-Degradation Gates

- With no linked root, Chatbook starts no file watcher, reconciliation worker,
  recovery writer, lease, or file scan.
- Apart from the one-time guarded schema migration, existing Database Notes
  behavior and the zero-root steady-state startup path remain unchanged.
- A cached 5,000-file root reaches interactive tree navigation without reading
  file bodies before first paint.
- Tree rows carry metadata only and load children lazily.
- Navigator search returns its first 100-result page within 200 ms p95 on the
  fixed benchmark fixture, after warm-up and over at least 30 samples.
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
versions, generated tree fixture, warm-up, sample count, and timing boundaries.
Shared CI verifies deterministic behavior; the pinned runner owns hard timing
gates.

## Verification Strategy

### Focused per-PR tests

- Authority routing and immutable `storage_kind`.
- Projection idempotency by root, normalized path, raw hash, and presence state.
- Exact frontmatter/BOM/newline/final-newline/mode preservation.
- Strict encoding, size, malformed-frontmatter, hardlink, symlink, unreadable,
  and special-file classification.
- Save no-op behavior and editor-generation races.
- Every mutation outcome at each journal checkpoint, destination-path locking,
  no-replace publication, and operation-linked temporary cleanup.
- Normal complete-last ordering and pinned recovery-only `deferred_rebuild`
  consumption.
- Expected-hash conflict, overwrite/discard safety, and durable Attention.
- Missing/root-Offline/tombstone transitions and path-reuse identity.
- Delete snapshot verification, 30-day pin, stale confirmation, and restore.
- Protection inheritance/overrides, current replica invariant, quota, corrupt
  revision, bounded decompression, payload/operation garbage collection, and
  recovery failure.
- Search rank/paging/dedup/path-only fallback, durable incremental FTS progress,
  and path-only RAG metadata refresh.
- Notes/keyword/relation trigger isolation plus Sync
  v2/legacy/MCP/public-DB fail-closed boundaries.
- Narrow/wide focus, resize, autosave, conflict, preview, export, template, and
  Console handoff parity.
- Zero-configuration non-degradation.

Minimal test seams are:

- one coordinator fault-checkpoint callback;
- an injectable clock;
- a fake watcher event source.

Integration tests continue to use real `pathlib`/file descriptors, SQLite, and
temporary files rather than a broad fake filesystem abstraction.

### Nightly and release-blocking suites

- Process termination at every journal/activation/backup boundary, proving
  old-or-intended complete bytes and recoverable intent after process crashes.
- Real watcher timing, duplicate/move/overflow storms, polling fallback, and Git
  checkout/bulk/conflict-marker reconciliation.
- 5,000-file scale and fixed-runner p95 benchmarks.
- Two-process lease contention and stale-owner recovery.
- Equal/parent/child root and legacy-root overlap.
- Root identity change and same-path directory recreation.
- Migration backup failure, integrity failure, upgraded-pair rollback, and
  older-binary rejection documentation.
- Mode change during mutation, recovery corruption/full disk/over-quota, unlink,
  relink, forget, purge, and occupied-destination restore.
- POSIX mode-bit tests on POSIX; platform-appropriate expectations on Windows.

Process crash tests do not claim to simulate power loss. Directory-fsync ordering
is fault-injected; actual power-loss durability is documented as
filesystem-dependent. Network/cloud mounts are outside release guarantees.

The Git smoke matrix is intentionally small: status/diff, one local commit, one
local push to a disposable remote, checkout/bulk reconciliation, and conflict
markers. Chatbook-specific value does not justify automating broad pull/rebase
matrices here.

## Rollout

The architecture is multi-root capable, but delivery is phased so each safety
boundary can be proven before broader automation:

1. **Migration, guards, and kill switch**
   - pre-constructor verified backup;
   - additive schema and trigger isolation;
   - recovery DB bootstrap;
   - modes and fail-closed service routing;
   - zero-root non-degradation.
2. **Read-only file sources**
   - preview/activation/resume;
   - one writable-candidate root plus additional roots linked read-only;
   - tree, dedicated search, external monitoring, unlink/relink;
   - existing preview/export/template/Console parity;
   - no file mutation.
3. **Journaled writing**
   - save/create/rename/move and durable conflict resolution;
   - mandatory operation safety;
   - one active writable root;
   - exact confirmed-delete retention and minimal restore must land together
     before Delete is enabled.
4. **Opt-in continuous recovery**
   - folder/file protection rules and current replicas;
   - coalesced checkpoints, quota/history UI, and general restore/purge.
5. **Optional backup/RAG and additional post-gate hardening**
   - application-consistent database-pair backup;
   - optional file RAG and additional optimization/platform coverage after all
     required 5,000-file and supported-platform release gates already pass.
6. **Separately approved expansion**
   - multiple writable roots;
   - any Git stage/commit/push controls;
   - any remote/portable file-note identity or synchronization.

Implementation planning should decompose TASK-399 into atomic child tasks/PRs in
this dependency order. No phase may expose an action whose recovery and
fail-closed prerequisites belong only to a later phase.

## Alternatives Considered

### Keep legacy equal-peer bidirectional sync

Rejected. It retains ambiguous ownership, timestamp winners, weak move/delete
identity, and avoidable write noise. It remains only as a compatibility feature
for Database notes.

### Keep SQLite canonical and export files

Rejected. Chatbook edits would not immediately appear as ordinary Git changes,
and external editors/Git would require a second import/export authority loop.

### Use disk only, without a projection

Rejected. It would degrade Library FTS, metadata, links, Console handoff, RAG
invalidation, and large-tree performance.

### Store UUIDs in frontmatter or a repository manifest

Rejected. It would alter user files solely for Chatbook, create Git noise, and
make Chatbook metadata part of repositories that currently contain only notes.

### Store recovery bytes in the main database

Rejected. A separate recovery database better satisfies independent enumeration
and recovery when the main projection is unavailable, and it isolates recovery
quota/corruption from Database-note availability.

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
