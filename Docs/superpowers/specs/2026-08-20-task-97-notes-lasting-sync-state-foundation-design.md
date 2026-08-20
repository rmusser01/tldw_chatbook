# TASK-97: Notes Lasting-Sync State Foundation — Design

Date: 2026-08-20
Status: Draft for independent review
Task: [TASK-97](../../../backlog/tasks/task-97%20-%20Conflict-resolution-dialog-for-Notes-sync.md)
Governing decisions:

- [ADR-059: Notes Folder Import and Device-Local Sync Ownership](../../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md)
- [ADR-060: Notes Sync Round-trip and Interoperability Constraints](../../../backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md)

## Summary

TASK-97 establishes the private storage foundation for local Database Notes
lasting sync. It upgrades the existing profile-local `notes.sync_state` SQLite
owner from its one-time-import receipt schema to a shared schema that can also
represent paused sync roots, provisional bindings, and legacy-migration
receipts.

This slice does not activate a root, inspect the filesystem, reconcile content,
capture conflicts, resolve conflicts, run watchers, or change the Library UI.
The legacy single-root sync engine remains the only active sync owner throughout
this task. Later atomic tasks will build setup and activation, reconciliation
and durable conflict capture, journaled resolution and Undo, then the
Needs-attention UI and legacy cutover.

The original TASK-97 wording described a modal that interrupts a sync run and
offers file/app/skip. That contract predates ADR-059 and ADR-060. The accepted
product contract instead pauses an affected binding as **Needs attention** and
later offers **Keep file**, **Keep note**, or **Keep both** without blocking the
running worker. This design therefore redefines TASK-97 as the prerequisite
storage slice rather than implementing the obsolete interim prompt.

## Goals

1. Give the existing `notes.sync_state` owner one concurrency-safe schema
   coordinator shared by import receipts and lasting-sync state.
2. Preserve every existing import receipt behavior while upgrading schema v1
   to v2 additively.
3. Persist bounded, device-private paused candidate roots and provisional
   bindings for local Database Notes.
4. Enforce the ownership invariants that are knowable before filesystem
   admission without pretending lexical paths are canonical identities.
5. Provide an idempotent, read-only legacy-metadata migration seam that performs
   no filesystem, note, configuration, or legacy-row mutation.
6. Make migration drift, malformed records, and capacity failures explicit,
   redacted, and independently recoverable.
7. Leave an unambiguous boundary for the later setup/coordinator task to
   revalidate candidates before activation.

## Non-goals

- No root picker, setup wizard, dry-run UI, or capability probe.
- No path resolution, `stat`, file open/read/write, directory traversal, or
  symlink handling.
- No watcher, coordinator lease, reconciliation loop, or automatic sync.
- No file/note content snapshots, hashes of content, conflicts, deletion
  records, recovery payloads, or cross-authority journal.
- No conflict resolver or **Keep file / Keep note / Keep both** behavior.
- No Needs-attention queue, modal, activity-log entry, notification, or focus
  behavior.
- No server-backed lasting-sync roots. The product must continue to describe
  that capability as unavailable until the versioned server contract exists.
- No changes to File Notes ownership, tables, recovery, or write paths.
- No retirement, disabling, or mutation of the legacy single-root sync engine.
- No portable export or centralized backup of device-private sync state.

## Workstream and Dependency Shape

TASK-97 is an atomic foundation task, not an umbrella parent. Later tasks use
higher IDs and may depend on TASK-97 without reversing the repository's task-ID
dependency rule.

The approved sequence is:

1. **TASK-97 — private schema and local registry foundation** (this design).
2. **Root setup and activation** — explanation, guarded dry-run, coordinator
   lease, overlap/capability validation, and disabling the legacy route before
   the first new root becomes active.
3. **Reconciliation and conflict capture** — safe observations, deterministic
   classification, bounded private snapshots, and durable Needs-attention
   records.
4. **Resolution and recovery** — revalidation, durable journal, Keep file,
   Keep note, Keep both, receipts, and Undo.
5. **Review UI and cutover** — Needs-attention queue/modal/activity entry and
   removal of the obsolete legacy conflict flow.

Each slice must remain independently testable and must not pull behavior from a
later slice forward merely because its tables have been anticipated.

## Existing Boundary

`NoteImportReceiptRepository` already owns a profile-local private SQLite file
registered as `notes.sync_state`. Its v1 schema stores only opaque identifiers,
private digests, bounded lifecycle state, and one-time-import receipt metadata.
The owner is intentionally excluded from centralized backup and portable
export. The registry description already reserves it for "device-private import
receipts and future lasting-sync state."

The repository currently initializes and versions this owner directly. Adding a
second repository that independently owns `PRAGMA user_version` would make
import-first, sync-first, and simultaneous initialization race or overwrite one
another. TASK-97 replaces that split ownership with a shared, domain-model-free
schema coordinator while preserving the public receipt API.

## Architecture

### 1. Shared schema coordinator

A small Notes-private schema module,
`tldw_chatbook/Notes/notes_sync_state_schema.py`, is the registered owner and
only production caller of `connect_private_sqlite` for `notes.sync_state`. It
imports neither receipt-domain nor sync-domain models. Both repositories use
its `notes_sync_state_transaction(...)` context manager; neither repository
opens this SQLite owner directly. The private-owner registry and inventory move
the existing owner seam from `note_import_receipts` to this shared module rather
than widening one owner to several connection callers.

Initialization must:

1. open the existing private connection with the owner's established pragmas;
2. read `PRAGMA user_version` without reserving the writer slot;
3. when it is 2, validate the exact committed census in a read transaction and
   take no writer lock;
4. when it is 0 or 1, enter `BEGIN IMMEDIATE`, re-read `user_version` after the
   writer lock is held, and either validate v2 if another initializer won or
   create/migrate to the complete v2 schema additively;
5. reject unknown future or malformed supported versions without destructive
   repair;
6. set `user_version = 2` only after every required table, index, and constraint
   exists;
7. commit schema initialization atomically or roll it back completely; and
8. only after schema initialization ends, begin the repository operation's
   requested deferred or immediate transaction and yield the connection.

The writer-lock reread is required only for 0/1 migration paths. Checking a
version before the lock and migrating from that stale value would let two real
connections choose the same path. Conversely, making every ordinary v2 receipt
read acquire `BEGIN IMMEDIATE` would turn readers into competing writers and
violate receipt compatibility. Seeing committed version 2 is safe because the
atomic migrator writes `user_version` last; its read-only census fast path never
observes a partially committed v2.

The existing receipt tables, columns, indexes, rows, and lifecycle semantics are
not rebuilt or transformed. The compatibility promise is behavioral: existing
receipt projections, transitions, retry behavior, and data remain unchanged.
Whole-database byte identity is not promised because SQLite may legitimately
change pages while adding schema objects.

### 2. Canonical v2 schema

Version 2 is the exact v1 receipt schema plus the four tables and seven indexes
below. Every v1 table, column, constraint, and index remains byte-for-byte the
canonical SQL already defined by `note_import_receipts`; the shared coordinator
moves those constants without changing them. No verified path identity,
filesystem capability, serialization profile, observation, conflict, recovery,
or journal column is reserved in v2. The later slice that first owns those
contracts must add a new schema version rather than repurpose a nullable field.

The canonical new-table DDL is:

```sql
CREATE TABLE sync_migration_runs (
    migration_id TEXT PRIMARY KEY
        CHECK (length(migration_id) = 36),
    source_kind TEXT NOT NULL
        CHECK (source_kind = 'legacy_notes_sync_v1'),
    source_revision_before TEXT NOT NULL
        CHECK (
            length(source_revision_before) = 64
            AND source_revision_before NOT GLOB '*[^0-9a-f]*'
        ),
    source_revision_after TEXT
        CHECK (
            source_revision_after IS NULL OR (
                length(source_revision_after) = 64
                AND source_revision_after NOT GLOB '*[^0-9a-f]*'
            )
        ),
    state TEXT NOT NULL DEFAULT 'pending_recheck'
        CHECK (state IN ('pending_recheck', 'matched_recheck', 'drifted')),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    UNIQUE (source_kind, source_revision_before),
    CHECK (
        (state = 'pending_recheck' AND source_revision_after IS NULL)
        OR (state = 'matched_recheck'
            AND source_revision_after IS NOT NULL
            AND source_revision_after = source_revision_before)
        OR (state = 'drifted'
            AND source_revision_after IS NOT NULL
            AND source_revision_after <> source_revision_before)
    )
);

CREATE TABLE sync_roots (
    root_id TEXT PRIMARY KEY
        CHECK (length(root_id) BETWEEN 1 AND 256),
    lexical_root_path TEXT NOT NULL
        CHECK (
            length(lexical_root_path) BETWEEN 1 AND 32768
            AND instr(lexical_root_path, char(0)) = 0
        ),
    display_name TEXT NOT NULL
        CHECK (length(display_name) BETWEEN 1 AND 255),
    direction TEXT NOT NULL
        CHECK (direction IN (
            'unspecified', 'folder_to_notes', 'notes_to_folder', 'bidirectional'
        )),
    state TEXT NOT NULL DEFAULT 'candidate'
        CHECK (state IN ('candidate', 'paused', 'disconnected')),
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version > 0),
    needs_rescan INTEGER NOT NULL DEFAULT 1 CHECK (needs_rescan IN (0, 1)),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    source_kind TEXT CHECK (
        source_kind IS NULL OR source_kind = 'legacy_notes_sync_v1'
    ),
    source_locator_digest TEXT CHECK (
        source_locator_digest IS NULL OR (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    source_migration_id TEXT
        REFERENCES sync_migration_runs(migration_id) ON DELETE RESTRICT,
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    CHECK (
        (source_kind IS NULL
         AND source_locator_digest IS NULL
         AND source_migration_id IS NULL)
        OR (source_kind IS NOT NULL
            AND source_locator_digest IS NOT NULL
            AND source_migration_id IS NOT NULL)
    ),
    CHECK (
        direction <> 'unspecified' OR (
            source_kind = 'legacy_notes_sync_v1'
            AND needs_rescan = 1
            AND reason_code = 'legacy_direction_invalid'
        )
    )
);

CREATE TABLE sync_bindings (
    binding_id TEXT PRIMARY KEY
        CHECK (length(binding_id) BETWEEN 1 AND 256),
    root_id TEXT NOT NULL
        REFERENCES sync_roots(root_id) ON DELETE RESTRICT,
    note_id TEXT NOT NULL
        CHECK (length(note_id) BETWEEN 1 AND 256),
    lexical_relative_path TEXT NOT NULL
        CHECK (
            length(lexical_relative_path) BETWEEN 1 AND 32768
            AND instr(lexical_relative_path, char(0)) = 0
        ),
    path_key TEXT CHECK (
        path_key IS NULL OR (
            length(path_key) BETWEEN 1 AND 32768
            AND instr(path_key, char(0)) = 0
        )
    ),
    state TEXT NOT NULL DEFAULT 'candidate'
        CHECK (state IN ('candidate', 'needs_attention', 'disconnected')),
    row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version > 0),
    needs_rescan INTEGER NOT NULL DEFAULT 1 CHECK (needs_rescan IN (0, 1)),
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    source_kind TEXT CHECK (
        source_kind IS NULL OR source_kind = 'legacy_notes_sync_v1'
    ),
    source_locator_digest TEXT CHECK (
        source_locator_digest IS NULL OR (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    source_migration_id TEXT
        REFERENCES sync_migration_runs(migration_id) ON DELETE RESTRICT,
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    updated_at INTEGER NOT NULL CHECK (updated_at > 0),
    CHECK (
        (source_kind IS NULL
         AND source_locator_digest IS NULL
         AND source_migration_id IS NULL)
        OR (source_kind IS NOT NULL
            AND source_locator_digest IS NOT NULL
            AND source_migration_id IS NOT NULL)
    )
);

CREATE TABLE sync_migration_items (
    migration_id TEXT NOT NULL
        REFERENCES sync_migration_runs(migration_id) ON DELETE RESTRICT,
    item_kind TEXT NOT NULL
        CHECK (item_kind IN ('root', 'binding', 'legacy_conflict')),
    source_locator_digest TEXT NOT NULL
        CHECK (
            length(source_locator_digest) = 64
            AND source_locator_digest NOT GLOB '*[^0-9a-f]*'
        ),
    outcome TEXT NOT NULL
        CHECK (outcome IN ('created', 'matched', 'rejected', 'needs_rescan')),
    root_id TEXT REFERENCES sync_roots(root_id) ON DELETE RESTRICT,
    binding_id TEXT REFERENCES sync_bindings(binding_id) ON DELETE RESTRICT,
    reason_code TEXT CHECK (
        reason_code IS NULL OR (
            length(reason_code) BETWEEN 1 AND 64
            AND reason_code NOT GLOB '*[^a-z0-9_]*'
            AND substr(reason_code, 1, 1) GLOB '[a-z]'
        )
    ),
    created_at INTEGER NOT NULL CHECK (created_at > 0),
    PRIMARY KEY (migration_id, item_kind, source_locator_digest),
    CHECK (
        (
            item_kind = 'root'
            AND outcome IN ('created', 'matched', 'needs_rescan')
            AND root_id IS NOT NULL
            AND binding_id IS NULL
            AND (
                (outcome IN ('created', 'matched') AND reason_code IS NULL)
                OR (outcome = 'needs_rescan' AND reason_code IS NOT NULL)
            )
        ) OR (
            item_kind = 'root'
            AND outcome = 'rejected'
            AND root_id IS NULL
            AND binding_id IS NULL
            AND reason_code IS NOT NULL
        ) OR (
            item_kind = 'binding'
            AND outcome IN ('created', 'matched', 'needs_rescan')
            AND root_id IS NULL
            AND binding_id IS NOT NULL
            AND (
                (outcome IN ('created', 'matched') AND reason_code IS NULL)
                OR (outcome = 'needs_rescan' AND reason_code IS NOT NULL)
            )
        ) OR (
            item_kind = 'binding'
            AND outcome = 'rejected'
            AND root_id IS NULL
            AND binding_id IS NULL
            AND reason_code IS NOT NULL
        ) OR (
            item_kind = 'legacy_conflict'
            AND outcome = 'needs_rescan'
            AND reason_code IS NOT NULL
            AND NOT (root_id IS NOT NULL AND binding_id IS NOT NULL)
        )
    )
);
```

The canonical new indexes, including their exact partial predicates, are:

```sql
CREATE INDEX idx_sync_migration_runs_state
    ON sync_migration_runs(state, updated_at);
CREATE INDEX idx_sync_roots_state
    ON sync_roots(state, updated_at);
CREATE UNIQUE INDEX idx_sync_roots_legacy_source
    ON sync_roots(source_kind, source_locator_digest)
    WHERE source_kind IS NOT NULL AND state <> 'disconnected';
CREATE INDEX idx_sync_bindings_root_state
    ON sync_bindings(root_id, state, updated_at);
CREATE UNIQUE INDEX idx_sync_bindings_live_note
    ON sync_bindings(note_id)
    WHERE state <> 'disconnected';
CREATE UNIQUE INDEX idx_sync_bindings_live_path_key
    ON sync_bindings(root_id, path_key)
    WHERE state <> 'disconnected' AND path_key IS NOT NULL;
CREATE INDEX idx_sync_migration_items_outcome
    ON sync_migration_items(migration_id, outcome, item_kind);
```

The coordinator owns an exact schema census covering table names, ordered
columns, declared types, nullability, defaults, primary keys, foreign keys,
canonical table SQL, index columns, uniqueness, and partial-index SQL. A
database claiming v2 but differing from that census fails closed. Tests compare
a hand-authored fresh-v2 fixture with an actual v1 repository database upgraded
through the coordinator; init-order equality alone is not the oracle.
Migration receipt aggregates are exact SQL counts grouped by
`(item_kind, outcome)` from `sync_migration_items` in the same read transaction.
No generic or independently mutable root/binding/failure summary is stored.

### 3. Lasting-sync repository

`NotesSyncStateRepository` is the sole application API for roots, bindings, and
legacy-migration receipts in this slice. It returns immutable, slotted typed
models with private-safe `repr` output. Raw rows, SQLite connections, absolute
paths, note IDs, digests, and exception strings do not leak through diagnostic
representations.

The repository is deliberately narrow:

- create/list/get/update paused candidate roots;
- create/list/get/update provisional bindings;
- validate global note ownership and admitted path-key uniqueness;
- record/read migration source revisions, item outcomes, and completion
  receipts; and
- report bounded aggregate counts and reason codes.

It exposes no activation, watcher, reconciliation, conflict, resolution,
journal, or content API.

### 4. Paused candidate roots

A root projection maps exactly the `sync_roots` columns above. The v2 lifecycle
contains only `candidate`, `paused`, and `disconnected`; active/running is not a
representable value. Its source fields are all-null for a future user-created
candidate or all-present for a legacy migration generation.

Direction `unspecified` is permitted only for an explicit invalid migrated
legacy value; it always carries `needs_rescan = 1` and reason
`legacy_direction_invalid`, and the setup slice must require a direction choice
before admission. A missing direction uses the documented `bidirectional`
default.

Verified canonical identity and filesystem capability do not exist in v2. The
later setup slice adds them in a new schema version after it defines their exact
cross-platform contract. A lexical path is migration input, not proof of
containment, existence, equivalence, writability, or safe ownership.

No root created by this slice can enter an active/running state. The repository
must reject such a transition because there is no coordinator or admission
proof yet.

### 5. Provisional bindings

A binding projection maps exactly the `sync_bindings` columns above. The v2
lifecycle contains only `candidate`, `needs_attention`, and `disconnected`;
admitted/bound is not representable. `path_key` remains null until the later
dry-run defines portability normalization.

Verified file identity, representation profile, and last observations do not
exist in v2. Legacy values are inputs to the private source-revision digest only;
they are not copied into authoritative observation fields. Later reconciliation
must observe them anew.

### 6. Ownership and uniqueness

The private owner enforces transactionally:

- at most one non-disconnected lasting-sync binding for a Database Note across
  all roots on this device;
- a paused or candidate root still owns its non-disconnected bindings;
- disconnected bindings no longer participate in active ownership uniqueness;
- once a non-null `path_key` exists, it is unique within the root among
  non-disconnected bindings; and
- optimistic versions advance on every mutable row update.

The per-note invariant is enforceable now because note identity is already
known. The per-root path invariant is conditional: TASK-97 stores the lexical
relative path but does not derive `path_key`. The later dry-run must determine
filesystem portability rules and populate a normalized key before a binding can
become admitted. A plain case-fold or host-only normalization here would create
false authority.

The exact partial unique indexes above express these invariants in SQLite.
Repository preflight still provides deterministic batch behavior; indexes are
the final race guard, not a mechanism for choosing an arbitrary winner.

Root lifecycle changes use named repository operations, not a generic state
setter. `disconnect_root(root_id, expected_version)` is one immediate
transaction that changes the root to `disconnected`, advances its version, and
changes every non-disconnected child binding to `disconnected` with its own
version advance. A disconnected root cannot be reopened. An individual binding
may disconnect without disconnecting its root. Repository reads fail closed if
corrupt data ever presents a disconnected root with a live child binding. Thus
root disconnection releases both root and binding capacity and cannot leave a
hidden note owner behind.

Every binding create or mutable update acquires an immediate transaction and
checks its parent root in that same transaction. The parent must be `candidate`
or `paused`; a disconnected or missing root rejects the whole operation. Binding
`disconnected` is terminal in v2—there is no reopen operation. These write
guards, the atomic root-disconnect operation, and the read-side corruption check
close the parent/child invariant without triggers.

## Legacy Metadata Migration Seam

### Invocation and authority

Migration is lazy. TASK-97 provides and validates the migration operation; the
next slice invokes it when lasting-sync setup is first opened. Merely opening
the application or receipt repository does not migrate legacy metadata.

The legacy engine remains the sole active sync authority while candidate state
is recorded. Migration neither disables nor edits it. The later activation
slice must disable the legacy route before any new root can activate; the two
engines must never run in parallel.

### Read-only source capture

Migration reads legacy configuration and per-note metadata using their normal
read boundaries. Reading the config file and opening the two SQLite owners are
necessary I/O. The no-filesystem authority guarantee is narrower and exact:
migration performs no access through a migrated candidate/root path, reads no
candidate file content, and performs no note write, configuration write, or
legacy-row cleanup.

Configuration, ChaChaNotes, and `notes.sync_state` cannot share a transaction.
No protocol can prove a globally atomic source snapshot, so every migrated root
and binding remains provisional even when two reads match. The bounded protocol
is:

1. read config snapshot A through the real config boundary;
2. open a fresh ChaChaNotes read transaction and read Notes snapshot A;
3. form the canonical source projection defined below from config fields
   `sync_directory`, `sync_direction`, and `sync_conflict_resolution`, plus each
   relevant note's `id`, `file_path_on_disk`,
   `relative_file_path_on_disk`, `sync_root_folder`,
   `last_synced_disk_file_hash`, `last_synced_disk_file_mtime`,
   `is_externally_synced`, `sync_strategy`, `sync_excluded`, `file_extension`,
   `version`, and `deleted`, plus the legacy `sync_conflicts` fields `id`,
   `session_id`, `note_id`, `file_path`, `conflict_type`, `db_content_hash`,
   `disk_content_hash`, `db_modified_time`, `disk_modified_time`, `resolution`,
   and `resolved_at`; derive private digest A;
4. under one `notes.sync_state` immediate transaction, insert or reopen a
   `pending_recheck` migration run keyed by `(source_kind, digest A)`, preflight
   the complete batch, and write its provisional candidates/items atomically;
5. after that destination commit, read fresh config snapshot B and a fresh
   ChaChaNotes transaction B, canonicalize the same projection, and derive
   digest B; a second read from transaction A is explicitly invalid evidence;
6. in a new destination transaction, compare-and-set the run to
   `matched_recheck` only when A equals B or `drifted` otherwise, recording
   digest B. The `UPDATE` predicate must include
   `state = 'pending_recheck' AND source_revision_after IS NULL`; when it changes
   zero rows, reread and return the already committed terminal receipt without
   mutation; and
7. on crash after step 4, leave `pending_recheck`; replay of digest A performs
   the missing fresh recheck without duplicating candidates.

Matching digests mean only that this bounded protocol did not observe drift.
They never make candidates authoritative or remove the later dry-run and
revalidation requirement. A new digest creates a new migration run and may
update only migration-owned `candidate` rows through their stable private
source-locator digests. It cannot overwrite a paused/reviewed row, activate,
delete, or steal an existing non-disconnected binding.

Replay follows one exact state machine:

- no run for digest A: execute step 4 once, including candidate/item writes;
- existing `pending_recheck` run for digest A: do not preflight or rewrite its
  candidates/items; perform only a fresh snapshot B and the conditional final
  state update. Concurrent rechecks may race, but only one compare-and-set can
  commit and the loser returns that immutable receipt;
- existing `matched_recheck` or `drifted` run for digest A: return its immutable
  receipt without source or candidate writes; and
- a genuinely new digest: open a new run. An exact source-locator match owned by
  the same migration run is `matched`, never a duplicate. An exact locator
  owned by an older run may be version-updated/rebound only while the row is
  still migration-owned `candidate`; a paused, needs-attention, disconnected,
  manually created, or differently located claim is not overwritten.

Duplicate-note preflight excludes exact locator matches permitted by this state
machine. All other incoming locators in the same note equivalence class are
rejected together, and a claim against a different existing live locator is
rejected. This makes replay independent of input and insertion order.

“Relevant note” is one exact source predicate. A note is selected when any of
these legacy fields indicates participation:

```text
is_externally_synced = 1
OR sync_excluded = 1
OR file_path_on_disk IS NOT NULL
OR relative_file_path_on_disk IS NOT NULL
OR sync_root_folder IS NOT NULL
OR last_synced_disk_file_hash IS NOT NULL
OR last_synced_disk_file_mtime IS NOT NULL
OR sync_strategy IS NOT NULL
```

The source reader selects every such row, including soft-deleted rows, and then
sorts by exact note ID as defined below. It selects legacy conflict rows only
when `resolution IS NULL OR resolution = 'skip'`, because those are the
unresolved/review outcomes; it sorts them by integer conflict ID. Resolved
`use_db`, `use_disk`, or `merge` history is left untouched and omitted from the
migration source. This selection rule is identical for snapshots A and B.

### Canonical source and locator digests

Every migration digest uses the existing Notes import canonical JSON contract:
`json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)`,
encoded as UTF-8 and hashed with SHA-256 lowercase hex. Values are validated
before hashing; JSON scalar types are preserved and Unicode is not normalized.
SQLite REAL timestamp fields are encoded as `float.hex()` strings; non-finite
values use the literal marker `"invalid_non_finite_real"` and make their source
item rejected/review-only without blocking valid siblings. Missing config keys
use the current product defaults exactly:
`sync_directory = "~/Documents/Notes"` and
`sync_direction = "bidirectional"`. For conflict policy, the newer
`notes.sync_conflict_resolution` key wins when present; otherwise the legacy
`notes.conflict_resolution` key is read; otherwise the value is
`"newer_wins"`. The selected value is stored in the projection under
`sync_conflict_resolution`. Present config values must be JSON scalars and are
encoded with their exact type; a non-string directory/direction is retained in
the digest but yields a rejected root or `unspecified` direction respectively.
Non-scalar config values abort before destination writes with bounded reason
`invalid_config_type`. Conflict policy is digest input only in TASK-97 and
grants no new resolution behavior.

The source-revision value has this exact top-level shape:

```json
{
  "config": {
    "sync_conflict_resolution": "<json-scalar>",
    "sync_direction": "<json-scalar>",
    "sync_directory": "<json-scalar>"
  },
  "conflicts": [
    {
      "db_content_hash": "<string-or-null>",
      "db_modified_time": "<string-or-null>",
      "disk_content_hash": "<string-or-null>",
      "disk_modified_time": "<hex-float-string-or-null>",
      "file_path": "<string>",
      "id": "<integer>",
      "note_id": "<string-or-null>",
      "resolution": "<string-or-null>",
      "resolved_at": "<string-or-null>",
      "session_id": "<string>",
      "conflict_type": "<string>"
    }
  ],
  "notes": [
    {
      "deleted": "<boolean>",
      "file_extension": "<string-or-null>",
      "file_path_on_disk": "<string-or-null>",
      "id": "<string>",
      "is_externally_synced": "<boolean>",
      "last_synced_disk_file_hash": "<string-or-null>",
      "last_synced_disk_file_mtime": "<hex-float-string-or-null>",
      "relative_file_path_on_disk": "<string-or-null>",
      "sync_excluded": "<boolean>",
      "sync_root_folder": "<string-or-null>",
      "sync_strategy": "<string-or-null>",
      "version": "<integer>"
    }
  ],
  "type": "tldw_notes_sync_legacy_source_revision",
  "version": 1
}
```

Angle-bracket strings above denote the required scalar type, not literal
stored values. Notes are sorted by exact `id` text. Conflicts are sorted by
integer `id`. Within objects, `sort_keys=True` supplies key order. String/null,
integer, boolean, and finite-real types remain distinct JSON scalars, so values
such as `null`, `false`, `0`, and `"0"` cannot collide.

Stable locator digests use the same canonical encoder:

- raw legacy scalar: `{"type": "tldw_notes_sync_legacy_value", "value":
  <exact JSON scalar>, "version": 1}`. Its digest is computed before path-text
  admission and is the fixed-size identity for null, non-string, empty, NUL,
  or over-limit values; raw malformed values are not persisted in receipts;
- root: `{"lexical_root_path": <exact bounded stored text>, "type":
  "tldw_notes_sync_legacy_root_locator", "version": 1}`;
- recognizable binding: `{"lexical_relative_path": <exact bounded stored
  text>, "note_id": <exact note id>, "root_locator_digest": <root digest>,
  "type": "tldw_notes_sync_legacy_binding_locator", "version": 1}`; and
- migration item: `{"item_kind": <root|binding|legacy_conflict>,
  "legacy_primary_key": <valid root/binding locator digest | rejected-root
  object | rejected-binding object | integer conflict id>, "type":
  "tldw_notes_sync_legacy_item_locator", "version": 1}`. A rejected-root
  primary key is `{"field": "notes.sync_directory", "value_digest": <raw
  scalar digest>}`. A rejected-binding primary key is `{"note_id": <exact
  note id>, "relative_value_digest": <raw scalar digest>,
  "root_value_digest": <raw scalar digest>}`.

The migration-item form also identifies rejected inputs and therefore never
depends on a destination ID or a valid candidate path. Its canonical hash is
the non-null `sync_migration_items.source_locator_digest`. Lexical root grouping
compares exact stored strings; it performs no Unicode, case, separator, tilde,
environment-variable,
absolute-path, or filesystem normalization. Legacy direction maps
`disk_to_db`/`folder_to_notes` to `folder_to_notes`,
`db_to_disk`/`notes_to_folder` to `notes_to_folder`, and `bidirectional` to
itself. A missing key takes the default above; any other value produces
`unspecified` plus a bounded review reason.

### Grouping and malformed inputs

Recognizable legacy rows are grouped by stored lexical root into paused
candidate roots and provisional bindings. They are not described as canonical
or safe. A missing, nonexistent, relative, adversarial, or platform-foreign path
may still be recorded as bounded private lexical text for later review.

One malformed or out-of-contract root/binding does not block independent
siblings. Duplicate ownership is preflighted as an equivalence class: every
incoming member that claims the same note is recorded as rejected/review-only,
and no binding for that note is inserted. An incoming claim against an existing
non-disconnected binding is likewise rejected while the existing owner remains
unchanged. Input ordering can never choose a winner. The partial unique index is
retained as the final invariant guard. Raw exception text and physical paths are
never persisted as error messages or logged.

Legacy conflict rows are not converted into actionable new conflict records.
Their content and observations cannot satisfy the new exact-side and recovery
contract. Migration sets only `needs_rescan`; the later reconciler must
rediscover and capture both current sides through the new boundary.

### No-filesystem guarantee

The migration layer treats stored candidate paths as data. It must not call
`resolve`, `absolute`, `stat`, `lstat`, `open`, directory iteration, or an
equivalent operation **on a migrated root or relative file candidate**. A
focused test uses nonexistent/adversarial stored candidates and operand-aware
spies or injected source collaborators that fail only when such a candidate is
used for filesystem access. Normal config reads and private database artifact
opens remain allowed and observable; the test must not globally patch `open`,
`lstat`, or `Path.resolve`.

## Capacity and Bounds

The foundation uses fixed safety constants rather than adding premature user
configuration:

- `MAX_SYNC_ROOTS = 64` live non-disconnected roots per profile; and
- `MAX_IMPORT_ENTRIES = 100_000` total live non-disconnected lasting-sync
  bindings per profile, reusing the established Notes import discovery ceiling.

Capacity is a global preflight after malformed and duplicate equivalence classes
have been classified but before any destination write. If the remaining valid
request would exceed either ceiling, the entire request aborts: it creates no
roots, bindings, migration items, or migration-run receipt. Capacity is never an
item-local outcome and no sibling is committed. Error objects and messages
report only bounded counts and reason codes, not identifiers or paths.

These are storage safety ceilings, not product pagination sizes. Later UI and
coordinator slices may choose smaller operational batches without changing the
owner's maximum admitted state.

## Privacy, Backup, and Diagnostics

ADR-059 and ADR-060 keep all physical sync ownership device-private. Therefore:

- `notes.sync_state` remains registered as a private file owner;
- centralized backup and portable export remain disabled;
- absolute/lexical root paths, relative paths, note IDs, file identities,
  hashes, revision digests, and future recovery content stay inside the owner;
- public/repr/log projections contain lifecycle, counts, booleans, versions,
  and bounded reason codes only;
- raw SQLite errors are wrapped without private values; and
- no schema or API encourages copying this state into ordinary note metadata,
  Sync-v2 payloads, activity logs, or server requests.

The repository may accept and return private values to its trusted caller, but
its `repr`, exception text, logging, and aggregate diagnostics are redacted by
construction.

## Failure and Recovery Behavior

- Unknown schema versions fail closed before any schema mutation.
- A v1-to-v2 migration is all-or-nothing under the writer transaction.
- Concurrent initializers converge on one valid v2 schema.
- A capacity failure or unexpected final-index race rolls back the whole
  requested operation; known duplicate classes follow the deterministic
  preflight rule and choose no winner.
- Source drift records a provisional generation and `needs_rescan`; it does not
  discard prior candidates or claim freshness.
- A malformed source item records a bounded failure independently of valid
  siblings.
- Candidate state never grants mutation authority, so a process crash during
  this slice cannot cause file or note changes.
- The later setup task must revalidate every candidate and observation before
  activation regardless of migration receipt state.

## Verification Strategy

### Shared schema and receipt compatibility

- Create an actual v1 database through the real
  `NoteImportReceiptRepository`, seed meaningful sessions/effects, then open the
  sync repository and verify the v2 upgrade.
- Verify every pre-upgrade receipt row, projection, aggregate, allowed
  transition, rejection, and retry behavior that the seeded fixture exercises.
- Initialize a new database receipt-first and sync-first; both yield the same
  complete v2 schema and both repositories remain usable.
- Start separate real SQLite connections behind a barrier so two initializers
  contend. Prove they converge without duplicate migration, partial schema, or
  lock leakage.
- Mutation-test the post-lock `user_version` reread and final version write.
- Prove ordinary committed-v2 reads validate through the read-only fast path
  and do not reserve SQLite's writer slot.
- Compare a hand-authored complete fresh-v2 schema to a real v1 upgrade using
  the exact census, including table checks and partial-index predicates.

### Root and binding contracts

- Cover immutable/redacted typed projections and error messages.
- Prove non-disconnected note ownership is global across roots, including
  paused roots, and disconnected rows release that ownership.
- Prove conditional path-key uniqueness and allow null provisional keys until
  dry-run.
- Prove optimistic version checks and atomic batch rejection.
- Prove root disconnect atomically disconnects/version-bumps every live child,
  releases both ceilings, cannot reopen, and rejects corrupted
  disconnected-root/live-binding state.
- Prove create/update under a disconnected parent and binding reopen both fail,
  including a two-connection disconnect/create race; prove the canonical DDL
  rejects `unspecified` direction without the exact migration review state.
- Cover exact limits at 64 roots and 100,000 bindings plus one-over rejection
  without constructing unbounded payload content.

### Legacy migration

- Use a real ChaChaNotes fixture with recognizable legacy configuration and
  per-note metadata; do not handwrite a partial schema and call it current.
- Cover multiple lexical roots, duplicate equivalence classes with no arbitrary
  winner, malformed siblings,
  legacy conflict markers, idempotent replay, and a changed source revision.
- Mutation-test canonical digest stability across row input order, dictionary
  order, Unicode spellings, missing/default/aliased config, and distinct JSON
  scalar types; prove every locator input change changes the intended digest.
- Prove null, non-string, empty, NUL, and over-limit rejected paths each produce
  a deterministic non-null item locator without persisting the raw value.
- Mutation-test every relevant-note predicate term and the exact unresolved/skip
  conflict predicate so unrelated rows cannot perturb or disappear from a run.
- Reject impossible migration-item field combinations through the canonical
  DDL and derive aggregate counts from items rather than stored summaries.
- Simulate cross-owner drift and prove the resulting generation remains
  provisional and marked for rescan.
- Race two real connections finalizing the same pending run with different B
  snapshots; prove one terminal receipt wins and the loser cannot overwrite its
  state or `source_revision_after`.
- Use operand-aware filesystem spies and nonexistent/adversarial stored paths;
  prove no candidate path is accessed while required config/database I/O still
  occurs.
- Assert no file, note, config, or legacy metadata mutation and no new active
  root/watcher/coordinator path.

### Owner and governance boundaries

- Extend the private-owner inventory tests so the shared schema and new
  repository still map to `notes.sync_state` and remain backup excluded.
- Search schema/API/log surfaces for prohibited content and raw path leakage.
- Run the complete existing import-receipt test module after the migration
  tests, not only new focused nodes.

## Acceptance Criteria Mapping

1. **Shared schema compatibility:** v1 receipt databases upgrade atomically to
   v2, and both repositories work in either initialization order and under real
   concurrent connections without changing existing receipt behavior.
2. **Paused registry:** local Database Notes candidate roots and provisional
   bindings persist privately with optimistic versions, redacted projections,
   global live-note uniqueness, and conditional admitted-path uniqueness.
3. **Safe migration seam:** legacy metadata can be captured idempotently into
   paused candidates without accessing migrated candidate paths or mutating
   files, notes, config, or legacy rows; fresh A/B snapshots bound drift claims,
   and conflicts/malformed items remain provisional and require rescan.
4. **Bounds and privacy:** the exact 64-root and 100,000-binding ceilings reject
   the whole request atomically, and
   physical identifiers remain confined to the backup-excluded
   `notes.sync_state` owner and absent from diagnostics.
5. **No premature authority:** this slice provides no activation, watcher,
   reconciliation, conflict-content, resolver, journal, UI, or server-backed
   sync behavior, and the legacy engine remains the only active owner.

## ADR Check

ADR required: no

ADR paths:

- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: TASK-97 directly implements the already accepted device-private owner,
binding uniqueness, migration, backup, and legacy-engine boundaries. It makes no
new architectural choice beyond selecting the smallest atomic first slice of
those decisions.

## Alternatives Considered

| Alternative | Decision |
| --- | --- |
| Implement the original per-conflict blocking modal now | Rejected. It conflicts with the accepted Needs-attention model and would build UI on the legacy engine that must later be removed. |
| Deliver setup, reconciliation, resolution, and UI in TASK-97 | Rejected. The cross-authority safety contract is too large for one independently reviewable PR. |
| Create a second lasting-sync SQLite file | Rejected. The registered `notes.sync_state` owner already reserves this purpose; another owner would duplicate private lifecycle and backup policy. |
| Let both repositories manage `PRAGMA user_version` | Rejected. Independent version owners race and make initialization order observable. |
| Canonicalize paths during migration | Rejected. Migration is read-only source capture; filesystem authority belongs to guarded dry-run and capability admission. |
| Convert legacy conflicts into new conflict records | Rejected. Legacy records lack the exact observations and recovery admission required by ADR-059/060. |
| Keep the legacy and new engines active together | Rejected by ADR-059. The activation slice must cut over before granting new mutation authority. |
| Add configurable limits now | Rejected as unnecessary surface area. Fixed safety ceilings are sufficient for the storage foundation. |

## Open Questions Deferred to Later Slices

The following are deliberately not decided by TASK-97's implementation plan:

- platform-specific path-key normalization and filesystem capability policy;
- lease duration, heartbeat cadence, watcher implementation, and passive-owner
  presentation;
- exact conflict snapshot byte limits and recovery capacity accounting;
- journal operation/state taxonomy and Undo retention mechanics;
- Textual queue/modal composition, focus, and activity-log copy; and
- the versioned server capability and claim contract.

Those decisions must remain consistent with ADR-059/060 and be specified in the
atomic task that first needs them.
