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

A small Notes-private schema module is the only code allowed to create or
upgrade the `notes.sync_state` database. It imports neither receipt-domain nor
sync-domain models. Both repositories call it after opening their connection
through `connect_private_sqlite`.

Initialization must:

1. open the existing private connection with the owner's established pragmas;
2. enter `BEGIN IMMEDIATE` before deciding whether migration is required;
3. re-read `PRAGMA user_version` after the writer lock is held;
4. create the complete v2 schema for a new database or migrate v1 to v2
   additively;
5. reject unknown future or malformed supported versions without destructive
   repair;
6. set `user_version = 2` only after every required table, index, and constraint
   exists; and
7. commit atomically or roll back completely.

The lock-and-reread sequence is required. Checking the version before acquiring
the writer lock would allow two real connections to both choose a migration
path from stale state.

The existing receipt tables, columns, indexes, rows, and lifecycle semantics are
not rebuilt or transformed. The compatibility promise is behavioral: existing
receipt projections, transitions, retry behavior, and data remain unchanged.
Whole-database byte identity is not promised because SQLite may legitimately
change pages while adding schema objects.

### 2. Lasting-sync repository

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

### 3. Paused candidate roots

A root record contains at least:

- opaque root ID;
- private lexical root path as supplied by the legacy source;
- bounded display name;
- requested direction;
- lifecycle state, initially `candidate` or `paused`;
- optimistic row version;
- nullable verified canonical path identity;
- nullable verified filesystem identity/capability fields;
- bounded reason code and `needs_rescan` marker where applicable; and
- created/updated timestamps.

TASK-97 never fills verified identity fields. They remain null until the later
setup task performs the approved guarded dry-run. A lexical path is migration
input, not proof of containment, existence, equivalence, writability, or safe
ownership.

No root created by this slice can enter an active/running state. The repository
must reject such a transition because there is no coordinator or admission
proof yet.

### 4. Provisional bindings

A binding record contains at least:

- opaque binding ID;
- owning root ID;
- private Database Note ID;
- private lexical root-relative path;
- nullable portability-normalized `path_key`;
- lifecycle state such as `candidate`, `bound`, `needs_attention`, or
  `disconnected`;
- optimistic row version;
- nullable verified file identity;
- nullable representation profile;
- nullable last observations; and
- bounded reason code / provisional marker.

Verified identity, representation, and observations remain null in TASK-97.
Legacy values may be retained as explicitly provisional source observations,
but they never satisfy activation or mutation preconditions.

### 5. Ownership and uniqueness

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

Partial unique indexes should express these invariants in SQLite rather than
relying only on preflight queries.

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
read boundaries. It performs no source-path resolution, filesystem I/O, note
write, configuration write, or legacy-row cleanup.

Because ChaChaNotes and `notes.sync_state` are separate SQLite owners, their
work cannot be one transaction. The migration therefore:

1. opens a stable read transaction on the main Notes authority;
2. reads a bounded source projection containing only the fields required to
   group lexical roots and candidate bindings;
3. derives a private source-revision digest from that projection;
4. writes candidates and a migration receipt atomically in `notes.sync_state`;
5. records whether the source revision stayed stable through capture; and
6. treats drift as a provisional revision requiring rescan, never as authority.

Migration replay with the same source revision is a no-op. A new revision may
add or update only the corresponding provisional migration generation; it
cannot silently activate, delete, or steal an existing non-disconnected
binding.

### Grouping and malformed inputs

Recognizable legacy rows are grouped by stored lexical root into paused
candidate roots and provisional bindings. They are not described as canonical
or safe. A missing, nonexistent, relative, adversarial, or platform-foreign path
may still be recorded as bounded private lexical text for later review.

One malformed root or binding does not block independent siblings. The item
records a bounded reason code, such as malformed metadata, duplicate note
ownership, out-of-contract relative path, or capacity exceeded. Raw exception
text and physical paths are never persisted as error messages or logged.

Legacy conflict rows are not converted into actionable new conflict records.
Their content and observations cannot satisfy the new exact-side and recovery
contract. Migration sets only `needs_rescan`; the later reconciler must
rediscover and capture both current sides through the new boundary.

### No-filesystem guarantee

The migration layer treats stored paths as data. It must not call `resolve`,
`absolute`, `stat`, `lstat`, `open`, directory iteration, or an equivalent
filesystem seam. A focused test will use nonexistent and adversarial stored
paths while patching those operations to raise immediately; candidate migration
must still complete using lexical text alone.

## Capacity and Bounds

The foundation uses fixed safety constants rather than adding premature user
configuration:

- `MAX_SYNC_ROOTS = 64` live non-disconnected roots per profile; and
- `MAX_IMPORT_ENTRIES = 100_000` total live non-disconnected lasting-sync
  bindings per profile, reusing the established Notes import discovery ceiling.

Capacity validation occurs before writes. A rejected batch is atomic: it creates
no partial roots, bindings, or completion receipt. Error objects and messages
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
- A capacity or uniqueness failure rolls back the whole requested operation.
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

### Root and binding contracts

- Cover immutable/redacted typed projections and error messages.
- Prove non-disconnected note ownership is global across roots, including
  paused roots, and disconnected rows release that ownership.
- Prove conditional path-key uniqueness and allow null provisional keys until
  dry-run.
- Prove optimistic version checks and atomic batch rejection.
- Cover exact limits at 64 roots and 100,000 bindings plus one-over rejection
  without constructing unbounded payload content.

### Legacy migration

- Use a real ChaChaNotes fixture with recognizable legacy configuration and
  per-note metadata; do not handwrite a partial schema and call it current.
- Cover multiple lexical roots, duplicate note ownership, malformed siblings,
  legacy conflict markers, idempotent replay, and a changed source revision.
- Simulate cross-owner drift and prove the resulting generation remains
  provisional and marked for rescan.
- Patch filesystem operations to fail and use nonexistent/adversarial stored
  paths; prove migration still succeeds lexically.
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
   paused candidates without filesystem, file, note, config, or legacy-row
   mutation; drift, conflicts, and malformed items remain provisional and
   require rescan.
4. **Bounds and privacy:** exact root/binding ceilings reject atomically, and
   physical identifiers remain confined to the backup-excluded
   `notes.sync_state` owner and absent from diagnostics.
5. **No premature authority:** this slice provides no activation, watcher,
   reconciliation, conflict-content, resolver, journal, UI, or server-backed
   sync behavior, and the legacy engine remains the only active owner.

## ADR Check

ADR required: no

ADR path: N/A (existing ADR-059 and ADR-060 govern this design)

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
