# File Notes A0 Storage Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a deterministic, owner-protected `file_notes.db` boundary and a one-shot, post-Database-paint inspection path without changing or coupling the existing ChaChaNotes store.

**Architecture:** A dedicated `tldw_chatbook.Notes.file_notes` package owns pure storage layout, the version-1 projection schema, explicit-only fresh bootstrap, and an app-lifetime passive startup probe. `TldwCli` constructs only pure path/probe state; `LibraryScreen` may claim the probe only after the existing Database Notes canvas has mounted. A pristine namespace performs no SQLite open and starts no File Notes worker. All failures remain typed Files-source results and never enter the existing Library snapshot/error pipeline.

**Tech Stack:** Python 3.11+, stdlib `hashlib`/`pathlib`/`sqlite3`/`threading`, SQLite FTS5, Textual workers and `call_after_refresh`, pytest/pytest-asyncio.

## Global Constraints

- Implement only [TASK-399.1](../../../backlog/tasks/task-399.1%20-%20A0-Isolate-file-note-projection-storage.md). Root preview, Link UI, projection population, indexing workers, watchers, leases for active coordination, file editing, and recovery-store creation remain later tasks.
- Do not modify `tldw_chatbook/DB/base_db.py`, `tldw_chatbook/DB/ChaChaNotes_DB.py`, `NotesInteropService`, `NotesScopeService`, existing note CRUD/sync, `initialize_all_databases()`, global lazy DB getters, or Settings backup/restore maps.
- Do not subclass `BaseDB` or `CharactersRAGDB`. Their constructors create directories/schema, open writable SQLite, wait on locks, manage WAL, and log absolute paths.
- Do not add File Notes work to `LibraryScreen._list_local_source_snapshot()`. Its shared timeout/error state can replace healthy Database Notes.
- The normal startup path must never call the explicit fresh-bootstrap method. TASK-399.2 becomes its first production caller, after confirmed Link.
- A pristine profile may perform an exact-path `lstat` evidence check after Database Notes paint, but it creates no directory, opens no SQLite connection, and starts no File Notes worker, watcher, scan, or lease.
- An existing projection DB gets at most one automatic query per app/profile lifetime. Reconstructed Library screens reuse the app-owned result/latch.
- The 100 ms budget is a result-publication deadline as well as a SQLite deadline: use `timeout=0`, `PRAGMA busy_timeout=0`, a SQLite progress handler, an outer async deadline, and a claim token that rejects late thread completion.
- Do not use SQLite `immutable=1`; it can ignore committed WAL content. Do not set journal mode, checkpoint, migrate, repair, rename, delete, or rebuild during passive inspection.
- Routine logging and diagnostics may include stable codes and the non-secret storage-instance ID, but never absolute storage/root paths, note bodies, hashes, SQLite exception text, or recovery payloads.
- Preserve unrelated working-tree changes. Execute product work in an isolated `codex/file-notes-a0-storage-isolation` worktree.

### ADR check

```text
ADR required: yes
ADR path: backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
Reason: A0 establishes an independently versioned storage/schema, runtime namespace, failure boundary, and bootstrap policy. ADR-021 already records the approved decision, so no duplicate ADR is needed.
```

---

## Task 1: Lock the approved design and add pure storage layouts

**Files:**

- Modify: `Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md`
- Modify: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`
- Create: `tldw_chatbook/Notes/file_notes/__init__.py`
- Create: `tldw_chatbook/Notes/file_notes/storage.py`
- Create: `Tests/Notes/File_Notes/__init__.py`
- Create: `Tests/Notes/File_Notes/test_storage.py`

### Step 1: Record the approval

- [x] Change the design status to `Approved 2026-07-23; A0 implementation planning complete`.
- [x] Change ADR-021 from `Proposed` to `Accepted`.
- [x] Do not alter the approved decision text while changing governance state.

### Step 2: Write failing identity/layout tests

- [ ] Add tests proving:

  1. equivalent canonical main-DB paths produce the same full 64-character lowercase SHA-256 ID;
  2. different main-DB paths produce different IDs and instance directories;
  3. changing user-data roots relocates every DB/sidecar/marker/diagnostic path without changing the storage ID;
  4. the runtime namespace is unchanged when user-data/main-DB/repository/cache/log paths change;
  5. layout construction is pure and leaves `user_data_dir / "file_notes"` absent;
  6. SQLite-reserved characters, spaces, Unicode, and symlinked configured DB paths canonicalize safely;
  7. neither the ID nor `repr()` of public result objects contains an absolute main-DB path.

Use this domain separator and retain the full digest:

```python
STORAGE_INSTANCE_DOMAIN = (
    b"tldw-chatbook:file-notes-storage-instance:v1\0"
)
```

The fixed byte-vector assertion is:

```python
assert (
    hashlib.sha256(
        STORAGE_INSTANCE_DOMAIN + b"/srv/chatbook/main.db"
    ).hexdigest()
    == "d16a123799c72bf1c93986d9eb98c60580264b4b00fe7de563262f10cb1b92d4"
)
```

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_storage.py
```

Expected: FAIL because `tldw_chatbook.Notes.file_notes.storage` does not exist.

### Step 3: Implement pure storage and runtime layouts

- [ ] Add these stable contracts to `storage.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


STORAGE_INSTANCE_DOMAIN = (
    b"tldw-chatbook:file-notes-storage-instance:v1\0"
)
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class EvidenceEntryState(StrEnum):
    ABSENT = "absent"
    REGULAR_FILE = "regular_file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    SPECIAL = "special"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FileNotesStorageLayout:
    storage_instance_id: str
    instance_dir: Path
    projection_db: Path
    projection_wal: Path
    projection_shm: Path
    recovery_db: Path
    recovery_wal: Path
    recovery_shm: Path
    bootstrap_marker: Path
    diagnostics_dir: Path


@dataclass(frozen=True, slots=True)
class FileNotesRuntimeLayout:
    namespace_dir: Path
    bootstrap_lock: Path
    coordinator_lock: Path
    mutation_lock: Path
```

- [ ] Implement these pure functions:

```python
def canonical_main_database_path(path: str | Path) -> Path:
    """Return one absolute, real, platform-normalized configured DB path."""


def derive_storage_instance_id(main_database_path: str | Path) -> str:
    """Hash canonical filesystem bytes with STORAGE_INSTANCE_DOMAIN."""


def resolve_file_notes_storage_layout(
    *,
    user_data_dir: Path,
    main_database_path: Path,
) -> FileNotesStorageLayout:
    """Calculate <user-data>/file_notes/<storage-id>/ without creating it."""


def resolve_file_notes_runtime_layout(
    *,
    storage_instance_id: str,
    runtime_base_override: Path | None = None,
) -> FileNotesRuntimeLayout:
    """Calculate fixed per-user runtime paths without creating them."""
```

Canonicalization must use `expanduser()`, `resolve(strict=False)`, `os.path.normcase()`, and `os.fsencode()` at the hash boundary. Do not hash raw TOML text and do not place any original path component in a filename.

The default runtime base is OS-owned, not app-configured:

1. use an absolute `XDG_RUNTIME_DIR` when available on POSIX;
2. otherwise use `tempfile.gettempdir()`;
3. add a per-user token (`os.getuid()` where available, otherwise a SHA-256 token over the canonical home path);
4. append `tldw-chatbook/file-notes`;
5. keep `coordinator.lock` and `mutation.lock` global to that per-user namespace;
6. keep `bootstrap-<storage-instance-id>.lock` instance-specific.

`runtime_base_override` is test-only dependency injection. Production callers must omit it.

### Step 4: Add exact, non-mutating evidence inspection

- [ ] Add:

```python
@dataclass(frozen=True, slots=True)
class FileNotesEvidence:
    instance_dir: EvidenceEntryState
    projection_db: EvidenceEntryState
    projection_wal: EvidenceEntryState
    projection_shm: EvidenceEntryState
    recovery_db: EvidenceEntryState
    recovery_wal: EvidenceEntryState
    recovery_shm: EvidenceEntryState
    bootstrap_marker: EvidenceEntryState
    diagnostics_dir: EvidenceEntryState

    @property
    def is_pristine(self) -> bool:
        return (
            self.instance_dir
            in (EvidenceEntryState.ABSENT, EvidenceEntryState.DIRECTORY)
            and all(
                state is EvidenceEntryState.ABSENT
                for state in (
                    self.projection_db,
                    self.projection_wal,
                    self.projection_shm,
                    self.recovery_db,
                    self.recovery_wal,
                    self.recovery_shm,
                    self.bootstrap_marker,
                    self.diagnostics_dir,
                )
            )
        )


def inspect_file_notes_evidence(
    layout: FileNotesStorageLayout,
) -> FileNotesEvidence:
    """Classify only the fixed evidence paths with lstat; change nothing."""
```

Inspect `instance_dir` first. If it is a symlink, special file, or unavailable,
return that state without traversing through it. For a real directory, inspect
children relative to a no-follow directory handle where the platform supports
`dir_fd`; on Windows, reject reparse points before child inspection. Broken
symlinks count as `SYMLINK`. Permission errors become `UNAVAILABLE`. An empty,
ordinary instance directory is still pristine. Do not call `mkdir`, `chmod`,
`resolve`, SQLite, logging with a path, or cleanup from this function.

### Step 5: Verify and commit

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_storage.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes
git diff --check
```

Expected: all pass with no output from `compileall` or `git diff --check`.

- [ ] Commit:

```bash
git add Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md tldw_chatbook/Notes/file_notes/__init__.py tldw_chatbook/Notes/file_notes/storage.py Tests/Notes/File_Notes/__init__.py Tests/Notes/File_Notes/test_storage.py
git commit -m "feat(notes): define isolated file notes storage layout"
```

---

## Task 2: Define the independent version-1 schema and explicit fresh bootstrap

**Files:**

- Create: `tldw_chatbook/Notes/file_notes/schema.py`
- Create: `tldw_chatbook/Notes/file_notes/private_paths.py`
- Create: `tldw_chatbook/Notes/file_notes/bootstrap.py`
- Modify: `tldw_chatbook/Notes/file_notes/__init__.py`
- Create: `Tests/Notes/File_Notes/test_schema.py`
- Create: `Tests/Notes/File_Notes/test_private_paths.py`
- Create: `Tests/Notes/File_Notes/test_bootstrap.py`

### Step 1: Write failing schema/bootstrap tests

- [ ] Cover:

  1. fresh initialization is possible only through `initialize_after_confirmed_first_link()`;
  2. it creates only the selected storage instance;
  3. the singleton identity and `PRAGMA user_version` both equal schema version 1;
  4. roots, projections, indexes, FTS, and FTS shadow tables exist only in `file_notes.db`;
  5. no projection or FTS trigger exists;
  6. `file_note_projections` retains an integer rowid compatible with external-content FTS;
  7. instance/runtime directories are owner-only and DB/lock/sidecars are owner-only;
  8. DB, WAL, SHM, diagnostics, marker, and lock paths never overlap another storage instance;
  9. a symlinked directory/file, special file, wrong owner, or unsafe permission fails closed;
  10. DB/WAL/SHM/recovery/marker evidence blocks initialization without changing bytes, names, mtimes, or sidecars;
  11. truncated/corrupt/partial DB evidence is preserved rather than renamed or rebuilt;
  12. an unexpected entry inside the internal instance directory blocks initialization and is preserved;
  13. two simultaneous bootstrap contenders produce one `CREATED` and one `BUSY`, and a later retry by the loser returns `EVIDENCE_PRESENT`.

On POSIX, assert exact `0700` directories and `0600` regular files. On
Windows, assert the paths remain beneath the current user's data/runtime bases,
reject reparse points, and prove the security descriptor grants access only to
the current user, LocalSystem, and Builtin Administrators. Do not treat POSIX
mode bits or `os.chmod` as a Windows ACL proof.

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_schema.py Tests/Notes/File_Notes/test_private_paths.py Tests/Notes/File_Notes/test_bootstrap.py
```

Expected: FAIL because schema/bootstrap modules do not exist.

### Step 2: Add the exact schema

- [ ] Put `FILE_NOTES_SCHEMA_VERSION = 1` and the following DDL in `schema.py`. Execute statements individually inside one explicit transaction; do not use a BaseDB constructor or a migration registry.

```sql
CREATE TABLE file_notes_storage (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
    storage_instance_id TEXT NOT NULL CHECK (length(storage_instance_id) = 64),
    feature_state TEXT NOT NULL DEFAULT 'ready'
        CHECK (feature_state IN ('ready', 'disabled', 'attention')),
    recovery_instance_uuid TEXT,
    bootstrap_generation INTEGER NOT NULL DEFAULT 0
        CHECK (bootstrap_generation >= 0),
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE note_file_roots (
    root_uuid TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    canonical_path TEXT NOT NULL UNIQUE,
    filesystem_identity_json TEXT,
    runtime_mode TEXT NOT NULL DEFAULT 'read_only'
        CHECK (runtime_mode IN ('disabled', 'read_only', 'read_write')),
    lifecycle_state TEXT NOT NULL DEFAULT 'detached'
        CHECK (
            lifecycle_state IN (
                'preview', 'active', 'offline', 'detached', 'attention'
            )
        ),
    path_comparison_policy TEXT NOT NULL DEFAULT 'unknown'
        CHECK (
            path_comparison_policy IN (
                'unknown', 'case_sensitive', 'case_insensitive'
            )
        ),
    read_capabilities_json TEXT NOT NULL DEFAULT '{}',
    write_capabilities_json TEXT NOT NULL DEFAULT '{}',
    protection_prefixes_json TEXT NOT NULL DEFAULT '[]',
    activation_generation INTEGER NOT NULL DEFAULT 0
        CHECK (activation_generation >= 0),
    scan_generation INTEGER NOT NULL DEFAULT 0
        CHECK (scan_generation >= 0),
    observation_generation INTEGER NOT NULL DEFAULT 0
        CHECK (observation_generation >= 0),
    metadata_verification_generation INTEGER NOT NULL DEFAULT 0
        CHECK (metadata_verification_generation >= 0),
    raw_verification_generation INTEGER NOT NULL DEFAULT 0
        CHECK (raw_verification_generation >= 0),
    reconcile_deadline_at TEXT,
    last_scan_at TEXT,
    last_observed_at TEXT,
    diagnostic_code TEXT,
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_note_file_roots_lifecycle
    ON note_file_roots(lifecycle_state, runtime_mode);

CREATE TABLE file_note_projections (
    projection_id INTEGER PRIMARY KEY,
    note_uuid TEXT NOT NULL UNIQUE,
    root_uuid TEXT NOT NULL
        REFERENCES note_file_roots(root_uuid)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    relative_path TEXT NOT NULL,
    path_comparison_key TEXT NOT NULL,
    title TEXT NOT NULL,
    editable_body TEXT,
    raw_hash_algorithm TEXT NOT NULL DEFAULT 'sha256',
    raw_hash TEXT,
    semantic_hash TEXT,
    presence_state TEXT NOT NULL DEFAULT 'present'
        CHECK (presence_state IN ('present', 'missing', 'tombstoned')),
    body_eligible INTEGER NOT NULL DEFAULT 0
        CHECK (body_eligible IN (0, 1)),
    observed_size_bytes INTEGER CHECK (observed_size_bytes >= 0),
    observed_mtime_ns INTEGER,
    observed_mode_bits INTEGER,
    bom_kind TEXT NOT NULL DEFAULT 'none',
    newline_kind TEXT NOT NULL DEFAULT 'none',
    has_final_newline INTEGER
        CHECK (has_final_newline IS NULL OR has_final_newline IN (0, 1)),
    file_identity_json TEXT,
    observed_link_count INTEGER CHECK (observed_link_count >= 0),
    security_facts_json TEXT,
    security_fingerprint TEXT,
    frontmatter_mode TEXT NOT NULL DEFAULT 'none',
    format_diagnostic_code TEXT,
    read_only_diagnostic_code TEXT,
    projection_generation INTEGER NOT NULL DEFAULT 0
        CHECK (projection_generation >= 0),
    observed_at TEXT,
    body_index_state TEXT NOT NULL DEFAULT 'suppressed'
        CHECK (
            body_index_state IN (
                'pending', 'current', 'suppressed', 'paused', 'error'
            )
        ),
    indexed_semantic_hash TEXT,
    indexed_generation INTEGER NOT NULL DEFAULT 0
        CHECK (indexed_generation >= 0),
    path_metadata_generation INTEGER NOT NULL DEFAULT 0
        CHECK (path_metadata_generation >= 0),
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (root_uuid, path_comparison_key)
);

CREATE INDEX idx_file_note_projection_browse
    ON file_note_projections(root_uuid, presence_state, path_comparison_key);

CREATE INDEX idx_file_note_projection_index_state
    ON file_note_projections(body_index_state, indexed_generation);

CREATE INDEX idx_file_note_projection_title
    ON file_note_projections(title);

CREATE VIRTUAL TABLE file_notes_fts USING fts5(
    title,
    editable_body,
    content='file_note_projections',
    content_rowid='projection_id',
    tokenize='unicode61'
);
```

After the DDL and parameterized singleton insert, set:

```sql
PRAGMA user_version = 1;
```

The singleton insert must bind the derived storage ID:

```python
connection.execute(
    """
    INSERT INTO file_notes_storage (
        singleton_id,
        schema_version,
        storage_instance_id,
        feature_state,
        recovery_instance_uuid,
        bootstrap_generation
    ) VALUES (1, ?, ?, 'ready', NULL, 0)
    """,
    (FILE_NOTES_SCHEMA_VERSION, storage_instance_id),
)
```

Do not add FTS triggers. Later indexing code owns FTS publication explicitly.

### Step 3: Implement owner-protected explicit bootstrap

- [ ] Add `private_paths.py` with one platform-dispatched boundary:

```python
class PrivatePathSecurityError(OSError):
    """An internal File Notes path could not be proven owner-protected."""


def ensure_private_directory(path: Path) -> None:
    """Create or verify one no-follow owner-protected internal directory."""


def ensure_private_file(path: Path) -> None:
    """Verify and, when owned, tighten one internal regular file."""
```

On POSIX, use `lstat`, reject symlinks/non-directories/non-regular files,
require `st_uid == os.geteuid()`, apply `0700`/`0600`, and re-read the stat
after chmod. Use `dir_fd` plus `O_NOFOLLOW` where available.

On Windows, implement a focused stdlib `ctypes` wrapper over
`GetFileAttributesW`, `GetNamedSecurityInfoW`, `GetAce`,
`OpenProcessToken`/`GetTokenInformation(TokenUser)`, `EqualSid`, and
`ConvertStringSidToSidW`, `SetEntriesInAclW`, and `SetNamedSecurityInfoW`.
Reject any reparse point. Build a protected DACL granting full control only to
the current token SID, LocalSystem (`S-1-5-18`), and Builtin Administrators
(`S-1-5-32-544`); directory ACEs inherit to child containers and objects.
Install it with `PROTECTED_DACL_SECURITY_INFORMATION`, then re-read and verify
every allow ACE. Fail closed on null/invalid DACLs, unrecognized allow
trustees, or any Win32 error. Free ACLs, converted SIDs, and native security
descriptors with `LocalFree`, and close token handles in `finally`. Keep all
native constants and handle cleanup inside `private_paths.py`; return only
sanitized error codes to callers.

- [ ] Add path-safe result types:

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class FileNotesBootstrapStatus(StrEnum):
    CREATED = "created"
    EVIDENCE_PRESENT = "evidence_present"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    FAILED_PRESERVED = "failed_preserved"


@dataclass(frozen=True, slots=True)
class FileNotesDiagnostic:
    code: str
    message: str
    source: Literal["files"] = "files"


@dataclass(frozen=True, slots=True)
class FileNotesBootstrapResult:
    status: FileNotesBootstrapStatus
    diagnostic: FileNotesDiagnostic | None
```

- [ ] Implement:

```python
class FileNotesStoreBootstrap:
    def __init__(
        self,
        *,
        storage: FileNotesStorageLayout,
        runtime: FileNotesRuntimeLayout,
    ) -> None:
        self._storage = storage
        self._runtime = runtime

    def initialize_after_confirmed_first_link(
        self,
    ) -> FileNotesBootstrapResult:
        """Exclusively create one fresh A-stage projection store."""
```

The method must perform this exact order:

1. create/verify the fixed runtime namespace and its parent chain without following a symlink;
2. open the owner-only bootstrap lock and acquire a non-blocking OS lock (`fcntl.flock` on POSIX, one-byte `msvcrt.locking` on Windows);
3. create/verify `<user-data>/file_notes/<storage-id>/` as owner-only without following a symlink;
4. re-run fixed-path evidence inspection while holding the lock and inventory the internal instance directory for unexpected entries;
5. return `EVIDENCE_PRESENT` on any DB/sidecar/recovery/marker/diagnostic or unexpected internal evidence;
6. exclusively create `file_notes.db` with `os.open(O_CREAT | O_EXCL | O_RDWR)` plus `O_NOFOLLOW`/`O_CLOEXEC` where supported and mode `0600`;
7. open that already-created file with SQLite, set `foreign_keys=ON`, set `journal_mode=WAL`, and immediately verify/tighten the DB plus any live WAL/SHM sidecars;
8. install DDL plus identity in one explicit `BEGIN IMMEDIATE` transaction, then verify/tighten live sidecars again before continuing;
9. verify `PRAGMA user_version`, singleton identity, required tables/indexes, no triggers, and `PRAGMA quick_check(1)`;
10. close SQLite in `finally`;
11. recheck every DB/sidecar that survives close as owner-only;
12. release the OS lock in `finally`.

If any failure occurs after exclusive DB creation, close handles and return `FAILED_PRESERVED`; do not unlink, truncate, rename, reinitialize, or hide the partial database. Never log the caught exception object.

### Step 4: Verify and commit

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_schema.py Tests/Notes/File_Notes/test_private_paths.py Tests/Notes/File_Notes/test_bootstrap.py
pytest -q Tests/Notes/File_Notes/test_storage.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes
git diff --check
```

Expected: all pass.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes/__init__.py tldw_chatbook/Notes/file_notes/schema.py tldw_chatbook/Notes/file_notes/private_paths.py tldw_chatbook/Notes/file_notes/bootstrap.py Tests/Notes/File_Notes/test_schema.py Tests/Notes/File_Notes/test_private_paths.py Tests/Notes/File_Notes/test_bootstrap.py
git commit -m "feat(notes): add explicit file notes store bootstrap"
```

---

## Task 3: Add the app-lifetime, read-only startup probe

**Files:**

- Create: `tldw_chatbook/Notes/file_notes/startup_probe.py`
- Modify: `tldw_chatbook/Notes/file_notes/__init__.py`
- Create: `Tests/Notes/File_Notes/test_startup_probe.py`

### Step 1: Write failing probe tests

- [ ] Cover:

  1. pristine evidence publishes `ABSENT` without `sqlite3.connect`;
  2. missing projection plus WAL/SHM/recovery/marker evidence publishes `RETAINED_EVIDENCE` without creating/opening a DB;
  3. a healthy DB performs one evidence query through a correctly encoded `file:` URI with `mode=ro`, `uri=True`, and `timeout=0.0`;
  4. the connection sets `query_only=ON`, `busy_timeout=0`, and closes in `finally`;
  5. root counts distinguish `HEALTHY_UNPAIRED`, `ACTIVE`, and `DETACHED`;
  6. incompatible schema, storage-ID mismatch, missing schema rows/tables, corruption, permission errors, and special/symlink evidence return typed Files diagnostics;
  7. a real exclusive SQLite lock returns `BUSY` without waiting;
  8. an expired progress-handler deadline returns `TIMED_OUT`;
  9. an outer timeout/cancellation invalidates the claim so late thread completion cannot publish;
  10. repeated claims and reconstructed screen consumers never cause a second query;
  11. DB, WAL, recovery, and marker bytes/names/mtimes remain unchanged; a pre-existing SHM sidecar is never app-renamed, app-deleted, or app-truncated, while SQLite coordination-page/mtime changes are explicitly allowed;
  12. captured logs and diagnostic `repr()` omit a seeded absolute root, note body, raw hash, and SQLite exception text.

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_startup_probe.py
```

Expected: FAIL because `startup_probe.py` does not exist.

### Step 2: Implement typed one-shot state

- [ ] Add:

```python
class FileNotesProbeStatus(StrEnum):
    ABSENT = "absent"
    HEALTHY_UNPAIRED = "healthy_unpaired"
    ACTIVE = "active"
    DETACHED = "detached"
    RETAINED_EVIDENCE = "retained_evidence"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"
    STORAGE_MISMATCH = "storage_mismatch"
    PARTIAL = "partial"
    BUSY = "busy"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FileNotesProbeResult:
    status: FileNotesProbeStatus
    active_root_count: int = 0
    detached_root_count: int = 0
    diagnostic: FileNotesDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class FileNotesProbeClaim:
    generation: int
    started_at: float


@dataclass(frozen=True, slots=True)
class FileNotesProbeDecision:
    claim: FileNotesProbeClaim | None
    result: FileNotesProbeResult | None
```

- [ ] `FileNotesStartupProbe` owns a `threading.Lock`, lifecycle (`pending`, `running`, `done`), generation, and final result. Its constructor and `for_profile()` factory perform only path calculations.

```python
class FileNotesStartupProbe:
    @classmethod
    def for_profile(
        cls,
        *,
        user_data_dir: Path,
        main_database_path: Path,
        runtime_base_override: Path | None = None,
    ) -> "FileNotesStartupProbe":
        """Build pure app-lifetime state; do not inspect or create paths."""

    def claim_if_existing_evidence(self) -> FileNotesProbeDecision:
        """Claim once, or finish synchronously when no DB query is allowed."""

    def inspect_claim(
        self,
        claim: FileNotesProbeClaim,
        *,
        budget_seconds: float = 0.100,
    ) -> FileNotesProbeResult:
        """Run one bounded read-only query and always close its connection."""

    def publish(
        self,
        claim: FileNotesProbeClaim,
        result: FileNotesProbeResult,
    ) -> bool:
        """Publish only if this claim is still current."""

    def cancel(self, claim: FileNotesProbeClaim) -> None:
        """Finish the automatic probe as cancelled and reject late results."""
```

`claim_if_existing_evidence()` may use only fixed-path evidence inspection. It returns `ABSENT` and marks done when pristine. It returns a synchronous retained/orphan diagnostic when the projection DB is absent or unsafe to open. It returns a claim only for one regular, existing projection DB.

### Step 3: Implement the single detached-evidence query

- [ ] Build the read-only URI with `Path.as_uri()` and append `?mode=ro`; do not interpolate an unescaped filesystem string.
- [ ] Set the SQLite progress handler before the evidence query and check `time.monotonic()` against the claim deadline.
- [ ] Execute this one evidence query:

```sql
SELECT
    pragma_version.user_version AS user_version,
    storage.schema_version AS schema_version,
    storage.storage_instance_id AS storage_instance_id,
    storage.recovery_instance_uuid AS recovery_instance_uuid,
    storage.bootstrap_generation AS bootstrap_generation,
    COUNT(roots.root_uuid) AS root_count,
    COALESCE(
        SUM(CASE WHEN roots.lifecycle_state = 'active' THEN 1 ELSE 0 END),
        0
    ) AS active_root_count,
    COALESCE(
        SUM(CASE WHEN roots.lifecycle_state = 'detached' THEN 1 ELSE 0 END),
        0
    ) AS detached_root_count
FROM file_notes_storage AS storage
CROSS JOIN pragma_user_version AS pragma_version
LEFT JOIN note_file_roots AS roots ON 1 = 1
WHERE storage.singleton_id = 1
GROUP BY
    pragma_version.user_version,
    storage.schema_version,
    storage.storage_instance_id,
    storage.recovery_instance_uuid,
    storage.bootstrap_generation
```

The setup PRAGMAs are not evidence queries:

```python
connection.execute("PRAGMA query_only = ON")
connection.execute("PRAGMA busy_timeout = 0")
```

Require both `user_version` and `schema_version` to equal
`FILE_NOTES_SCHEMA_VERSION`; disagreement or any other version is
`INCOMPATIBLE`. Map only stable exception categories. `locked`/`busy` becomes
`BUSY`; malformed/not-a-database becomes `CORRUPT`; missing expected
objects/row becomes `PARTIAL`; deadline interruption becomes `TIMED_OUT`;
permission/URI/open failure becomes `UNAVAILABLE`. Never return or log
`str(exc)`.

If recovery DB/sidecar/marker evidence exists, do not open the recovery store. The projection query may identify retained roots/pairing, but the result is `RETAINED_EVIDENCE` with an A-stage upgrade diagnostic.

### Step 4: Verify and commit

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_startup_probe.py
pytest -q Tests/Notes/File_Notes/test_storage.py Tests/Notes/File_Notes/test_schema.py Tests/Notes/File_Notes/test_bootstrap.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes
git diff --check
```

Expected: all pass.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes/__init__.py tldw_chatbook/Notes/file_notes/startup_probe.py Tests/Notes/File_Notes/test_startup_probe.py
git commit -m "feat(notes): add bounded file notes startup probe"
```

---

## Task 4: Wire the probe after the real Database Notes first paint

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Create: `Tests/Notes/File_Notes/test_app_wiring.py`

### Step 1: Write failing pure app-wiring tests

- [ ] In `test_app_wiring.py`, prove:

  1. a file-backed `chachanotes_db` gets one app-owned `FileNotesStartupProbe`;
  2. an absent/in-memory ChaChaNotes DB gets no unstable file profile;
  3. app construction creates no `file_notes` or runtime subtree and opens no File Notes SQLite connection;
  4. the app-owned probe survives reconstructed `LibraryScreen` instances;
  5. File Notes is absent from `initialize_all_databases()` and existing backup/restore maps.

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_app_wiring.py
```

Expected: FAIL because `TldwCli` has no `file_notes_startup_probe`.

### Step 2: Add pure app-owned probe construction

- [ ] In `TldwCli.__init__`, immediately after the existing `self.chachanotes_db` assignment, add a helper-backed field:

```python
self.file_notes_startup_probe = self._build_file_notes_startup_probe()
```

```python
def _build_file_notes_startup_probe(
    self,
) -> FileNotesStartupProbe | None:
    database = getattr(self, "chachanotes_db", None)
    if database is None or bool(getattr(database, "is_memory_db", False)):
        return None
    database_path = getattr(database, "db_path", None)
    if not isinstance(database_path, Path):
        return None
    try:
        return FileNotesStartupProbe.for_profile(
            user_data_dir=get_user_data_dir(),
            main_database_path=database_path,
        )
    except (OSError, ValueError):
        return FileNotesStartupProbe.unavailable()
```

Add an `unavailable()` classmethod that carries only a safe Files diagnostic and performs no filesystem work.

Do not add this object to eager startup tasks, service pools, shutdown barriers, DB registries, Settings backups, or generic worker management. It owns no open handle.

### Step 3: Write failing post-paint Textual tests

- [ ] Extend the existing `LibraryHarness` tests in `Tests/UI/test_library_shell.py` to prove:

  1. a pristine probe decision occurs only after `#library-notes-canvas` is mounted and calls no File Notes query worker;
  2. an existing-DB query starts strictly after Database Notes rows/actions are visible;
  3. both a fresh snapshot and cached repeat-visit snapshot preserve that ordering;
  4. mounting Landing/Media does not claim the probe; selecting Browse Notes later does;
  5. a locked/corrupt/incompatible/mismatched result leaves Database rows/actions usable;
  6. File failure never sets `_library_lookup_error`, mounts `#library-canvas-error`, or changes `_library_loaded`;
  7. snapshot refreshes/recompositions do not repeat the query;
  8. popping and pushing a fresh Library screen on the same app does not repeat the query;
  9. an outer 100 ms timeout publishes `TIMED_OUT`, invalidates the claim, and ignores a gated thread's later return;
  10. no watcher, scan, indexer, coordinator election, mutation lease, recovery writer, or long-lived SQLite connection remains.

Use a bounded `threading.Event` gate and release it in `finally`, matching the existing anti-hang convention in `test_library_shell.py`.

- [ ] Run:

```bash
pytest -q Tests/UI/test_library_shell.py -k "file_notes_startup"
```

Expected: FAIL because Library has no bridge.

### Step 4: Add a source-isolated Library bridge

- [ ] Initialize only a separate result slot:

```python
self._file_notes_startup_result: FileNotesProbeResult | None = None
```

- [ ] Add these methods near `_refresh_local_source_snapshot()`:

```python
def _schedule_file_notes_probe_after_database_paint(self) -> None:
    """Queue a one-shot check after a mounted Database Notes canvas paints."""


def _claim_file_notes_probe_after_database_paint(self) -> None:
    """Claim/apply the app-owned probe only when Database Notes is mounted."""


async def _run_file_notes_startup_probe(
    self,
    probe: FileNotesStartupProbe,
    claim: FileNotesProbeClaim,
) -> None:
    """Run one bounded thread query without touching Database Notes state."""
```

Scheduling contract:

1. call `_schedule_file_notes_probe_after_database_paint()` after `_apply_local_source_snapshot()` schedules its successful/error recompose;
2. call it from cached-snapshot `on_mount()` after the explicit recompose;
3. call it after `_select_library_rail_row()` recomposes Browse Notes;
4. inside the post-refresh callback, return unless `_library_loaded` is true, Browse Notes is selected, and `#library-notes-canvas` is mounted;
5. ask the app-owned probe for a decision;
6. apply a synchronous pristine/evidence result directly to `_file_notes_startup_result`;
7. call `run_worker` only when the decision contains a claim, with `exclusive=True` and group `library_file_notes_startup_probe`;
8. create `probe_task = asyncio.to_thread(probe.inspect_claim, claim)` and await `asyncio.wait_for(probe_task, timeout=0.100)`;
9. on timeout or cancellation, invalidate the claim before returning/raising;
10. publish/apply only when `probe.publish()` accepts the still-current claim.

The bridge must never write `_library_lookup_error`, `_library_lookup_recovery_state`, `_local_source_records`, `_local_source_counts`, `_library_loaded`, Database-note editor state, or `LibraryNotesCanvas` props. A0 intentionally renders no new Files UI; TASK-399.2 can use the result for Link preview state and TASK-399.4 renders it in the File Notes workbench.

### Step 5: Verify and commit

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes/test_app_wiring.py
pytest -q Tests/UI/test_library_shell.py -k "file_notes_startup or notes_row_opens_notes_list_canvas or repeat_visit_renders_cached_snapshot_before_refresh_resolves or loading_state_before_snapshot_loads"
pytest -q Tests/Notes/test_notes_scope_service_library_canvas.py
python3 -m compileall -q tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

Expected: all pass.

- [ ] Commit:

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py Tests/Notes/File_Notes/test_app_wiring.py
git commit -m "feat(library): inspect file notes after database paint"
```

---

## Task 5: Prove ChaChaNotes isolation, evidence preservation, and privacy

**Files:**

- Create: `Tests/Notes/File_Notes/test_database_notes_isolation.py`
- Modify: `Tests/Notes/File_Notes/test_bootstrap.py`
- Modify: `Tests/Notes/File_Notes/test_startup_probe.py`

### Step 1: Add structural and behavioral isolation tests

- [ ] Create one real temporary `CharactersRAGDB`, then capture:

  - `_CURRENT_SCHEMA_VERSION`;
  - `db_schema_version`;
  - normalized `sqlite_master` rows for tables, indexes, and triggers;
  - existing note CRUD/search results.

- [ ] Exercise fresh File Notes bootstrap in a sibling storage instance and assert every captured ChaChaNotes value remains identical. Then add, update, search, soft-delete, and restore a Database note successfully.
- [ ] Assert `file_notes_storage`, `note_file_roots`, `file_note_projections`, and `file_notes_fts` do not appear in ChaChaNotes.
- [ ] Assert `notes`, `notes_fts`, existing note triggers, and sync rows do not appear in `file_notes.db`.
- [ ] Replace `file_notes.db` with each failure fixture (corrupt, incompatible, partial, mismatched), mount Library, and prove Database Notes still reaches first paint and CRUD remains available.

### Step 2: Strengthen preservation and privacy tests

- [ ] For every bootstrap refusal and every probe failure that does not enter SQLite, snapshot fixed evidence as `(name, bytes, stat mode, mtime_ns)` before and after and require exact equality, except that access time is intentionally ignored.
- [ ] For probes that open SQLite in WAL mode, require projection DB, WAL, recovery DB, and marker bytes/names/mtimes to remain exact; require application code never to rename, delete, or truncate SHM. Allow only SQLite's documented SHM coordination-page and mtime updates, and assert those updates cannot change the typed result or trigger cleanup/rebootstrap.
- [ ] Capture Loguru output while using sentinel values:

```text
/private/secret-notes/research/[draft].md
BODY-SENTINEL-DO-NOT-LOG
RAW-HASH-SENTINEL-DO-NOT-LOG
```

Require all three to be absent from logs, diagnostics, notifications, and exception representations.

### Step 3: Run the A0 regression set

- [ ] Run:

```bash
pytest -q Tests/Notes/File_Notes
pytest -q Tests/UI/test_library_shell.py -k "file_notes_startup or notes or first_paint or repeat_visit_renders_cached_snapshot_before_refresh_resolves"
pytest -q Tests/Notes/test_notes_scope_service_library_canvas.py Tests/Notes/test_notes_library_unit.py
pytest -q Tests/ChaChaNotesDB/test_chachanotes_db.py
git diff --check
```

Expected: all pass.

- [ ] Commit:

```bash
git add Tests/Notes/File_Notes/test_database_notes_isolation.py Tests/Notes/File_Notes/test_bootstrap.py Tests/Notes/File_Notes/test_startup_probe.py
git commit -m "test(notes): prove file store failure isolation"
```

---

## Task 6: Finish TASK-399.1 with evidence

**Files:**

- Modify: `backlog/tasks/task-399.1 - A0-Isolate-file-note-projection-storage.md`
- Modify: `Docs/superpowers/plans/2026-07-23-file-notes-a0-storage-isolation.md`

### Step 1: Run final verification

- [ ] Run the complete focused matrix:

```bash
pytest -q Tests/Notes/File_Notes
pytest -q Tests/UI/test_library_shell.py -k "file_notes_startup or notes or first_paint or repeat_visit_renders_cached_snapshot_before_refresh_resolves"
pytest -q Tests/Notes/test_notes_scope_service_library_canvas.py Tests/Notes/test_notes_library_unit.py
pytest -q Tests/ChaChaNotesDB/test_chachanotes_db.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

- [ ] Run the full suite before claiming completion:

```bash
pytest
```

Expected: all tests pass. If an unrelated pre-existing failure remains, record its exact command/output and do not mark TASK-399.1 Done until the repository's Definition of Done is genuinely satisfied.

### Step 2: Perform a scope/privacy self-review

- [ ] Confirm with `git diff --name-only` that no ChaChaNotes, BaseDB, existing Notes service/sync, Settings backup, or global DB-init file changed beyond the planned `app.py` bridge.
- [ ] Search the new package for forbidden behavior:

```bash
rg -n "BaseDB|CharactersRAGDB|initialize_all_databases|immutable=1|logger\\..*path|logger\\.opt\\(exception=True\\)|unlink\\(|rename\\(|replace\\(" tldw_chatbook/Notes/file_notes
```

Expected: no unsafe inheritance, eager DB registration, immutable probe, exception traceback logging, or evidence cleanup. Review any intentional bootstrap-only match manually.

- [ ] Run `git diff --check` again.

### Step 3: Update Backlog only after all gates pass

- [ ] Check every acceptance criterion in TASK-399.1.
- [ ] Add concise Implementation Notes covering:

  - profile-derived isolated layout;
  - explicit-only owner-protected bootstrap;
  - schema/FTS isolation;
  - app-lifetime post-paint passive probe;
  - evidence preservation and sanitized diagnostics;
  - exact verification commands/results;
  - ADR-021.

- [ ] Set TASK-399.1 to Done via Backlog CLI only after its complete Definition of Done is met.

### Step 4: Final commit

- [ ] Commit:

```bash
git add 'backlog/tasks/task-399.1 - A0-Isolate-file-note-projection-storage.md' Docs/superpowers/plans/2026-07-23-file-notes-a0-storage-isolation.md
git commit -m "docs(notes): complete file notes A0 foundation"
```

---

## Acceptance-Criteria Traceability

| TASK-399.1 criterion | Plan coverage |
| --- | --- |
| #1 Stable domain-hashed instance storage | Task 1 identity/layout tests and pure resolver |
| #2 Dedicated owner-protected DB/sidecars/markers | Tasks 1–2 layout, permissions, schema, concurrency |
| #3 Fixed independent runtime namespace | Tasks 1–2 runtime layout and bootstrap lock |
| #4 ChaChaNotes unchanged | Global constraints and Task 5 structural/CRUD tests |
| #5 File failure is Files-scoped | Tasks 3–5 typed diagnostics and Library isolation tests |
| #6 Pristine means no DB/open/worker/lease | Tasks 1, 3, and 4 pristine tests |
| #7 One post-paint zero-wait 100 ms query | Tasks 3–4 claim/deadline/first-paint tests |
| #8 Preserve corrupt/partial/orphan evidence | Tasks 2, 3, and 5 byte/stat snapshot tests |
| #9 No absolute roots or bodies in diagnostics | Tasks 3 and 5 sentinel log/result tests |

## Deferred by Design

This plan does not add a File Notes rail group, `Link notes folder…`, root preview, projection inventory, FTS indexing work, navigator/editor UI, monitoring, recovery DB, file mutation, Git controls, or migration machinery. Those are independently accepted milestones after A0 proves that the existing Database Notes path remains untouched and available.
