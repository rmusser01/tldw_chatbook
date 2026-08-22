# TASK-97 Notes Lasting-Sync State Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing device-private `notes.sync_state` owner to an exact, concurrency-safe v2 schema that preserves import receipts while adding paused local Notes sync roots, provisional bindings, and idempotent legacy-migration receipts.

**Architecture:** A domain-model-free schema coordinator becomes the only `notes.sync_state` connection owner and is shared by the existing receipt repository and a new narrow lasting-sync repository. A separate legacy-migration module captures canonical read-only config/ChaChaNotes snapshots and asks the repository to atomically persist provisional candidates; no path admission, activation, filesystem access through candidates, reconciliation, conflict payloads, UI, or server behavior enters this slice.

**Tech Stack:** Python 3.11+, SQLite, dataclasses/enums, existing `connect_private_sqlite`, `CharactersRAGDB`, pytest, Ruff, MyPy.

**Governing design:** `Docs/superpowers/specs/2026-08-20-task-97-notes-lasting-sync-state-foundation-design.md`

**Governing ADRs:** `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`

**ADR required:** no

**ADR path:** existing ADR-059 and ADR-060 above

**Reason:** This plan directly implements the already approved private-owner, migration, binding-uniqueness, and legacy-authority boundaries without introducing another architectural decision.

**Plan review:** Independently approved on 2026-08-22 with no remaining issues.

---

## File map

### Create

- `tldw_chatbook/Notes/notes_sync_state_schema.py` — the only registered `notes.sync_state` connection caller; owns exact v1+v2 DDL, census validation, migration, and transaction sequencing.
- `tldw_chatbook/Notes/notes_sync_state.py` — immutable redacted projections, typed errors, root/binding APIs, invariants, capacities, and durable migration receipts.
- `tldw_chatbook/Notes/notes_sync_legacy_migration.py` — canonical source capture/digests, deterministic preflight, A/B drift protocol, and migration orchestration.
- `Tests/Notes/test_notes_sync_state_schema.py` — exact schema, parity, malformed-version, lock, and initialization-order coverage.
- `Tests/Notes/test_notes_sync_state.py` — root/binding lifecycle, ownership, optimistic versioning, corruption, privacy, and capacity coverage.
- `Tests/Notes/test_notes_sync_legacy_migration.py` — real-source projection, canonical digest, malformed/duplicate/capacity, replay, drift, CAS, and no-candidate-filesystem-access coverage.

### Modify

- `tldw_chatbook/Notes/note_import_receipts.py:22,45,290-461,603-663,3031-3066` — move schema ownership to the coordinator while preserving the receipt API and canonical v1 SQL.
- `Tests/Notes/test_note_import_receipts.py:482-812,4032-4040` — update v2 schema expectations and pin receipt compatibility across the shared transaction seam.
- `tldw_chatbook/DB/private_sqlite.py:222-227` — move the registered owner module from `note_import_receipts` to `notes_sync_state_schema`; keep backup disabled.
- `Tests/DB/test_private_sqlite_inventory.py:930-950` — update the exact connection census and prove the repository cannot bypass the coordinator.
- `backlog/docs/sqlite-private-owner-inventory.md:71` — move curated C50 ownership from the receipt repository to the shared coordinator without changing its privacy/backup disposition.
- `tldw_chatbook/DB/ChaChaNotes_DB.py:11936-11975` — add one bounded, read-only legacy sync source query that returns the exact note/conflict rows in one Notes transaction.
- `backlog/tasks/task-97 - Notes-lasting-sync-private-state-foundation.md` — record this implementation plan, verification evidence, final AC state, and concise implementation notes.
- `Docs/superpowers/specs/2026-08-20-task-97-notes-lasting-sync-state-foundation-design.md` — record user approval; change no technical contract.
- `Docs/superpowers/plans/2026-08-22-task-97-notes-lasting-sync-state-foundation.md` — check steps only after their evidence exists.

### Explicitly untouched

- `tldw_chatbook/Notes/sync_engine.py` and `sync_service.py` — the legacy engine remains the only active owner.
- Library/Textual UI files — no setup, queue, Needs-attention, modal, or activity-log behavior.
- Sync-v2/server/backup/export code — state remains local and backup-excluded.

---

### Task 1: Centralize the private schema owner without changing receipt behavior

**Files:**
- Create: `tldw_chatbook/Notes/notes_sync_state_schema.py`
- Modify: `tldw_chatbook/Notes/note_import_receipts.py:22,45,290-461,603-663,3031-3066`
- Modify: `tldw_chatbook/DB/private_sqlite.py:222-227`
- Modify: `Tests/DB/test_private_sqlite_inventory.py:930-950`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md:71`
- Test: `Tests/Notes/test_notes_sync_state_schema.py`
- Test: `Tests/Notes/test_note_import_receipts.py`

- [x] **Step 1: Write failing ownership and shared-seam tests**

Add tests that assert:

```python
assert SQLITE_OWNER_REGISTRY["notes.sync_state"].production_module == (
    "tldw_chatbook/Notes/notes_sync_state_schema"
)
receipt_calls, receipt_violations = _private_sqlite_seam_violations(
    Path("tldw_chatbook/Notes/note_import_receipts.py"),
    "tldw_chatbook/Notes/note_import_receipts",
)
coordinator_calls, coordinator_violations = _private_sqlite_seam_violations(
    Path("tldw_chatbook/Notes/notes_sync_state_schema.py"),
    "tldw_chatbook/Notes/notes_sync_state_schema",
)
assert receipt_calls == []
assert receipt_violations == []
assert len(coordinator_calls) == 1
assert coordinator_violations == []
```

Add a test that imports `notes_sync_state_transaction`, opens an empty private
database, and observes the unchanged `PRAGMA user_version = 1` receipt schema.
This task centralizes ownership only; Task 2 owns the v2 upgrade.

- [x] **Step 2: Run the focused RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/DB/test_private_sqlite_inventory.py::test_notes_sync_state_inventory_row_is_exact_and_backup_excluded \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_note_import_receipts.py::test_receipt_repository_creates_v1_normalized_schema_without_private_text
```

Expected: FAIL because the coordinator does not exist and the registered owner
still points at `note_import_receipts`. The existing receipt-schema assertion is
a passing non-regression control.

- [x] **Step 3: Move the canonical v1 SQL unchanged**

In `notes_sync_state_schema.py`, cut the existing
`_SCHEMA_TABLE_STATEMENTS` and `_SCHEMA_INDEX_STATEMENTS` tuples from
`note_import_receipts.py:290-459` and paste them byte-for-byte as
`_V1_TABLE_STATEMENTS` and `_V1_INDEX_STATEMENTS`. Keep these constants private
and assemble them explicitly:

```python
SCHEMA_VERSION = 1
_COMPLETE_V1_STATEMENTS = (
    *_V1_TABLE_STATEMENTS,
    *_V1_INDEX_STATEMENTS,
)
```

Do not add any v2 table or reserve future columns in this ownership-only commit.

- [x] **Step 4: Implement schema-before-operation transaction sequencing**

Use one public context manager and one bounded error type:

```python
class NotesSyncStateSchemaError(RuntimeError):
    pass


@contextmanager
def notes_sync_state_transaction(
    database_path: str | Path,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    connection = connect_private_sqlite("notes.sync_state", Path(database_path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        _initialize_schema(connection)
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
```

`_initialize_schema` must preserve the current v1 behavior: version 0 acquires
`BEGIN IMMEDIATE`, creates the complete v1 schema, writes version 1 last, and
commits schema work before the repository operation begins; version 1 runs the
existing canonical index compatibility check without taking a writer slot;
unknown versions fail closed. The exact v2 census/migration replaces this
bounded v1 initializer in Task 2.

- [x] **Step 5: Rewire the receipt repository and preserve its public error boundary**

Remove its direct `connect_private_sqlite`, schema constants, and initializer. Delegate its existing `transaction(immediate=...)` to `notes_sync_state_transaction`; translate `NotesSyncStateSchemaError` into the existing bounded `ImportReceiptError` without exposing paths or raw SQLite text. Keep every receipt model, method signature, transition, and SQL operation unchanged.

- [x] **Step 6: Run GREEN and the complete existing receipt module**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_note_import_receipts.py \
  Tests/DB/test_private_sqlite_inventory.py -k 'notes_sync_state or private_sqlite_seam'
```

Expected: all selected tests PASS; the existing 112 receipt tests remain green,
the database remains canonical v1, and `notes.sync_state` stays absent from
centralized backup rows.

- [x] **Step 7: Run static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/note_import_receipts.py \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_note_import_receipts.py \
  Tests/DB/test_private_sqlite_inventory.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state_schema.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_state_schema.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/note_import_receipts.py
git diff --check
git add tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/note_import_receipts.py \
  tldw_chatbook/DB/private_sqlite.py \
  backlog/docs/sqlite-private-owner-inventory.md \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_note_import_receipts.py \
  Tests/DB/test_private_sqlite_inventory.py
git commit -m "refactor(notes): centralize private sync schema"
```

---

### Task 2: Prove exact v2 schema parity and concurrency

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_state_schema.py`
- Modify: `Tests/Notes/test_notes_sync_state_schema.py`
- Modify: `Tests/Notes/test_note_import_receipts.py`

- [x] **Step 1: Write the exact v2 DDL and historical-upgrade tests**

Add the four v2 tables and seven indexes as literal independent test-fixture SQL
copied from the approved design. Build the historical fixture with a test-only
`_legacy_v1_transaction` that creates the complete moved v1 DDL, then
temporarily monkeypatch the receipt module's coordinator transaction symbol so
the real current `NoteImportReceiptRepository.begin` and transition APIs seed
meaningful rows into that v1 database. Remove the monkeypatch before opening the
production coordinator for upgrade. This proves real receipt behavior on an
exact v1 schema without labeling a partial handwritten database as historical.
Test:

- v1 → v2 retains every receipt row and behavior;
- fresh v2 and upgraded v1 have identical table/index/column/type/null/default/PK/FK/CHECK/partial-index census;
- receipt-seam-first and direct-coordinator-first schema initialization yield
  the same v2 (the actual two-repository order is added in Task 3);
- a database claiming v2 with a missing/changed table or index fails closed;
- versions above 2 fail before mutation;
- `migration_id`, `root_id`, and `binding_id` reject `NULL` through executed INSERT probes;
- `direction='unspecified'` rejects manual/null/wrong-reason rows through executed INSERT probes.

The fresh-v2 side must be an independently hand-authored test fixture containing
literal canonical SQL copied from the approved design, not a call into production
DDL constants. Its census is the external oracle; compare both production fresh
creation and the real-repository v1 upgrade against it so the same production
mistake cannot confirm itself.

- [x] **Step 2: Run the schema RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_state_schema.py \
  -k 'parity or malformed or null_primary_key or unspecified_direction'
```

Expected: FAIL until the census and all exact validation branches exist. Fixture errors are not acceptable RED evidence.

- [x] **Step 3: Implement the exact census**

First add the four production v2 table constants and seven production index
constants verbatim from the approved design, set `SCHEMA_VERSION = 2`, and
assemble `_COMPLETE_V2_STATEMENTS` from the unchanged v1 tuples plus the new v2
tuples. Then replace the bounded v1 initializer with the final version-aware
initializer: read `user_version` without a writer; validate committed v2 in a
read transaction; for 0/1 acquire `BEGIN IMMEDIATE`, reread the version, create
or upgrade additively, validate, write version 2 last, and commit before the
operation transaction begins.

Add a frozen internal snapshot used by tests and validation:

```python
@dataclass(frozen=True, slots=True)
class SyncStateSchemaSnapshot:
    user_version: int
    tables: tuple[TableCensus, ...]
    indexes: tuple[IndexCensus, ...]
```

Build it only from allowlisted names using `sqlite_master`, `pragma_table_xinfo`, `pragma_foreign_key_list`, and `pragma_index_xinfo`. Normalize whitespace only through one deterministic SQL normalizer used for both expected and observed SQL; do not weaken predicates or CHECK text into substring checks.

- [x] **Step 4: Write real two-connection initialization and fast-path tests**

Use `threading.Barrier` and two independent direct coordinator connections
against one file. Assert one complete v2 and no leaked transaction. For the
fast path, hold a separate `BEGIN IMMEDIATE` writer and prove an ordinary
committed-v2 schema validation/read completes without waiting for that writer;
mutation-test removal of the under-lock `user_version` reread and moving
`PRAGMA user_version = 2` earlier. Task 3 supplies the integration proof using
both actual repositories.

- [x] **Step 5: Run GREEN and repeat the receipt compatibility module**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_note_import_receipts.py
```

Expected: PASS with no changed receipt projection, transition, retry, or aggregate behavior.

- [x] **Step 6: Static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state_schema.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state_schema.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_state_schema.py
git diff --check
git add tldw_chatbook/Notes/notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_note_import_receipts.py
git commit -m "test(notes): harden sync schema migration"
```

---

### Task 3: Add redacted paused-root persistence

**Files:**
- Create: `tldw_chatbook/Notes/notes_sync_state.py`
- Create: `Tests/Notes/test_notes_sync_state.py`

- [x] **Step 1: Write failing root/model tests**

Cover immutable/slotted projections, redacted `repr`, bounded typed errors, create/get/list, lexical path preservation without normalization, candidate update, pause, optimistic conflict, disconnect, terminal disconnect, and the exact 64-live-root capacity. Assert public `repr` and exception text contain none of the supplied path, source locator, migration ID, or raw SQLite message.

Also add actual repository-order/concurrency integration cases against one
database: receipt repository then sync repository; sync repository then receipt
repository; and a `threading.Barrier` race where each repository initializes and
writes through its own real connection. In every case both repositories must
remain usable and the exact v2 census must match.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_state.py -k 'root or projection or capacity'
```

Expected: collection or assertion FAIL because `NotesSyncStateRepository` and its models do not exist.

- [x] **Step 3: Implement the narrow public model and error surface**

Use string enums and frozen slotted dataclasses; mark private fields `repr=False` and provide an explicitly redacted representation:

```python
MAX_SYNC_ROOTS = 64

class SyncRootState(StrEnum):
    CANDIDATE = "candidate"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"

@dataclass(frozen=True, slots=True)
class SyncRootRecord:
    root_id: str = field(repr=False)
    lexical_root_path: str = field(repr=False)
    display_name: str
    direction: str
    state: SyncRootState
    row_version: int
    needs_rescan: bool
    reason_code: str | None
    source_kind: str | None = field(repr=False)
    source_locator_digest: str | None = field(repr=False)
    source_migration_id: str | None = field(repr=False)
    created_at: int
    updated_at: int

    def __repr__(self) -> str:
        return (
            "SyncRootRecord(state="
            f"{self.state.value!r}, row_version={self.row_version}, "
            f"needs_rescan={self.needs_rescan!r}, reason_code={self.reason_code!r})"
        )
```

Define `NotesSyncStateError`, `SyncStateConflictError`, `SyncStateCapacityError`, and `SyncStateCorruptionError` with bounded reason codes/counts only.

- [x] **Step 4: Implement named root operations**

Implement only:

```python
create_candidate_root(lexical_root_path, display_name, direction)
get_root(root_id)
list_roots()
update_candidate_root(
    root_id,
    expected_version,
    *,
    display_name=None,
    direction=None,
)
pause_root(root_id, expected_version, reason_code)
disconnect_root(root_id, expected_version)
```

Every mutation uses `notes_sync_state_transaction(..., immediate=True)`, exact `row_version` compare-and-set, and one preflight before writes. Do not add generic state setters, reopen, activate, or run methods. Root disconnect must version-bump and disconnect all live children in the same transaction, even though binding APIs arrive in Task 4.

- [x] **Step 5: Run GREEN and mutation-check the optimistic predicate**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_state.py -k 'root or projection or capacity'
```

Temporarily remove `AND row_version = ?` from one update; confirm the stale-version test fails, then restore it and rerun GREEN.

Run the actual repository-order cases explicitly as part of this GREEN:

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_state.py \
  -k 'repository_initialization_order or concurrent_repository_initialization'
```

- [x] **Step 6: Static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_state.py \
  Tests/Notes/test_notes_sync_state.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_state.py \
  Tests/Notes/test_notes_sync_state.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_state.py
../../.venv/bin/python -m compileall -q tldw_chatbook/Notes/notes_sync_state.py
git diff --check
git add tldw_chatbook/Notes/notes_sync_state.py \
  Tests/Notes/test_notes_sync_state.py
git commit -m "feat(notes): persist paused sync roots"
```

---

### Task 4: Add provisional bindings and close ownership invariants

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_state.py`
- Modify: `Tests/Notes/test_notes_sync_state.py`

- [x] **Step 1: Write failing binding/invariant tests**

Cover:

- create/get/list/update provisional bindings;
- global non-disconnected note uniqueness across candidate and paused roots;
- nullable `path_key` allowed and non-null per-root uniqueness enforced;
- individual disconnect releases ownership and capacity;
- disconnected binding cannot reopen;
- create/update rejects missing or disconnected parent;
- root disconnect atomically versions/disconnects every child and releases root/binding capacity;
- corrupt disconnected-root/live-child reads fail closed;
- two real connections racing disconnect vs create preserve the invariant;
- exact 100,000-binding ceiling and one-over atomic rejection using a lowered test constant or preseeded SQL counts rather than a 100,001-object payload.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_state.py \
  -k 'binding or ownership or disconnect or path_key or race'
```

Expected: FAIL only for the absent binding APIs/invariants.

- [x] **Step 3: Implement the binding projection and named operations**

Mirror every approved `sync_bindings` column—including `created_at` and
`updated_at`—in a frozen slotted `SyncBindingRecord`, with IDs, paths, digests,
and migration IDs excluded from `repr`. Migration run/item records added in
Task 6 must likewise project every column from their exact tables while keeping
private identifiers/digests out of diagnostic representations. Implement only:

```python
create_provisional_binding(root_id, note_id, lexical_relative_path)
get_binding(binding_id)
list_bindings(root_id=root_id)
update_provisional_binding(
    binding_id,
    expected_version,
    *,
    lexical_relative_path=None,
    path_key=None,
)
mark_binding_needs_attention(binding_id, expected_version, reason_code)
disconnect_binding(binding_id, expected_version)
```

Each write begins immediate, validates the parent root in that transaction, preflights deterministic conflicts, and leaves SQLite partial unique indexes as the final race guard. Wrap `sqlite3.IntegrityError` into a bounded repository error without echoing identifiers or paths.

- [x] **Step 4: Implement and verify atomic root-child disconnect**

In `disconnect_root`, update all non-disconnected children first with `row_version = row_version + 1`, then the root with its expected-version predicate, and roll back everything when the root predicate changes zero rows. Reads must check for `root.state='disconnected' AND binding.state<>'disconnected'` and raise `SyncStateCorruptionError` rather than return contradictory authority.

- [x] **Step 5: Run GREEN and the whole state module**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_state.py
```

Mutation-check one parent-state guard and one partial unique predicate; each mutation must fail a named test before restoration.

- [x] **Step 6: Static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_state.py \
  Tests/Notes/test_notes_sync_state.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_state.py \
  Tests/Notes/test_notes_sync_state.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_state.py
git diff --check
git add tldw_chatbook/Notes/notes_sync_state.py \
  Tests/Notes/test_notes_sync_state.py
git commit -m "feat(notes): enforce provisional sync bindings"
```

---

### Task 5: Capture and canonicalize the real legacy source read-only

**Files:**
- Create: `tldw_chatbook/Notes/notes_sync_legacy_migration.py`
- Create: `Tests/Notes/test_notes_sync_legacy_migration.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:11936-11975`

- [x] **Step 1: Write a real ChaChaNotes/config source fixture and failing digest tests**

Use `CharactersRAGDB` to create a current database, then seed legacy sync metadata/conflicts through its transaction seam. Use a real temporary `TLDW_CONFIG_PATH`. Pin the exact relevant-note OR predicate and unresolved conflict predicate by making each term independently include a row and making unrelated/resolved rows not affect the source digest.

Test canonical stability across input/dictionary order; exact Unicode spellings; `None`/`False`/`0`/`"0"`; finite float hex; non-finite markers; missing config defaults; new-vs-legacy conflict key precedence; direction aliases; and every root/binding/item locator input. Assert malformed null/non-string/empty/NUL/over-limit path values still yield deterministic non-null rejected-item locators without persisting their raw value.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_legacy_migration.py \
  -k 'source or canonical or digest or locator or predicate'
```

Expected: FAIL because the read seam and canonical migration module do not exist.

- [x] **Step 3: Add one bounded ChaChaNotes source method**

Add `read_legacy_notes_sync_source_rows()` to `CharactersRAGDB`. Within one ordinary read transaction, select exactly the approved note fields with the exact OR predicate (including soft-deleted rows), ordered by exact note ID, then select the exact unresolved/skip conflict fields ordered by integer ID. Return frozen tuples of plain mappings; do not resolve paths, read note content, mutate rows, or expose a connection.

- [x] **Step 4: Implement exact canonical models and digest helpers**

In `notes_sync_legacy_migration.py`, use the existing Notes import canonical JSON representation:

```python
def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _real_value(value: object) -> str | None:
    if value is None:
        return None
    number = float(value)
    return number.hex() if math.isfinite(number) else "invalid_non_finite_real"
```

Read config through `get_atomic_config_snapshot()` and apply exact defaults/alias precedence from the design. Validate scalar types before hashing. Represent the source revision and locator shapes exactly as specified; preserve JSON scalar types and do no Unicode/path normalization.

- [x] **Step 5: Add operand-aware no-candidate-filesystem-access proof**

Use nonexistent/adversarial candidate values. Spy on `Path.resolve`, `Path.absolute`, `Path.stat`, `Path.lstat`, `Path.open`, built-in `open`, and directory iteration so they raise only when the operand equals/contains a migrated candidate; allow the real config and two SQLite files. Assert config/database reads occur, while no candidate operand is touched and all file/note/config/legacy rows are byte/row identical afterward.

- [x] **Step 6: Run GREEN and mutate canonical ordering/predicates**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_legacy_migration.py \
  -k 'source or canonical or digest or locator or predicate or filesystem'
```

Temporarily remove note sorting and one OR-predicate term; confirm the respective named tests fail, restore, and rerun GREEN.

- [x] **Step 7: Static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py
git diff --check
git add tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
git commit -m "feat(notes): capture legacy sync source"
```

---

### Task 6: Persist deterministic migration generations and drift receipts

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_state.py`
- Modify: `tldw_chatbook/Notes/notes_sync_legacy_migration.py`
- Modify: `Tests/Notes/test_notes_sync_legacy_migration.py`
- Modify: `Tests/Notes/test_notes_sync_state.py`

- [x] **Step 1: Write failing end-to-end migration tests**

Cover multiple lexical roots; exact direction mapping; valid/malformed siblings; duplicate-note equivalence classes where no member wins; claims against an existing owner; legacy conflicts as `needs_rescan` only; item combination CHECKs; aggregate counts derived from items; exact same digest replay; pending replay after simulated crash; changed digest updating only exact migration-owned candidate rows; paused/reviewed/manual/disconnected rows never overwritten; matched and drifted A/B runs; and no activation/watcher/content/conflict record created.

- [x] **Step 2: Write failing capacity and CAS race tests**

Use lowered constants or preseeded counts to prove global 64-root/100,000-binding overflow writes no run/root/binding/item. Use two real destination connections and controlled A/B readers so both observe one pending generation but propose different B digests; assert one conditional terminal update wins and the loser rereads the immutable winner.

- [x] **Step 3: Run RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_notes_sync_legacy_migration.py \
  -k 'migration or replay or duplicate or drift or capacity or finalize'
```

Expected: FAIL only for missing persistence/orchestration behavior.

- [x] **Step 4: Implement deterministic preflight and repository write models**

Define private frozen request models for roots, bindings, and migration items. Classify malformed items and duplicate note equivalence classes before opening the destination write transaction. Stable destination IDs derive only from the validated locator digest (for example `legacy-root-<digest>` / `legacy-binding-<digest>`); rejected inputs use only their typed item locator and no destination ID.

Add one repository operation that atomically:

1. finds/creates the `(source_kind, source_revision_before)` run;
2. returns terminal runs unchanged;
3. returns pending runs without rewriting candidates/items;
4. preflights current live counts and all existing ownership/source claims;
5. aborts globally on capacity;
6. inserts/updates only permitted migration-owned candidate rows;
7. writes exact item outcomes; and
8. commits with every root/binding still provisional and `needs_rescan=1`.

- [x] **Step 5: Implement A/commit/B/CAS orchestration**

`migrate_legacy_notes_sync_state(repository, notes_db)` must:

```python
snapshot_a = capture_legacy_source(notes_db)
pending_or_terminal = repository.record_legacy_generation(snapshot_a)
if pending_or_terminal.state != MigrationState.PENDING_RECHECK:
    return pending_or_terminal
snapshot_b = capture_legacy_source(notes_db)  # always fresh
return repository.finalize_legacy_generation(
    pending_or_terminal.migration_id,
    source_revision_after=snapshot_b.digest,
)
```

The final SQL is an exact compare-and-set:

```sql
UPDATE sync_migration_runs
SET state = ?, source_revision_after = ?, updated_at = ?
WHERE migration_id = ?
  AND state = 'pending_recheck'
  AND source_revision_after IS NULL
```

When `rowcount == 0`, reread and return the existing terminal record without mutation. A digest match produces `matched_recheck`; mismatch produces `drifted`; both leave candidates provisional and requiring rescan.

- [x] **Step 6: Run GREEN and mutation proofs**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_legacy_migration.py \
  Tests/Notes/test_notes_sync_state.py
```

Individually mutate and restore: duplicate preflight winner suppression, capacity-before-write ordering, pending replay short circuit, and CAS predicate. Each mutation must make a named test fail before restoration.

- [x] **Step 7: Static checks and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py
git diff --check
git add tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
git commit -m "feat(notes): migrate legacy sync candidates"
```

---

### Task 7: Verify governance, privacy, and deliberate non-goals

**Files:**
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md:71`
- Modify: `Tests/Notes/test_notes_sync_state.py`
- Modify: `Tests/Notes/test_notes_sync_legacy_migration.py`
- Modify: `Tests/Notes/test_note_import_receipts.py`

- [x] **Step 1: Add source/inventory privacy ratchets**

Assert:

- only `notes_sync_state_schema.py` calls `connect_private_sqlite("notes.sync_state", ...)`;
- `notes.sync_state` remains a private-file owner and absent from all centralized backup/export matrices;
- state/migration model `repr`, errors, logs, and aggregates contain no path, note ID, digest, migration ID, raw exception, content, or hash input;
- no new code references activation, watcher, reconcile, resolver, journal, UI, server, Sync-v2, or backup APIs;
- legacy `sync_engine.py` remains unchanged and the new migration has no startup/app invocation.

- [x] **Step 2: Run governance RED/GREEN**

Run the new ratchets before implementing any last correction; if one fails, make only the smallest boundary fix and rerun:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/DB/test_private_sqlite_inventory.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py \
  -k 'inventory or privacy or redacted or non_goal or legacy_owner'
```

Expected final result: PASS. Temporarily add a forbidden direct owner call in a scratch mutation, confirm the inventory test fails, then restore it byte-for-byte.

- [x] **Step 3: Run the complete bounded related matrix**

Do not run unrelated test directories or a full-repository sweep. Run only the touched foundation and its direct compatibility boundary:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py \
  Tests/Notes/test_note_import_receipts.py \
  Tests/Notes/test_note_import_executor.py \
  Tests/DB/test_private_sqlite_inventory.py
```

Expected: all selected tests PASS. If an unchanged baseline failure appears, reproduce that exact node against the design base before classifying it; do not broaden the run.

- [x] **Step 4: Run final static analysis with provenance**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  tldw_chatbook/Notes/note_import_receipts.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py \
  Tests/Notes/test_note_import_receipts.py \
  Tests/DB/test_private_sqlite_inventory.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  Tests/Notes/test_notes_sync_state_schema.py \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Notes/notes_sync_state_schema.py \
  tldw_chatbook/Notes/notes_sync_state.py \
  tldw_chatbook/Notes/notes_sync_legacy_migration.py \
  tldw_chatbook/Notes/note_import_receipts.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py
git diff --check
```

Any diagnostic on a changed line must be fixed. Any claimed baseline diagnostic must be reproduced with the exact command against the design base and recorded, not waived by assertion.

- [x] **Step 5: Commit the governance tests**

```bash
git add Tests/DB/test_private_sqlite_inventory.py \
  backlog/docs/sqlite-private-owner-inventory.md \
  Tests/Notes/test_notes_sync_state.py \
  Tests/Notes/test_notes_sync_legacy_migration.py \
  Tests/Notes/test_note_import_receipts.py
git commit -m "test(notes): enforce sync foundation boundaries"
```

---

### Task 8: Review and close out TASK-97

**Files:**
- Modify: `backlog/tasks/task-97 - Notes-lasting-sync-private-state-foundation.md`
- Modify: `Docs/superpowers/specs/2026-08-20-task-97-notes-lasting-sync-state-foundation-design.md`
- Modify: `Docs/superpowers/plans/2026-08-22-task-97-notes-lasting-sync-state-foundation.md`
- Optional only if a genuinely new incident generalizes: `backlog/docs/lessons-testing-evidence.md`

- [x] **Step 1: Perform an independent cumulative review**

Use `superpowers:requesting-code-review` over `merge-base(origin/dev, HEAD)..HEAD`, not a stale literal branch range. Require explicit review of schema lock sequencing, census exactness, receipt compatibility, root-child invariants, privacy, migration idempotence/CAS, candidate-path non-access, capacities, and all non-goals. Resolve every P0-P2 finding with focused RED/GREEN evidence before proceeding.

- [x] **Step 2: Update the task and plan from actual evidence**

In TASK-97:

- check an AC only when its named tests and review evidence exist;
- add a concise `## Implementation Notes` covering approach, files, schema/transaction choices, migration protocol, test counts, warnings/baseline provenance, ADR-059/060, and deviations;
- record that no new lesson was needed, or link the exact incident-based lesson if one truly arose;
- keep status `In Progress` until every DoD item is complete.

Update this plan's checkboxes only after each step completes. Change the design status from user-approved to implemented only after the final matrix and review pass.

- [x] **Step 3: Verify exact task resolution and final hygiene**

```bash
backlog task 97 --plain
git status --short
git diff --check
```

Confirm the CLI resolves exact TASK-97, all ACs are checked, Implementation Plan/Notes and ADR links exist, and only intended closeout documents are dirty.

- [x] **Step 4: Mark Done through the Backlog CLI**

```bash
backlog task edit 97 -s Done
backlog task 97 --plain
```

Expected: exact TASK-97 reports `Done`; no duplicate or misnamed task file is created.

- [x] **Step 5: Commit closeout docs**

```bash
git add \
  "backlog/tasks/task-97 - Notes-lasting-sync-private-state-foundation.md" \
  Docs/superpowers/specs/2026-08-20-task-97-notes-lasting-sync-state-foundation-design.md \
  Docs/superpowers/plans/2026-08-22-task-97-notes-lasting-sync-state-foundation.md
git commit -m "docs(notes): complete TASK-97 sync foundation"
git status --short
git show --check --stat --oneline HEAD
```

Expected: clean worktree and a closeout commit containing only the task/spec/plan (plus an incident-backed lesson if one was genuinely required).

- [ ] **Step 6: Use the branch-finishing workflow**

Invoke `superpowers:finishing-a-development-branch` only after the clean-worktree verification. Do not push, open a PR, rebase, or merge unless the user chooses that integration action.
