# One-time Database Notes Import Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute an explicitly approved one-time import plan into local Database Notes with bounded progress, deterministic replay, durable private receipts, and failure-only retry.

**Architecture:** Keep the read-only planner unchanged. Add a private approval/execution vocabulary, a profile-local SQLite receipt repository registered with the central private-SQLite policy, and a synchronous local target adapter plus executor. The async facade runs the entire executor off the Textual event loop and marshals immutable progress back to the caller's loop. Every note, keyword, and membership effect receives a durable state; deterministic note/folder IDs and target read-back close the crash window between a ChaChaNotes commit and its receipt update.

**Tech Stack:** Python 3.11+, frozen dataclasses/enums, SQLite through `connect_private_sqlite`, `asyncio.to_thread`, `threading.Event`, existing `NotesInteropService`, `LocalNoteFolderRepository`, and pytest.

**Scope:** Local Database Notes only. Server-backed folders remain explicitly unsupported until the separately versioned `tldw_server` capability in delivery-roadmap steps 5–6. `ParsedNotePayload.template_name` remains private fingerprint input because Database Notes has no template-name field; title, content, keywords, and approved manual memberships are persisted.

**ADR required:** no

**ADR paths:** `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

**Reason:** ADR-059 already assigns one-time import provenance and receipts to the device-private Notes sync owner. ADR-073 already fixes optimistic replacement, backup exclusion, private matching material, and interruption semantics. This slice implements those accepted boundaries without creating a new authority.

---

## File map

- Create `tldw_chatbook/Notes/note_import_execution_models.py`: approval token, session/item/effect states, immutable progress and receipt projections, private plan/source digests.
- Create `tldw_chatbook/Notes/note_import_receipts.py`: schema-v1 private receipt ledger, state transitions, retry selection, and prior-observation lookup.
- Create `tldw_chatbook/Notes/note_import_executor.py`: local target adapter, deterministic IDs, bounded executor, crash reconciliation, and async facade.
- Modify `tldw_chatbook/Notes/note_folder_repository.py`: exact normalized-path lookup and caller-supplied deterministic folder IDs.
- Modify `tldw_chatbook/config.py`: profile-local `get_notes_sync_state_db_path()` accessor.
- Modify `tldw_chatbook/DB/private_sqlite.py`: register `notes.sync_state` as a private-file-only owner with centralized backup disabled.
- Modify `backlog/docs/sqlite-private-owner-inventory.md`: inventory the new connection owner and its backup exclusion.
- Create `Tests/Notes/test_note_import_execution_models.py`.
- Create `Tests/Notes/test_note_import_receipts.py`.
- Create `Tests/Notes/test_note_import_executor.py`.
- Modify `Tests/Notes/test_note_folder_repository.py`.
- Modify `Tests/DB/test_private_sqlite.py` and `Tests/DB/test_private_sqlite_inventory.py` only where the generic registry/inventory ratchets require explicit coverage.
- Modify `backlog/tasks/task-16309 - Execute-approved-one-time-Database-Notes-import-plans-with-durable-receipts.md` for closeout evidence.

---

### Task 1: Define explicit approval and redacted execution models

**Files:**

- Create: `tldw_chatbook/Notes/note_import_execution_models.py`
- Create: `Tests/Notes/test_note_import_execution_models.py`

- [ ] **Step 1: Write failing approval-boundary tests**

Cover these behaviors with real `NoteImportPlan` fixtures:

```python
def test_approval_rejects_an_unresolved_root_collision() -> None:
    with pytest.raises(ImportApprovalError, match="resolved"):
        approve_note_import_plan(_plan(root_collision=_unresolved_collision()))


def test_approved_plan_is_opaque_and_bound_to_exact_effects() -> None:
    approved = approve_note_import_plan(
        _plan(), approval_id="00000000-0000-4000-8000-000000000001"
    )
    assert approved.plan is not None
    assert "Body secret" not in repr(approved)
    assert approved.approval_id == "00000000-0000-4000-8000-000000000001"
    assert len(approved._private_plan_digest()) == 64


def test_public_execution_diagnostic_contains_counts_not_private_fields() -> None:
    diagnostic = _receipt().to_diagnostic()
    rendered = repr(diagnostic)
    assert diagnostic.imported == 1
    assert "note-id" not in rendered
    assert "source" not in rendered
    assert "fingerprint" not in rendered
```

- [ ] **Step 2: Run the new file and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_execution_models.py
```

Expected: collection fails because `note_import_execution_models` does not exist.

- [ ] **Step 3: Implement the minimal frozen model vocabulary**

Define:

```python
class ImportSessionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NEEDS_ATTENTION = "needs_attention"


class ImportItemOutcome(str, Enum):
    PENDING = "pending"
    IMPORTED = "imported"
    UPDATED = "updated"
    SKIPPED = "skipped"
    FAILED = "failed"


class ImportEffectState(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, repr=False)
class ApprovedNoteImportPlan:
    approval_id: str
    plan: NoteImportPlan
    __plan_digest: str

    def _private_plan_digest(self) -> str:
        return self.__plan_digest

    def __repr__(self) -> str:
        return "ApprovedNoteImportPlan(<private>)"
```

`approve_note_import_plan()` must validate UUID text, require any colliding root to have an explicit choice, and compute a canonical SHA-256 digest over private source-locator digests, payload fingerprints, selected actions/effects, match identity/version, memberships, resolved folder paths, and bounds. Never expose that canonical payload through a public serializer.

Add frozen `ImportExecutionProgress`, `ImportExecutionReceipt`, `ImportExecutionDiagnostic`, and bounded safe reason-code validation. Counts are public; note IDs, folder IDs, paths, hashes, and raw errors are private/repr-hidden.

- [ ] **Step 4: Run the execution-model and planner suites GREEN**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_execution_models.py Tests/Notes/test_note_import_planner.py
```

Expected: all pass.

- [ ] **Step 5: Commit the approval vocabulary**

Stage only the model and model-test files and commit `feat(notes): define approved import execution model`.

---

### Task 2: Add the private import receipt owner and schema

**Files:**

- Create: `tldw_chatbook/Notes/note_import_receipts.py`
- Create: `Tests/Notes/test_note_import_receipts.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `Tests/DB/test_private_sqlite.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`

- [ ] **Step 1: Write RED tests for privacy policy, schema, and lifecycle**

Required cases:

```python
def test_notes_sync_state_owner_is_private_file_only_and_not_backup_enabled() -> None:
    policy = SQLITE_OWNER_REGISTRY["notes.sync_state"]
    assert policy.allowed_target_kinds == frozenset({SQLiteTargetKind.PRIVATE_FILE})
    assert policy.centralized_backup_allowed is False


def test_receipt_repository_creates_v1_schema_without_payload_content(tmp_path: Path) -> None:
    repository = NoteImportReceiptRepository(tmp_path / "notes-sync.sqlite3")
    approved = approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)
    repository.begin(approved, batch_size=25)
    schema = repository._test_schema_snapshot()
    assert schema.user_version == 1
    assert {"import_sessions", "import_items", "import_payload_effects", "import_folder_effects", "import_membership_effects"} <= set(schema.tables)
    assert "Body secret" not in (tmp_path / "notes-sync.sqlite3").read_bytes().decode("utf-8", errors="ignore")


def test_begin_is_idempotent_for_one_approval_and_rejects_digest_substitution(tmp_path: Path) -> None:
    repository = NoteImportReceiptRepository(tmp_path / "notes-sync.sqlite3")
    first = repository.begin(_approved_plan(), batch_size=10)
    assert repository.begin(_approved_plan(), batch_size=10) == first
    with pytest.raises(ImportReceiptConflictError):
        repository.begin(_different_plan_same_approval_id(), batch_size=10)
```

Also pin: exact state-transition allowlist; transactional item/effect updates; reopen durability; no absolute source path/title/content/keyword/raw exception columns; connection mode `0600` and parent mode `0700` where POSIX verifies them; and no inclusion in generic backup matrices or Chatbook database paths.

- [ ] **Step 2: Run the receipt and private-SQLite tests and verify RED**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_receipts.py Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py
```

Expected: missing owner, accessor, and repository failures.

- [ ] **Step 3: Register the storage owner and accessor**

Add:

```python
def get_notes_sync_state_db_path() -> Path:
    """Return the device-private Notes import/sync state database path."""
    return get_user_data_dir() / "tldw_chatbook_notes_sync_state.db"
```

Register `notes.sync_state` in `SQLITE_OWNER_REGISTRY` with private-file-only access, no centralized backup, and a rationale naming device-private import receipts plus future lasting-sync state. Add one connection-owner inventory row; do not add the database to Chatbook export, Settings bulk backup, or centralized backup behavior matrices.

- [ ] **Step 4: Implement schema-v1 and repository transitions**

Use only parameterized SQL through `connect_private_sqlite("notes.sync_state", path)`. Schema v1 contains normalized rows for sessions, items, payload effects, folder effects, and membership effects. Persist only:

- opaque approval/session/item IDs;
- private plan/source/payload digests;
- selected action and effect state;
- opaque target note/folder IDs and expected/observed versions;
- bounded reason codes, retryable flags, counts, and timestamps.

Do not persist source paths, titles, content, keywords, template names, or exception text. Each public method opens a bounded transaction, validates state transitions, and returns frozen models rather than SQLite rows.

- [ ] **Step 5: Run receipt/private-owner tests GREEN**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_receipts.py Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py
```

Expected: all pass with the registry behavioral matrix exercising the new owner.

- [ ] **Step 6: Commit the receipt foundation**

Stage the accessor, private owner, inventory, receipt repository, and exact tests. Commit `feat(notes): add private import receipt ledger`.

---

### Task 3: Add deterministic local folder and note target operations

**Files:**

- Modify: `tldw_chatbook/Notes/note_folder_repository.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`
- Create: `tldw_chatbook/Notes/note_import_executor.py`
- Create: `Tests/Notes/test_note_import_executor.py`

- [ ] **Step 1: Write RED folder repository tests**

```python
def test_create_folder_accepts_a_valid_deterministic_id(db) -> None:
    repository = LocalNoteFolderRepository(db)
    folder = repository.create_folder(
        name="Imported",
        parent_id=None,
        folder_id="00000000-0000-5000-8000-000000000001",
    )
    assert folder.id == "00000000-0000-5000-8000-000000000001"


def test_get_folder_by_path_uses_normalized_exact_segments(db) -> None:
    repository = LocalNoteFolderRepository(db)
    parent = repository.create_folder(name="Café", parent_id=None)
    child = repository.create_folder(name="Ideas", parent_id=parent.id)
    assert repository.get_folder_by_path(("Cafe\u0301", "ideas")) == child
```

Also reject malformed caller IDs and prove existing callers without `folder_id` retain random UUID behavior.

- [ ] **Step 2: Run the two exact repository tests RED**

Expected: `create_folder` rejects the keyword and `get_folder_by_path` is absent.

- [ ] **Step 3: Implement the minimal repository seam**

Extend `create_folder(*, name, parent_id, folder_id=None)` with the repository's existing opaque-ID validation and keep the same transaction/collision behavior. Add `get_folder_by_path(folder_segments)` using `normalize_folder_name`, `join_normalized_folder_path`, one exact active-row query, and the existing row mapper.

- [ ] **Step 4: Write RED local-target tests**

Construct a real temporary `CharactersRAGDB`, `NotesInteropService`, and `LocalNoteFolderRepository`; do not write a fake that copies the executor's assumed signatures. Cover:

- deterministic folder ensure creates once and returns the same folder on retry;
- pre-existing root reuse is allowed only for explicit `USE_EXISTING`;
- a stale/concurrent path collision raises `ImportTargetConflictError` before note mutation;
- deterministic note creation reconciles a post-commit/pre-receipt crash by exact note ID and payload comparison;
- optimistic update distinguishes unchanged expected version, already-applied expected+1 payload, and conflicting later version;
- keyword synchronization is idempotent and exact;
- manual membership attachment is idempotent.

- [ ] **Step 5: Implement `LocalNoteImportTarget`**

The adapter is synchronous and owns the real local signatures:

```python
class LocalNoteImportTarget:
    def ensure_folder(self, *, segments, folder_id, allow_existing): ...
    def read_note(self, *, note_id): ...
    def create_note(self, *, note_id, payload): ...
    def replace_note(self, *, note_id, expected_version, payload): ...
    def keywords_match(self, *, note_id, keywords): ...
    def sync_keywords(self, *, note_id, keywords): ...
    def attach_membership(self, *, folder_id, note_id): ...
```

Wrap expected database contention as `ImportTargetRetryableError`, optimistic/path conflicts as `ImportTargetConflictError`, and validation/capability failures as `ImportTargetPermanentError`. Messages remain constant safe text; exception messages and paths never cross the adapter.

- [ ] **Step 6: Run target and repository suites GREEN**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_folder_repository.py Tests/Notes/test_note_import_executor.py -k 'target or deterministic or folder_by_path'
```

- [ ] **Step 7: Commit the deterministic local target**

Commit `feat(notes): add deterministic import target operations` with explicit-path staging.

---

### Task 4: Execute approved plans with per-effect durable receipts

**Files:**

- Modify: `tldw_chatbook/Notes/note_import_executor.py`
- Modify: `Tests/Notes/test_note_import_executor.py`
- Modify: `Tests/Notes/test_note_import_receipts.py`

- [ ] **Step 1: Write RED happy-path and independent-effect tests**

Use real temporary SQLite owners and cover:

```python
def test_create_new_executes_multi_payload_notes_and_parent_memberships(real_executor):
    receipt = real_executor.execute(_approved_multi_note_plan())
    assert receipt.imported == 2
    assert receipt.failed == 0
    _assert_notes_and_memberships_match_plan()


def test_update_can_replace_content_without_adding_membership(real_executor):
    receipt = real_executor.execute(_approved_update(replace=True, place=False))
    assert receipt.updated == 1
    _assert_content_replaced()
    _assert_import_folder_not_attached()


def test_update_can_add_membership_without_replacing_content(real_executor):
    receipt = real_executor.execute(_approved_update(replace=False, place=True))
    assert receipt.updated == 1
    _assert_original_content_preserved()
    _assert_import_folder_attached()
```

Also pin all-Skip behavior, unsupported/failed Skip outcomes, root creation only for approved effects, keyword persistence, and final imported/updated/skipped/failed/retryable counts.

- [ ] **Step 2: Run exact tests RED**

Expected: executor API is absent.

- [ ] **Step 3: Implement deterministic identities and folder preflight**

Derive UUIDv5 folder IDs from `(approval_id, normalized folder path)` and note IDs from `(approval_id, item_id, payload_index)`. Before note effects, ensure planned folders depth-first. Existing paths are accepted only when they have the deterministic ID from this session or the resolved root choice is `USE_EXISTING`; otherwise mark dependent work failed with `folder_conflict`.

- [ ] **Step 4: Implement per-effect execution**

For each selected item/payload:

1. transition the session/item to running;
2. reconcile or apply note creation/content replacement;
3. reconcile or apply exact keyword state when content is created/replaced;
4. reconcile or attach every approved manual membership;
5. persist each effect as applied before advancing;
6. finalize the payload, item, and aggregate session state.

Skipped items receive a durable skipped outcome without target mutation. A failure stores only a bounded reason code and retryable flag. Never roll back already confirmed work or describe it as missing.

- [ ] **Step 5: Run happy-path tests GREEN**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_executor.py Tests/Notes/test_note_import_receipts.py
```

- [ ] **Step 6: Commit the bounded executor core**

Commit `feat(notes): execute approved import plans`.

---

### Task 5: Add cancellation, crash reconciliation, retry, and repeat observations

**Files:**

- Modify: `tldw_chatbook/Notes/note_import_executor.py`
- Modify: `tldw_chatbook/Notes/note_import_receipts.py`
- Modify: `Tests/Notes/test_note_import_executor.py`
- Modify: `Tests/Notes/test_note_import_receipts.py`

- [ ] **Step 1: Write RED bounded progress and responsiveness tests**

Cover batch sizes at 1, 2, and the configured ceiling; immutable monotonic progress; cancellation checked before the first batch and between later batches; no current target call interrupted; and an `asyncio` heartbeat that continues while a deliberately blocked real executor call runs through `execute_async()`.

The async facade must use:

```python
loop = asyncio.get_running_loop()

def publish(progress: ImportExecutionProgress) -> None:
    loop.call_soon_threadsafe(progress_callback, progress)

return await asyncio.to_thread(
    self.execute,
    approved,
    cancel_event=cancel_event,
    progress_callback=publish,
)
```

- [ ] **Step 2: Verify cancellation/responsiveness tests RED**

Confirm the heartbeat test fails if the executor body is deliberately run inline.

- [ ] **Step 3: Implement bounded progress and cooperative cancellation**

Validate `batch_size` between 1 and 100. Check `threading.Event` before every batch, persist `CANCELLED`, publish final progress, and leave unfinished item/effect rows pending for retry. Do not catch `KeyboardInterrupt`, `SystemExit`, or `GeneratorExit` as item failures.

- [ ] **Step 4: Write crash-window and retry RED tests**

Inject interruption after each target mutation but before its receipt transition:

- folder created;
- note created;
- note updated;
- keywords synchronized;
- membership attached;
- item finalized.

Reopen the receipt repository and rerun the same approved plan. Assert one note, one folder path, one membership per planned effect, no repeated optimistic update, and an honest completed receipt. Then prove `retry_failed()` rejects another plan/digest, selects only pending or retryable effects, and leaves permanent conflicts untouched.

- [ ] **Step 5: Implement target read-back reconciliation and failure-only retry**

Creation reconciliation requires the deterministic ID and exact payload match. Update reconciliation accepts only `(expected_version + 1)` with exact payload/keywords; any later version becomes `note_conflict`. Membership and folder reconciliation use their idempotent exact identities. Retry uses the original approval/session ID and private plan digest and never creates a second session.

- [ ] **Step 6: Write and implement prior-observation tests**

For a completed single-payload source, `prior_observations_for_plan(plan)` computes the current source-locator digest in memory, finds the latest confirmed receipt, and returns `PriorImportObservation` with receipt fingerprint, note ID, and final version. Multi-payload, failed, cancelled, missing, and permanently conflicted sources return no exact observation. Assert the repository and diagnostic reprs contain no source path, content, fingerprint, or note ID.

- [ ] **Step 7: Run the complete executor/receipt suite GREEN**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_execution_models.py Tests/Notes/test_note_import_receipts.py Tests/Notes/test_note_import_executor.py
```

- [ ] **Step 8: Commit recovery and retry behavior**

Commit `feat(notes): make import execution resumable`.

---

### Task 6: Verify the boundary and close the backlog task

**Files:**

- Modify: `backlog/tasks/task-16309 - Execute-approved-one-time-Database-Notes-import-plans-with-durable-receipts.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` only if this task produces a genuinely reusable incident.

- [ ] **Step 1: Run the focused gate**

```bash
../../.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  Tests/Notes/test_note_import_execution_models.py \
  Tests/Notes/test_note_import_receipts.py \
  Tests/Notes/test_note_import_executor.py \
  Tests/Notes/test_note_import_planner.py \
  Tests/Notes/test_note_import_windows_fs.py \
  Tests/Notes/test_note_folder_models.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_scope_service.py \
  Tests/Notes/test_notes_scope_service_folders.py \
  Tests/DB/test_private_sqlite.py \
  Tests/DB/test_private_sqlite_inventory.py
```

Expected: all runnable tests pass; only native-platform guards may skip with explicit reasons.

- [ ] **Step 2: Run static and structural gates**

Run Ruff check/format on every task-owned Python file, `git diff --check`, compile the touched modules with `python -B -m compileall`, and run the duplicate backlog-task-ID guard. Search the diff for absolute-path logging, raw exception text, note content in SQL/diagnostics, raw `sqlite3.connect`, and accidental backup/export registration.

- [ ] **Step 3: Run a real file-backed smoke test through pytest**

Use `tmp_path` to create a real ChaChaNotes database and private receipt database, plan one recursive folder import, approve and execute it through `execute_async`, reopen both databases, verify the note/folder/membership/receipt, then retry the same approval and prove no duplicate. Do not launch the application against the user's real profile.

- [ ] **Step 4: Self-review and request code review**

Review the full `origin/dev...HEAD` diff for approval authority, deterministic identity collisions, stale optimistic versions, crash gaps, transition legality, cancellation, event-loop confinement, private-SQLite registration/inventory, sensitive repr/logging, and behavior beyond the task ACs. Resolve every actionable finding with RED/GREEN evidence.

- [ ] **Step 5: Close task documentation**

Check every acceptance criterion, add concise implementation notes with exact verification counts and the ADR-059/073 decision, state whether a lesson was warranted, and change TASK-16309 to Done only after all Definition-of-Done gates pass.

- [ ] **Step 6: Commit closeout**

Commit `docs(backlog): close one-time notes import executor task` with explicit-path staging.

---

## Plan self-review

- Spec coverage: approved plan, local folder hierarchy, Create/Update/Skip, independent content/membership effects, durable itemized receipts, cancellation, retry, repeats, bounded execution, private storage, and server deferral are each mapped to a task above.
- Placeholder scan: no incomplete implementation marker remains; the only deferred capability is the explicitly out-of-scope server roadmap.
- Type consistency: `ApprovedNoteImportPlan`, `ImportExecutionProgress`, `ImportExecutionReceipt`, `NoteImportReceiptRepository`, `LocalNoteImportTarget`, and executor method names are stable across tasks.
- Scope check: no Library UI, sync root, watcher, server API, File Notes, or legacy sync mutation is included.
