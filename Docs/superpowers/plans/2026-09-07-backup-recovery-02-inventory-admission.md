# Storage inventory and maintenance admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete declared owner census and enforced startup/capture admission boundary.

**Architecture:** Implement the approved Backup_Recovery service through existing storage owners, with fixed startup admission and private immutable staging. Keep archive parsing separate from normal application startup, and expose only qualified operations through thin user views.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite/FTS5, Pydantic, stdlib ZIP64/file primitives, and a bundled helper built from official age Go library.

**Spec:** [Approved complete local backup and restore](../specs/2026-09-07-complete-local-backup-restore-design.md), revision 4.

## Global Constraints

- Python ≥3.11; retain the repository's current Textual 8.x dependency contract.
- “All identified Chatbook-owned local profiles and durable data are included by default, including configured database locations outside the default directory.”
- “External folders and model files are explicit options. Server-owned data has a separate recovery boundary and is not captured by this feature.”
- “Managed credentials are excluded by default. Including exportable credentials requires a password-encrypted archive.”
- “Partial archives support extraction and isolated recovery of validated dependency groups; they cannot replace current installation data in v1.”
- “Use .tldw-backup.zip for plaintext and .tldw-backup.zip.age for encrypted output.”
- “v1 reader defaults: 100,000 members; 16 MiB manifest; 1 TiB expanded payload; 256 GiB per member; 1,024 UTF-8 bytes per path.”
- “Default encrypted/plain input-container and decrypted-container budgets are each 2 TiB, independently enforced as actual bytes stream”; outer header 64 KiB; one KDF with 256 MiB working-memory budget.
- “No rebuild starts automatically, including a local rebuild.” Restored execution and reconnection require durable local owner review.
- “Run targeted checks only; a full suite needs separate user authorization.” Runtime tests listed here are required evidence, not results already obtained.
- “Do not expose Complete backup or destructive replacement before its full contract is qualified.” Keep per-operation/platform capability failures visible.
- ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: direct implementation of the approved storage, credential, archive, and recovery contract; reuse ADR-126 with ADR-029/030/036/059/060.

---

[Delivery order and cross-plan contracts](2026-09-07-complete-local-backup-restore.md).

## Execution discipline

Read the approved spec and the matching Backlog task before implementation. Move only
the current task to In Progress, then copy its steps into that task's Implementation
Plan before touching production code. Tasks still To Do have design references and
acceptance criteria, not premature implementation notes. Use an isolated worktree at
execution time; this planning checkout contains unrelated staged/unstaged work.
Activate the project Python ≥3.11 environment before the commands below, for example
`source .venv/bin/activate`; the system `python3` on this host is older and must not
be used for application-test evidence.

The interface code fences below specify signatures, not stub implementations to ship.
Invariant fragments belong inside the named implementation and do not replace the
behavioral steps. All Python types use stdlib dataclasses/pathlib/threading/typing;
untrusted serialized data is validated with strict Pydantic models at the boundary.
`StorageItem`, `Inventory`, `SchemaPolicy`, and `OwnerAdapter` are defined in task 3;
`ArchiveLimits`/`SealedArchive` in task 12; `CaptureResult` in task 15;
`RestorePlan` in task 17; `Journal` in task 18. Import these definitions, do not
create lookalike cross-module types. Relative file paths below are repository-relative.

Capture options are validated at the service boundary: `external_roots: tuple[Path, ...]`,
`model_ids: tuple[str, ...]`, `temporary_media: bool`, `diagnostics: bool`,
`credential_mode: Literal["exclude", "include", "rollback"]`, `encrypted: bool`,
`allow_partial: bool`, and `limits: ArchiveLimits`. Unknown keys are rejected. Missing
optional collections/booleans mean empty/false; credentials default to exclude.
The backup UI cannot select rollback mode; only the local replacement executor can.
Inventory status vocabulary is included, included_directory, intentionally_excluded,
unused, intentionally_deleted, unavailable, unsupported, and missing_required. Only
validated owner tombstones may produce intentionally_deleted.

Frozen tuples carry snapshot data. A scope digest includes selected profile/config
selectors, owner/root identities, shared groups, dependencies, exclusions, and budgets;
it excludes ordinary row counts and changing record contents. Capture payload digests
and target fingerprints serve separate purposes and must not be substituted for it.
String status/error codes in tests are fixed sanitized codes, never raw exception text.
Owner-private method names in invariant fragments are local implementation details,
not undeclared public services. Define and test them in the same task if retained.

Every test example is the first focused red case, followed by the listed behavioral
matrix. Import failure may start a new module's TDD loop, but establish a behavioral
red failure after skeleton importability before claiming regression evidence. Use
real temporary SQLite/files/processes, owner APIs, and subprocess synchronization;
mock only external keyrings/network/process-effect sentinels where explicitly stated.
All app-importing tests live under Tests/ and its isolation fixture. Never read the
developer's real config, keychain, models, or databases to construct fixtures.

Run the named test file after each small behavior change, then the task's listed
guards once. Run changed Python through `ruff check --select E9,F63,F7,F82` and
new focused modules through `ruff format --check`; use `gofmt -l` and `go vet` for
Go changes. Provision these development tools in the isolated execution environment
if absent; do not reformat large unrelated existing modules. Record baseline lint
failures separately with a clean-HEAD reproduction. Run `git diff --check` before
the scoped commit. New SQLite/private writers update their existing inventories;
diagnostic changes run the production diagnostic inventory guard. No full suite or
collection sweep is implied by these targeted commands.

For each task: review the diff, add implementation notes with the ADR and actual
commands/outcomes, update documentation, then check criteria and mark Done only
after evidence passes. The five-digit Backlog CLI bug is documented in
`backlog/docs/lessons-backlog-hygiene.md`; verify the resulting file, use its documented
direct-file fallback if necessary, and never let a malformed CLI file reach a commit.
Commit only that task's changed files with the provided subject. Existing staged
changes in another checkout are not part of this work.

## File structure

- Task 3 owns `tldw_chatbook/Backup_Recovery/models.py`, `tldw_chatbook/Backup_Recovery/inventory.py`, `tldw_chatbook/Backup_Recovery/profile_paths.py`, `tldw_chatbook/Backup_Recovery/owner_registry.py`, `backlog/docs/backup-recovery-owner-inventory.md`.
- Task 4 owns `tldw_chatbook/Backup_Recovery/admission.py`, `tldw_chatbook/Backup_Recovery/native_files.py`, `tldw_chatbook/Backup_Recovery/qualification.py`.
- Task 5 owns `tldw_chatbook/Backup_Recovery/bootstrap.py`, `tldw_chatbook/Backup_Recovery/control_records.py`.
- Task 6 owns `tldw_chatbook/DB/recovery_core.py`.
- Task 7 owns `tldw_chatbook/Research_Interop/recovery.py`, `tldw_chatbook/Writing_Interop/recovery.py`, `tldw_chatbook/Study_Interop/recovery.py`, `tldw_chatbook/Evals/recovery.py`.
- Task 8 owns `tldw_chatbook/Workspaces/recovery.py`, `tldw_chatbook/Scheduling/recovery.py`, `tldw_chatbook/Sync_Interop/recovery.py`, `tldw_chatbook/Notes/recovery.py`, `tldw_chatbook/MCP/recovery.py`, `tldw_chatbook/Notifications/recovery.py`, `tldw_chatbook/DB/recovery_operations.py`.
- Task 9 owns `tldw_chatbook/Backup_Recovery/file_inventory.py`, `tldw_chatbook/Backup_Recovery/config_adapter.py`, `tldw_chatbook/TTS/recovery.py`, `tldw_chatbook/Persona_Visual/recovery.py`, `tldw_chatbook/Skills_Interop/recovery.py`, `tldw_chatbook/Model_Artifacts/recovery.py`.
- Task 10 owns `tldw_chatbook/Backup_Recovery/participants.py`.

<a id="task-3"></a>
## Task 3: Declare recovery inventory and side-effect-free profile discovery

**Backlog:** [TASK-31986](../../../backlog/tasks/task-31986%20-%20Declare-recovery-inventory-and-side-effect-free-profile-discovery.md) — Done.

**Dependencies:** Approved design only.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/models.py`
- Create: `tldw_chatbook/Backup_Recovery/inventory.py`
- Create: `tldw_chatbook/Backup_Recovery/profile_paths.py`
- Create: `tldw_chatbook/Backup_Recovery/owner_registry.py`
- Create: `backlog/docs/backup-recovery-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_inventory.py`
- Create: `Tests/Architecture/test_backup_owner_inventory.py`
- Modify: `tldw_chatbook/config.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# models.py: frozen dataclasses; collections are tuples/frozensets, no mutable defaults.
@dataclass(frozen=True)
class StorageItem:
    owner: str
    logical_id: str
    path: Path | None
    status: str
    dependencies: tuple[str, ...]
@dataclass(frozen=True)
class Inventory:
    items: tuple[StorageItem, ...]
    complete: bool
    scope_digest: str
    issues: tuple[str, ...]
def discover(config_paths: tuple[Path, ...]) -> Inventory: ...
def classify_entries(items: tuple[StorageItem, ...]) -> Inventory: ...

@dataclass(frozen=True)
class SchemaPolicy:
    owner: str
    versions: tuple[int, ...]
    schema_sql: tuple[tuple[int, tuple[str, ...]], ...]
    migration_steps: tuple[tuple[int, int, tuple[str, ...]], ...]
class OwnerAdapter(Protocol):
    owner_id: str
    activation_required: bool
    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]: ...
    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None: ...
    def validate(self, candidate: Path) -> tuple[str, ...]: ...
    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None: ...
    def schema_policy(self) -> SchemaPolicy | None: ...
# owner_registry.py
def register(adapter: OwnerAdapter) -> None: ...
def registered() -> tuple[OwnerAdapter, ...]: ...
```

- [x] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_inventory.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_unknown_durable_entry_blocks_completeness(tmp_path):
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    item = StorageItem("unknown", "unclassified", tmp_path / "new.db", "unsupported", ())
    result = classify_entries((item,))
    assert result.complete is False
    assert "unsupported_owner" in result.issues
```

- [x] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_inventory.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [x] **Step 3:** Create a census from SQLITE_OWNER_REGISTRY, backlog/docs/sqlite-private-owner-inventory.md, configured storage resolvers, private file writers, and existing durable root contents. Record every producer, path resolver, class, dependencies, capture/validation/relocation/activation adapter, cohort, and targeted evidence; classify memory, cookies/external input, process artifacts, and server data explicitly. A raw connection census alone is insufficient.

- [x] **Step 4:** Extract pure path resolution from config.py without changing priority rules. discover reads explicitly selected TOML configs and canonical defaults without load_settings, directory creation, fallback profile creation, optional-engine imports, keyring reads, or database constructors. Return a parse failure for damaged source configs; archive inspection does not call discover.

- [x] **Step 5:** Define frozen SchemaPolicy(owner, versions, schema_sql, migration_steps) and OwnerAdapter protocol here. discover(config: Mapping[str, object]) -> tuple[StorageItem, ...]; capture(item: StorageItem, destination: Path, cancel: Event) -> None; validate(candidate: Path) -> tuple[str, ...]; relocate(candidate: Path, mapping: Mapping[str, Path]) -> None; schema_policy() -> SchemaPolicy | None. Policies contain installed SQL only and exact supported schema metadata, never imported executable code.

- [x] **Step 6:** owner_registry.register(adapter: OwnerAdapter) -> None and registered() -> tuple[OwnerAdapter, ...] reject duplicate logical ownership unless verified physical identity maps it to an explicit shared group. Path aliases and nested owner roots cannot silently duplicate capture. Unknown durable entries block complete; unavailable required payloads do too, except validated intentional-deletion records.

- [x] **Step 7:** Add a source census guard for new persistence producers and a documented explicit exclusion review. Fixtures cover inactive features, multiple profile configs, custom DB paths, aliasing, unknown files, required absence, external folders, and malformed config. Do not mark the implementation coverage complete until every census row has a supported adapter or approved exclusion.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Inside classify_entries(); actual dependency/alias checks follow this gate.
blocking = {"unsupported", "unavailable", "missing_required"}
complete = all(item.status not in blocking for item in items)
if any(item.owner == "unknown" for item in items):
    complete = False
    issues.append("unsupported_owner")
```

- [x] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_inventory.py Tests/Architecture/test_backup_owner_inventory.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [x] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [x] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): declare recovery inventory and side-effect-free profile discovery`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Discovery identifies selected profiles, custom app-owned storage, shared aliases, and durable unknowns without opening services or modifying data.
- Coverage states and dependency failures accurately distinguish complete, partial, unavailable, excluded, and intentional deletion.
- Every existing persistence producer has an explicit owner-inventory row, and new unclassified producers fail an architecture guard.


<a id="task-4"></a>
## Task 4: Implement stable maintenance admission and native storage qualification

**Backlog:** [TASK-31987](../../../backlog/tasks/task-31987%20-%20Implement-stable-maintenance-admission-and-native-storage-qualification.md) — Done.

**Dependencies:** TASK-31986.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/admission.py`
- Create: `tldw_chatbook/Backup_Recovery/native_files.py`
- Create: `tldw_chatbook/Backup_Recovery/qualification.py`
- Create: `Tests/Backup_Recovery/test_admission.py`
- Create: `Tests/Backup_Recovery/test_native_files.py`
- Modify: `tldw_chatbook/Utils/instance_lock.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
class Admission:
    def __init__(self, control_root: Path): ...
    def normal(self, namespaces: tuple[str, ...]) -> ContextManager[None]: ...
    def maintenance(self, namespaces: tuple[str, ...], timeout: float) -> ContextManager[None]: ...
def qualified_for(operation: str, root: Path) -> tuple[bool, str]: ...
def publish_new(staged: Path, destination: Path) -> None: ...
```

- [x] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_admission.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_publication_never_overwrites_existing_file(tmp_path):
    import pytest
    from tldw_chatbook.Backup_Recovery.native_files import publish_new
    staged, destination = tmp_path / "stage", tmp_path / "good"
    staged.write_bytes(b"new")
    destination.write_bytes(b"previous backup")
    with pytest.raises(FileExistsError):
        publish_new(staged, destination)
    assert destination.read_bytes() == b"previous backup"
```

- [x] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_admission.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [x] **Step 3:** Implement namespace registration, alias identity, and native cross-process admission under a verified control root outside managed targets. Keep lock objects stable across rename and reserve old/new namespaces during remapping. Do not change advisory InstanceLockStatus into a claim that legacy processes are fenced.

- [x] **Step 4:** Acquire ordered registry/owner admission without holding locks needed by a draining participant. Closing admission prevents new mutations, then drains established participants to a safe boundary before exclusive capture. Add timeout/cancellation results without forced transaction rollback or draft loss.

- [x] **Step 5:** Provide owner-private directory/file creation and durable atomic no-replace publication using qualified native primitives. Handle same-volume hard-link publication only for verified regular operation-owned files on a qualified filesystem; support crash evidence for the temporary dual-name state. Never emulate no-replace with exists then replace.

- [x] **Step 6:** Use independent child processes and synchronization pipes/events for lock tests, including changing the target inode, shared aliases, remapping, connection retirement, process death, stale registry evidence, and contention. Test native directory fsync/flush and refusal on unqualified storage; record evidence per operation, not merely per OS.

- [x] **Step 7:** qualified_for returns unavailable until the corresponding native test evidence exists. Distinguish archive-output support, isolated publication, and replacement support. Known incompatible clients refuse maintenance; PID scanning is supplementary only.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Native operation dispatch must refuse missing qualification.
allowed, reason = qualified_for("publish_new", destination.parent)
if not allowed:
    raise OSError(reason)
# Dispatch to the qualified atomic no-replace primitive with pinned parents;
# FileExistsError from that primitive is final, never retried with replace().
```

- [x] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_admission.py Tests/Backup_Recovery/test_native_files.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [x] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [x] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): implement stable maintenance admission and native storage qualification`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
- Deadlock, timeout, cancellation, and crashed-holder cases preserve data and durable recovery evidence.
- Native publication cannot overwrite an existing artifact, and unsupported storage returns an explicit unavailable capability.


<a id="task-5"></a>
## Task 5: Fence all supported startup routes before storage bootstrap

**Backlog:** [TASK-31988](../../../backlog/tasks/task-31988%20-%20Fence-all-supported-startup-routes-before-storage-bootstrap.md) — To Do.

**Dependencies:** TASK-31986, TASK-31987.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/bootstrap.py`
- Create: `tldw_chatbook/Backup_Recovery/control_records.py`
- Modify: `tldw_chatbook/cli.py`
- Modify: `tldw_chatbook/__main__.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Web_Server/serve.py`
- Modify: `tldw_chatbook/MCP/__main__.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Create: `Tests/Backup_Recovery/test_bootstrap.py`
- Create: `Tests/Architecture/test_recovery_entrypoints.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# bootstrap.py is stdlib/private-path only; never imports app/config.
def startup_permission(config_selector: Path, bootstrap_root: Path) -> tuple[bool, str]: ...
# control_records.py
def register_pending(bootstrap_root: Path, operation_id: str,
                     namespaces: tuple[str, ...], control_root: Path,
                     selectors: tuple[Path, ...]) -> None: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_bootstrap.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_custom_root_pending_operation_blocks_startup(tmp_path):
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    bootstrap, control, config = tmp_path / "bootstrap", tmp_path / "control", tmp_path / "broken.toml"
    config.write_text("this is not TOML [")
    register_pending(bootstrap, "op1", ("profile1",), control, (config,))
    allowed, reason = startup_permission(config, bootstrap)
    assert allowed is False
    assert reason == "recovery_pending"
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_bootstrap.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Store versioned private admission associations in the fixed default bootstrap directory, with locally verified selectors/namespaces and custom control roots. Reject overlap with replacement targets; archives cannot supply these records. Register durably before publication.

- [ ] **Step 4:** Move the minimal check ahead of app/config imports for console script and python -m, and ahead of config reads for direct app, web-server, MCP, and headless persistence launchers. Inventory actual supported routes with AST/import subprocess tests, including multiprocessing spawn behavior. Refuse uncertain scope rather than guessing disjointness.

- [ ] **Step 5:** Add the same admission contract at the registered private SQLite open seam and owned file entry boundaries so a missed high-level launcher cannot advertise full protection. Memory-only/foreign read-only sources retain their classified exemptions; ordinary admitted opens must be released on shutdown.

- [ ] **Step 6:** A pending or corrupt scope returns a recovery-required result without migrations, cleanup, default config creation, process spawning, or network activity. At this stage print a bounded recovery-required message; the recovery UI arrives in its own task. Plain unrelated profile startup is allowed only with positively verified disjoint mappings.

- [ ] **Step 7:** Test custom-root disappearance, corrupt fixed records, catalog loss, interrupted registration, config replacement, and raw direct-app launch with import-time sentinels. Preserve unknown records. Clearing fences will be wired only through the later journal commit API.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# In the lightweight entry point, before importing app/config:
allowed, reason = startup_permission(config_selector, bootstrap_root)
if not allowed:
    raise SystemExit("Recovery required: " + reason)
# Ordinary application loading follows only on the admitted branch.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_bootstrap.py Tests/Architecture/test_recovery_entrypoints.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): fence all supported startup routes before storage bootstrap`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Every supported launch path checks fixed bootstrap admission before affected storage or runtime composition.
- Damaged config and inaccessible custom recovery roots cannot bypass a pending operation.
- Provably disjoint profiles remain usable; ambiguous scope is blocked without deleting evidence.


<a id="task-6"></a>
## Task 6: Add recovery adapters for core conversation and library stores

**Backlog:** [TASK-31989](../../../backlog/tasks/task-31989%20-%20Add-recovery-adapters-for-core-conversation-and-library-stores.md) — To Do.

**Dependencies:** TASK-31986, TASK-31987.

**Files and ownership:**

- Create: `tldw_chatbook/DB/recovery_core.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/DB/Library_Collections_DB.py`
- Modify: `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `backlog/docs/backup-recovery-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_core_owners.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# recovery_core.py: installed data/schema policies, no eager store construction.
def core_adapters() -> tuple[OwnerAdapter, ...]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_core_owners.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_core_owner_set_is_declared():
    from tldw_chatbook.DB.recovery_core import core_adapters
    names = {adapter.owner_id for adapter in core_adapters()}
    assert {"db.chachanotes.primary", "db.media.primary", "db.prompts.primary"} <= names
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_core_owners.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Declare core store resolvers and dependency groups for conversations/messages/characters/notes, prompts, media, collections, and ingestion history. Include soft-deleted rows and referenced attachments; do not export records through selective Chatbook serializers.

- [ ] **Step 4:** Implement capture with registered backup_connection_to_private or copy_private_sqlite owners. Snapshot committed WAL through SQLite while maintenance is held; never independently copy sidecars. Register recovery-specific backup authority where current centralized-backup policy disallows it; keep ordinary selective export exclusions intact.

- [ ] **Step 5:** Describe exact supported schema/FTS/trigger definitions and installed migration steps through SchemaPolicy. Add relocation of managed paths and domain reference validation without normal constructors. Unsupported historical versions stay explicit rather than attempting best-effort upgrades.

- [ ] **Step 6:** Create real database fixtures through current domain APIs under Tests isolation, then compare primary keys, relationships, soft deletions, FTS content, BLOBs, and byte assets after adapter capture. Add per-store WAL and interrupted-capture cases; the declaration smoke test below is only the smallest red step, not completion evidence.

- [ ] **Step 7:** Update both owner inventories and run private SQLite census/interop guards alongside these focused round trips.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Inside a SQLite core adapter's capture method:
def check_cancelled() -> None:
    if cancel.is_set():
        raise InterruptedError("cancelled")
copy_private_sqlite(self.backup_owner_id, item.path, destination,
                    progress_guard=check_cancelled)
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_core_owners.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): add recovery adapters for core conversation and library stores`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Core durable records and assets survive a real SQLite/WAL capture with stable identities and relationships.
- Source files remain unchanged, custom paths are honored, and missing required dependencies block completeness.
- Schema/relocation policies and registered private backup authority are explicit for every core owner.


<a id="task-7"></a>
## Task 7: Add recovery adapters for local research writing study and evaluation data

**Backlog:** [TASK-31990](../../../backlog/tasks/task-31990%20-%20Add-recovery-adapters-for-local-research-writing-study-and-evaluation-data.md) — To Do.

**Dependencies:** TASK-31986, TASK-31987.

**Files and ownership:**

- Create: `tldw_chatbook/Research_Interop/recovery.py`
- Create: `tldw_chatbook/Writing_Interop/recovery.py`
- Create: `tldw_chatbook/Study_Interop/recovery.py`
- Create: `tldw_chatbook/Evals/recovery.py`
- Modify: `tldw_chatbook/Research_Interop/local_research_service.py`
- Modify: `tldw_chatbook/Writing_Interop/local_writing_service.py`
- Modify: `tldw_chatbook/Study_Interop/local_study_service.py`
- Modify: `tldw_chatbook/Study_Interop/local_quiz_service.py`
- Modify: `tldw_chatbook/DB/Evals_DB.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `backlog/docs/backup-recovery-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_domain_owners.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# Each listed recovery.py provides the same cohort factory.
def recovery_adapters() -> tuple[OwnerAdapter, ...]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_domain_owners.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_optional_domain_discovery_does_not_require_engines():
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters
    adapters = recovery_adapters()
    assert adapters
    assert all(adapter.schema_policy() is not None for adapter in adapters)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_domain_owners.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Implement OwnerAdapter for local research/writing/study/quizzes/evaluation stores from the verified census, including secondary benches, custom roots, and persisted outputs. Classify mirrors and server-owned data separately; local durable user data remains baseline even when its feature is disabled.

- [ ] **Step 4:** Preserve history, attachments, identifiers, relationships, deleted/recovery records and schema metadata. Populate SchemaPolicy with supported installed versions and domain validation; relocate only managed locators. Register checked capture authority without broadening unrelated exports.

- [ ] **Step 5:** Keep discovery import-light: no cloud clients, evaluation runner/model imports, or database constructors that migrate. Capture under existing domain admission and emit explicit unavailable results for unreadable required stores.

- [ ] **Step 6:** Add a fixture for each census owner using the real domain persistence API, capture it, and compare complete domain records and referenced bytes. Include no optional engine installed, schema-too-new, old supported schema, missing asset, and custom path cases.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Register the concrete cohort factories at inventory composition.
for adapter in recovery_adapters():
    register(adapter)
# Factories return installed path/schema policies, not live service objects.
```

- [ ] **Step 7:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_domain_owners.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [ ] **Step 8:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 9:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): add recovery adapters for local research writing study and evaluation data`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Inactive optional features retain their existing local durable data in baseline inventory.
- All census owners in this cohort have lossless data/asset capture, checked schemas, and relocation evidence.
- Discovery and validation never start evaluations, models, or server requests.


<a id="task-8"></a>
## Task 8: Add recovery adapters for workspace operational and device-local state

**Backlog:** [TASK-31991](../../../backlog/tasks/task-31991%20-%20Add-recovery-adapters-for-workspace-operational-and-device-local-state.md) — To Do.

**Dependencies:** TASK-31986, TASK-31987.

**Files and ownership:**

- Create: `tldw_chatbook/Workspaces/recovery.py`
- Create: `tldw_chatbook/Scheduling/recovery.py`
- Create: `tldw_chatbook/Sync_Interop/recovery.py`
- Create: `tldw_chatbook/Notes/recovery.py`
- Create: `tldw_chatbook/MCP/recovery.py`
- Create: `tldw_chatbook/Notifications/recovery.py`
- Create: `tldw_chatbook/DB/recovery_operations.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `backlog/docs/backup-recovery-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_operational_owners.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# Each cohort recovery.py exposes recovery_adapters(); the DB cohort owns
# Workspace_DB, AgentRuns_DB, Subscriptions_DB and their local dependencies.
def recovery_adapters() -> tuple[OwnerAdapter, ...]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_operational_owners.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_sync_capture_policy_keeps_runtime_authority_inactive():
    from tldw_chatbook.Sync_Interop.recovery import recovery_adapters
    adapters = recovery_adapters()
    assert adapters
    assert all(adapter.activation_required for adapter in adapters)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_operational_owners.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Capture workspace registry/state, agent run history, subscriptions, notifications/cursors, scheduling definitions/history, local MCP/permission stores, File Notes recovery data, and device-local Notes sync journals according to each owner census row. Memory-only Sync keys remain uncaptured.

- [ ] **Step 4:** Add activation_required: bool to OwnerAdapter with False for passive content and True for execution/reconnection owners. Preserve imported operational bytes as quarantined evidence; relocate fresh identity/scope bindings separately from immutable historical references. Never replay pending filesystem intent during capture/inspection.

- [ ] **Step 5:** Implement owner-safe record export for recovery and checked SQLite snapshots without changing ADR-059/060 selective bundle restrictions. Distinguish File Notes disk authority from database projections; external folder capture stays opt-in.

- [ ] **Step 6:** Test historical running/queued states, permissions, leases, device claims, file bindings, pending intents, and local recovery bytes survive capture but cannot authorize a write. Include shared config/keyring scopes, optional-disabled services, and no silent conversion of managed folder membership.

- [ ] **Step 7:** Keep adapters near their actual owners; DB/recovery_operations.py aggregates only database policies. Add new census-discovered local operational owners to this cohort with explicit rows rather than scanning home or ignoring them.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Operational adapter declaration; capture never promotes these bytes.
activation_required = True
# Relocation preserves historical IDs while allocating new live claims.
if imported_binding_is_authoritative:
    raise ValueError("imported_authority_not_admitted")
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_operational_owners.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): add recovery adapters for workspace operational and device-local state`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Operational history and recoverable device-local bytes are retained while their execution authority remains quarantined.
- File Notes ownership and ordinary export exclusions remain consistent with ADR-021/059/060.
- All operational persistence census rows have capture, schema, relocation, and activation classification.


<a id="task-9"></a>
## Task 9: Add recovery inventory for configuration durable files and optional content

**Backlog:** [TASK-31992](../../../backlog/tasks/task-31992%20-%20Add-recovery-inventory-for-configuration-durable-files-and-optional-content.md) — To Do.

**Dependencies:** TASK-31986, TASK-31987.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/file_inventory.py`
- Create: `tldw_chatbook/Backup_Recovery/config_adapter.py`
- Create: `tldw_chatbook/TTS/recovery.py`
- Create: `tldw_chatbook/Persona_Visual/recovery.py`
- Create: `tldw_chatbook/Skills_Interop/recovery.py`
- Create: `tldw_chatbook/Model_Artifacts/recovery.py`
- Modify: `tldw_chatbook/Utils/path_validation.py`
- Modify: `backlog/docs/backup-recovery-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_file_inventory.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def inventory_tree(root: Path, *, owner: str, external: bool) -> tuple[StorageItem, ...]: ...
def config_adapter() -> OwnerAdapter: ...
# TTS/Persona/Skills/Model_Artifacts recovery.py each exports recovery_adapters().
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_file_inventory.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_empty_directory_is_an_inventory_item(tmp_path):
    from tldw_chatbook.Backup_Recovery.file_inventory import inventory_tree
    empty = tmp_path / "empty"
    empty.mkdir()
    items = inventory_tree(tmp_path, owner="test.files", external=True)
    assert any(item.path == empty and item.status == "included_directory" for item in items)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_file_inventory.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Declare config/current history, templates, internal prompts, skill/local definitions, persona artwork, voice references and TTS profile store, durable generated/saved assets, and remaining private-file census owners. Keep diagnostic files, disposable caches, backup outputs, journals, and rollback directories explicitly classified.

- [ ] **Step 4:** Represent regular files and empty directories with stable logical IDs and explicit parent relationships; detect file/directory collisions, aliases, links, nested mounts, unsupported metadata, and unknown durable entries. Use checked path helpers; no recursive symlink following.

- [ ] **Step 5:** Model external roots, model payloads, diagnostics, and current temporary media as separate opt-in selections. A configured app database outside the default directory remains baseline. Qualified model adapters may resolve only selected dependencies inside an identified store; do not follow generic links.

- [ ] **Step 6:** Config adapters expose known managed secret locations for the credential task and pure relocation rules. They do not sanitize arbitrary user prose or decrypt unknown secret blobs opportunistically.

- [ ] **Step 7:** Test nested empty trees, private permissions, custom assets, selected model dependency links, excluded output roots, and unavailable external roots. Reconcile every remaining file census row; unresolved rows must keep Complete unavailable.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Inside inventory_tree(), inspect entries without following links.
entry_mode = entry.lstat().st_mode
if stat.S_ISLNK(entry_mode):
    issues.append("unsupported_link")
elif stat.S_ISDIR(entry_mode):
    directories.append(entry)
# Persist even empty directories; checked owner paths still govern descent.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_file_inventory.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): add recovery inventory for configuration durable files and optional content`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
- External folders/models/temporary media/diagnostics remain opt-in without weakening custom app-owned storage coverage.
- Aliases, unknown durable entries, missing required files, and unsupported metadata produce truthful coverage results.


<a id="task-10"></a>
## Task 10: Integrate maintenance participants across persistence owners

**Backlog:** [TASK-31993](../../../backlog/tasks/task-31993%20-%20Integrate-maintenance-participants-across-persistence-owners.md) — To Do.

**Dependencies:** TASK-31988, TASK-31989, TASK-31990, TASK-31991, TASK-31992.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/participants.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `tldw_chatbook/Notes/sync_engine.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/Scheduling/services/scheduling_service.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `backlog/docs/backup-recovery-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_participants.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
class Participant(Protocol):
    owner_id: str
    def close_admission(self) -> None: ...
    def drain(self, deadline: float) -> bool: ...
    def resume(self) -> None: ...
def require_participant_coverage(inventory: Inventory,
                                 participants: tuple[Participant, ...]) -> None: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_participants.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_uncovered_persistent_owner_refuses_maintenance(tmp_path):
    import pytest
    from tldw_chatbook.Backup_Recovery.models import StorageItem, Inventory
    from tldw_chatbook.Backup_Recovery.participants import require_participant_coverage
    inventory = Inventory((StorageItem("new.writer", "db", tmp_path / "db", "included", ()),), True, "scope", ())
    with pytest.raises(ValueError, match="participant_missing"):
        require_participant_coverage(inventory, ())
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_participants.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Bind one participant per registered persistence owner in app composition, including background jobs, config saves, file writes, model/artifact publication, and headless routes. Shared files map to one admission namespace; passive owners document why no writer participant is needed.

- [ ] **Step 4:** Close new mutation admission, await committed transactions and cross-store asset publication, stop watchers at their existing safe boundary, and release normal holds before exclusive maintenance. A dirty unsaved editor returns needs-user-save/discard rather than automatically losing text.

- [ ] **Step 5:** Preserve pending work at a durable recoverable boundary; unresolved bytes or ownership block complete capture. Resume in reverse dependency order only after the capture coordinator releases its lease. Replacement will hold this boundary until installed validation.

- [ ] **Step 6:** Use two real processes to write related DB records/assets and config/registry changes; assert no captured generation combines mismatched dependencies. Add drain timeout, app-close, participant failure, and deadlock-order fixtures. Complete capability requires the census coverage guard, not a hand-maintained allow-all flag.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Inside require_participant_coverage():
covered = {participant.owner_id for participant in participants}
required = {item.owner for item in inventory.items if item.path is not None}
if required - covered:
    raise ValueError("participant_missing")
# Explicitly classified passive sources are removed from required first.
```

- [ ] **Step 7:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_participants.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

- [ ] **Step 8:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 9:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): integrate maintenance participants across persistence owners`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Every participating persistence owner drains safely and is covered by the shared admission protocol.
- Unsaved drafts and unfinished cross-store work are neither discarded nor falsely reported captured.
- Real multi-process evidence proves coherent ownership boundaries and safe resumption without deadlocks.
