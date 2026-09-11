# Restore publication and recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Both restore destinations, persistent activation safety, exact rollback, and interruption recovery.

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

- Task 17 owns `tldw_chatbook/Backup_Recovery/restore_plan.py`, `tldw_chatbook/Backup_Recovery/staging.py`.
- Task 18 owns `tldw_chatbook/Backup_Recovery/journal.py`, `tldw_chatbook/Backup_Recovery/publication.py`.
- Task 19 owns `tldw_chatbook/RAG_Search/recovery.py`.
- Task 20 owns `tldw_chatbook/Backup_Recovery/activation.py`.
- Task 21 owns `tldw_chatbook/Backup_Recovery/profile_catalog.py`, `tldw_chatbook/Backup_Recovery/isolated_restore.py`.
- Task 22 owns `tldw_chatbook/Backup_Recovery/replacement.py`.
- Task 23 owns `tldw_chatbook/Backup_Recovery/recovery_copies.py`, `tldw_chatbook/Backup_Recovery/recovery_service.py`.

<a id="task-17"></a>
## Task 17: Plan and stage both restore destinations with explicit dependency mapping

**Backlog:** [TASK-32000](../../../backlog/tasks/task-32000%20-%20Plan-and-stage-both-restore-destinations-with-explicit-dependency-mapping.md) — To Do.

**Dependencies:** TASK-31995, TASK-31996, TASK-31997, TASK-31999.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/restore_plan.py`
- Create: `tldw_chatbook/Backup_Recovery/staging.py`
- Create: `Tests/Backup_Recovery/test_restore_plan.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
@dataclass(frozen=True)
class RestorePlan:
    archive_digest: str
    mode: str
    restore: tuple[tuple[str, Path], ...]
    retire: tuple[tuple[str, Path], ...]
    preserve: tuple[tuple[str, Path], ...]
    target_fingerprint: str
def plan_restore(archive: SealedArchive, *, mode: Literal["isolated", "replace"],
                 destinations: Mapping[str, Path], target: Inventory | None) -> RestorePlan: ...
def stage_restore(archive: SealedArchive, plan: RestorePlan,
                  work_root: Path, cancel: Event) -> Path: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_restore_plan.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_replace_requires_independent_target_inventory(tmp_path):
    import pytest
    from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    archive = SealedArchive(tmp_path / "archive", "digest", b"{}")
    with pytest.raises(ValueError, match="target_unverified"):
        plan_restore(archive, mode="replace", destinations={}, target=None)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_restore_plan.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Validate immutable archive input before destination discovery. Isolated planning takes only archive/new local targets and never parses damaged current config. Replacement independently verifies target locators from local admission/catalog records or user-selected owner identity; no fallback defaults or archive absolute-path authority.

- [ ] **Step 4:** Resolve restore/retire/preserve sets from explicit producer inventory plus current owner classification. Include target-only obsolete state/sidecars in retirement with rollback coverage. Unknown newer owners block until classified. Partial archives cannot replace; isolated recovery admits only validated dependency groups.

- [ ] **Step 5:** Expand/refuse shared config/database/index effects explicitly. Remap managed paths and fresh profile/device identities, reject aliases and collisions, reserve old/new namespaces, restore external folders only into new directories, and keep models inert. Stage directories/files on qualified target volumes and apply final supported metadata last.

- [ ] **Step 6:** Bind plan to archive digest and target fingerprints; validate candidate schema/domain/assets and credential mappings in private staging. Recheck space by volume. Execution must invalidate a changed archive or destination, not recalculate paths silently.

- [ ] **Step 7:** Test both modes with shared stores, damaged current config, target-only records, case/Unicode collisions, unsupported metadata, older/newer schema, missing engines, empty roots, original-path aliases, and partial groups.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# At plan_restore entry, independent of archive schema parsing:
if mode == "replace" and target is None:
    raise ValueError("target_unverified")
# Incoming original paths cannot populate destinations; caller supplies
# locally verified mappings and independently discovered target inventory.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_restore_plan.py -q
git diff --check
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): plan and stage both restore destinations with explicit dependency mapping`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
- Replacement preserves unknown data until reviewed and includes managed retirement/rollback scope.
- Isolated restore works independently of damaged current config and rejects source aliases or untrusted destination authority.


<a id="task-18"></a>
## Task 18: Implement durable publication journal and crash reconciliation

**Backlog:** [TASK-32001](../../../backlog/tasks/task-32001%20-%20Implement-durable-publication-journal-and-crash-reconciliation.md) — To Do.

**Dependencies:** TASK-31987, TASK-31988, TASK-32000.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/journal.py`
- Create: `tldw_chatbook/Backup_Recovery/publication.py`
- Create: `Tests/Backup_Recovery/test_publication_crashes.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
class Journal:
    def __init__(self, root: Path, operation_id: str): ...
    def record(self, event: str, evidence: Mapping[str, object]) -> None: ...
    def recover(self) -> str: ...
def publish_candidate(candidate: Path, plan: RestorePlan, journal: Journal,
                      rollback_archive: Path | None) -> None: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_publication_crashes.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_incomplete_publication_remains_recovery_required(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal
    journal = Journal(tmp_path, "operation-1")
    journal.record("prepared", {"generation": "g1", "mode": "isolated"})
    journal.record("publication_started", {})
    assert Journal(tmp_path, "operation-1").recover() == "recovery_required"
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_publication_crashes.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Use versioned private durable records for prepared, rollback_verified, publication_started, artifact_published, installed_validated, activation_recorded, committed, rollback_started, and recovery_required. The record schema validates required evidence per event and state transition; unknown/corrupt events do not imply success.

- [ ] **Step 4:** Before publication durably register the fixed bootstrap pointer and affected namespaces. Record previous/candidate identity/digests, target-volume paths, generations, verified rollback reference, retirement intent, and artifact-level progress. Secrets never enter journals.

- [ ] **Step 5:** Publish only through qualified native primitives on each target volume, using no-replace for new objects and checked replacement/retirement for verified existing objects. Cross-volume work remains journal-recoverable, never advertised atomic. Observe filesystem identity/digests as well as intent after a crash.

- [ ] **Step 6:** Implement reconciliation for every transition including rename-completed/journal-not-flushed, dual temporary names, disconnected volumes, stale sidecars, corrupt control records, and failure during rollback. Return finish/rollback/manual-recovery actions only when filesystem evidence proves them safe; do not boot mixed generations.

- [ ] **Step 7:** Kill a child process at each durable boundary with real files/SQLite fixtures; restart a fresh journal reader and assert actual old/new bytes, fence state, and retained artifacts. Keep fixed admission fenced until installed validation and activation state are both durable.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# In journal reconciliation, evidence outranks a missing final event.
if publication_started and not installed_validation_proven:
    return "recovery_required"
if installed_validation_proven and not activation_record_proven:
    return "recovery_required"
# Never clear admission from an intent record alone.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_publication_crashes.py -q
git diff --check
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): implement durable publication journal and crash reconciliation`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Interrupted publication is classified using durable journal and actual filesystem evidence.
- No supported startup opens an ambiguous mixed generation, and original/candidate/rollback evidence is retained.
- Native crash tests cover every durable transition and refuse unqualified filesystem semantics.


<a id="task-19"></a>
## Task 19: Gate restored projections and invalidate stale retrieval state

**Backlog:** [TASK-32002](../../../backlog/tasks/task-32002%20-%20Gate-restored-projections-and-invalidate-stale-retrieval-state.md) — To Do.

**Dependencies:** TASK-31989, TASK-31990, TASK-31991, TASK-32000, TASK-32001.

**Files and ownership:**

- Create: `tldw_chatbook/RAG_Search/recovery.py`
- Modify: `tldw_chatbook/RAG_Search/simplified/vector_store.py`
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py`
- Modify: `tldw_chatbook/DB/RAG_Indexing_DB.py`
- Create: `Tests/Backup_Recovery/test_projection_recovery.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def projection_ready(*, source_digest: str, indexed_source_digest: str,
                     compatible: bool, reconciled: bool) -> bool: ...
def quarantine_projections(plan: RestorePlan, journal: Journal) -> None: ...
def validate_projections(plan: RestorePlan, candidate: Path) -> tuple[str, ...]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_projection_recovery.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_stale_index_cannot_serve_restored_sources():
    from tldw_chatbook.RAG_Search.recovery import projection_ready
    assert projection_ready(source_digest="older", indexed_source_digest="newer",
                            compatible=True, reconciled=True) is False
    assert projection_ready(source_digest="same", indexed_source_digest="same",
                            compatible=True, reconciled=False) is False
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_projection_recovery.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Register durable vector/index metadata owners and qualified capture policies with the existing census. Model/source/schema/embedding identity belongs to owner provenance; matching record IDs or filenames is insufficient.

- [ ] **Step 4:** Under ADR-030 disable affected retrieval and clear service query caches before a restored source becomes queryable. Omitted dependent indexes are retired into verified rollback or quarantined, never treated as independent preserved optional files. Shared indexes expand/refuse the previewed scope.

- [ ] **Step 5:** Validate restored/retained indexes against active authoritative source records, schema, model/embedding compatibility, and provenance. Persist readiness against the new local generation. Missing provenance or adapter leaves retrieval unavailable and surfaces explicit reconciliation/rebuild prerequisites.

- [ ] **Step 6:** No rebuild starts automatically, including local rebuild; runtime activation cannot override projection validity. Existing user-driven reconciliation produces readiness only after its owner verifies the result.

- [ ] **Step 7:** Use real source/index fixtures containing target-only, trashed/deleted, and changed-content records; test omitted index, equal IDs with different content, unsupported engine, isolated restore, later rollback, relaunch, and service cache isolation.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
def projection_ready(*, source_digest: str, indexed_source_digest: str,
                     compatible: bool, reconciled: bool) -> bool:
    return bool(source_digest) and source_digest == indexed_source_digest and compatible and reconciled
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_projection_recovery.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): gate restored projections and invalidate stale retrieval state`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- No stale or incompatible projection can serve restored source data, including after relaunch or rollback.
- Omitted/shared indexes obey explicit previewed retirement/quarantine and scope rules.
- Retrieval resumes only after qualified compatibility and reconciliation, without automatic rebuilds.


<a id="task-20"></a>
## Task 20: Persist per-generation recovery activation requirements

**Backlog:** [TASK-32003](../../../backlog/tasks/task-32003%20-%20Persist-per-generation-recovery-activation-requirements.md) — To Do.

**Dependencies:** TASK-31988, TASK-31991, TASK-31992, TASK-32001.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/activation.py`
- Modify: `tldw_chatbook/Backup_Recovery/bootstrap.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Notes/sync_service.py`
- Modify: `tldw_chatbook/Scheduling/services/scheduling_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_service.py`
- Create: `Tests/Backup_Recovery/test_activation.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
class ActivationStore:
    def __init__(self, root: Path): ...
    def require(self, generation: str, owners: tuple[str, ...]) -> None: ...
    def approve(self, generation: str, owner: str) -> None: ...
    def allowed(self, generation: str, owner: str) -> bool: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_activation.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_approving_one_owner_does_not_resume_another(tmp_path):
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    state = ActivationStore(tmp_path)
    state.require("generation-1", ("sync", "schedules"))
    state.approve("generation-1", "sync")
    reopened = ActivationStore(tmp_path)
    assert reopened.allowed("generation-1", "sync") is True
    assert reopened.allowed("generation-1", "schedules") is False
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_activation.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Persist private per-generation/per-owner requirements independently of the operation journal/report/catalog. require is idempotent without clearing an existing requirement; approve is invoked only by the corresponding existing local review owner and cannot accept imported approval metadata.

- [ ] **Step 4:** Associate required state durably before clearing publication fences. All supported launches check it before service composition, refresh timers, cleanup, scheduling, sync, MCP/tools/skills, agents, downloads, local models, network indexing, and credential probes.

- [ ] **Step 5:** Missing/corrupt/mismatched state for a known restored generation keeps execution/reconnection inactive while safe local content inspection remains available. Every restore and rollback creates fresh local generation requirements. Preserve queues/history as evidence without catch-up/replay.

- [ ] **Step 6:** Wire owner-specific review/reconciliation controls: Notes/File Notes requires fresh identities/claims and dry-run; permission stores require fresh root review; one provider approval cannot enable schedules or sync. Needs setup renders state but cannot own or clear it.

- [ ] **Step 7:** Use network and process-spawn sentinels in fresh processes to test first open, ordinary relaunch, headless launch, catalog/report rebuild, fence clearing, and corrupt activation data. Verify local content reads remain usable.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# In allowed(), using the checked local record for this generation/owner:
if record is None or not record_valid:
    return False
return record["generation"] == generation and owner in record["approved_owners"]
# require()/approve() update only the local durable owner-specific record.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_activation.py -q
python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): persist per-generation recovery activation requirements`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Restored capabilities remain inactive across every supported launch until their own owner review completes.
- Missing/corrupt activation records and imported approvals cannot grant execution authority.
- Safe local inspection works and one owner approval never activates unrelated automation or queued work.


<a id="task-21"></a>
## Task 21: Restore and reopen an isolated profile

**Backlog:** [TASK-32004](../../../backlog/tasks/task-32004%20-%20Restore-and-reopen-an-isolated-profile.md) — To Do.

**Dependencies:** TASK-32000, TASK-32001, TASK-32002, TASK-32003.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/profile_catalog.py`
- Create: `tldw_chatbook/Backup_Recovery/isolated_restore.py`
- Modify: `tldw_chatbook/cli.py`
- Create: `Tests/Backup_Recovery/test_isolated_restore.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
class ProfileCatalog:
    def __init__(self, control_root: Path): ...
    def register(self, profile_id: str, config: Path, data: Path) -> None: ...
    def resolve(self, profile_id: str) -> tuple[Path, Path]: ...
def restore_isolated(archive: SealedArchive, plan: RestorePlan,
                     control_root: Path, cancel: Event) -> str: ...
def launch_profile(profile_id: str, control_root: Path) -> int: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_isolated_restore.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_profile_catalog_reopens_explicit_paths(tmp_path):
    from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
    config, data = tmp_path / "new" / "config.toml", tmp_path / "new" / "data"
    data.mkdir(parents=True)
    config.write_text("[general]")
    ProfileCatalog(tmp_path / "control").register("opaque-id", config, data)
    assert ProfileCatalog(tmp_path / "control").resolve("opaque-id") == (config, data)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_isolated_restore.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Stage and validate into newly created private roots, remap every managed locator through its owner, and allocate fresh profile/device/credential identities. Never parse damaged current sources or treat ambient storage/config/credentials as default authority.

- [ ] **Step 4:** Journal publication, installed validation, durable activation requirements, and catalog registration. Catalog entries contain only opaque IDs and checked locators; rebuild from intact independent evidence. Failure leaves recoverable evidence without advertising a usable profile.

- [ ] **Step 5:** launch_profile verifies catalog/admission/activation mapping and starts a fresh process with explicit config/data selectors and filtered inherited authority. Do not hot-swap app globals. Recheck no writable aliases to original data or shared scopes.

- [ ] **Step 6:** Test archive-only restore with corrupt current config/DB, unencrypted locator remapping, multi-profile archives, shared artifacts, missing engines, partial valid groups, source hashes unchanged, and reopening after process restart. Compare content and recovered-media resolution through actual app read services.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# After installed validation in restore_isolated():
activation.require(generation, affected_owners)
journal.record("activation_recorded", {"generation": generation})
catalog.register(profile_id, new_config, new_data)
# Only then durably commit and clear the pending-operation fence.
```

- [ ] **Step 7:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_isolated_restore.py -q
git diff --check
```

- [ ] **Step 8:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 9:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): restore and reopen an isolated profile`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Isolated recovery creates and reopens a separate profile without altering original local data.
- Damaged current configuration and databases do not prevent archive-only recovery.
- Fresh launch respects relocated paths, credential/device isolation, durable activation, and projection readiness.


<a id="task-22"></a>
## Task 22: Replace selected local data with verified encrypted rollback

**Backlog:** [TASK-32005](../../../backlog/tasks/task-32005%20-%20Replace-selected-local-data-with-verified-encrypted-rollback.md) — To Do.

**Dependencies:** TASK-31993, TASK-31997, TASK-31999, TASK-32000, TASK-32001, TASK-32002, TASK-32003, TASK-32004.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/replacement.py`
- Modify: `tldw_chatbook/Backup_Recovery/journal.py`
- Create: `Tests/Backup_Recovery/test_replacement.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def replace(plan: RestorePlan, candidate: Path, *, control_root: Path,
            rollback_password: bytes, cancel: Event) -> str: ...
def require_rollback_password(password: bytes) -> None: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_replacement.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_replace_cannot_start_without_rollback_password():
    import pytest
    from tldw_chatbook.Backup_Recovery.replacement import require_rollback_password
    with pytest.raises(ValueError, match="rollback_password_required"):
        require_rollback_password(b"")
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_replacement.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Revalidate archive digest, target inventory/fingerprints, scope, native qualification, space, participant coverage, and credential omissions after maintenance begins. A changed target invalidates the reviewed plan before live mutation.

- [ ] **Step 4:** Capture every affected original restore/retire object plus supported credential values into an exact encrypted rollback archive; include raw corrupt config bytes without parsing. No portable redaction applies. Unreadable required original databases refuse replacement and retain the isolated recovery path.

- [ ] **Step 5:** Keep admission closed through rollback encryption/verification, journaled publication, installed validation, durable activation state, and commit. Do not release ordinary writers after safety capture. Publish credentials into nonconflicting scopes with journaled remapping, never overwrite shared keyring values. Journal the locally planned credential-scope intents before applying values, then record the applied mappings before commit.

- [ ] **Step 6:** Apply previewed restore/retire/preserve sets via owners; remove retired objects from active namespaces only after verified rollback. Validate final live inventory, SQLite sidecars, assets, tombstones, projections, and directory metadata. If validation fails, remain fenced and offer evidence-backed recovery.

- [ ] **Step 7:** Before publication cancellation is immediate; afterward honor Finish recovery/Roll back at a journal-safe boundary. Test writes arriving during rollback encryption, crashes at every artifact, ENOSPC, lost volume, changed keyring, invalid target mappings, malformed old config, and credential-unavailable acknowledgement.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
def require_rollback_password(password: bytes) -> None:
    if not password:
        raise ValueError("rollback_password_required")
# replace() calls this before maintenance and any live mutation; archive
# verification is still mandatory even when a password was supplied.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_replacement.py -q
git diff --check
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): replace selected local data with verified encrypted rollback`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
- Maintenance spans the entire safety-copy/publication/validation interval and final active inventory matches the approved generation.
- Interrupted or failed replacement retains recovery evidence and never boots ambiguous or automatically active state.


<a id="task-23"></a>
## Task 23: Expose retained recovery copies and safe later rollback

**Backlog:** [TASK-32006](../../../backlog/tasks/task-32006%20-%20Expose-retained-recovery-copies-and-safe-later-rollback.md) — To Do.

**Dependencies:** TASK-32001, TASK-32004, TASK-32005.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/recovery_copies.py`
- Create: `tldw_chatbook/Backup_Recovery/recovery_service.py`
- Create: `Tests/Backup_Recovery/test_recovery_copies.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def deletion_allowed(*, pending_operation: bool, active_hold: bool,
                     user_selected: bool) -> bool: ...
def rollback(operation_id: str, *, control_root: Path,
             old_password: bytes, new_password: bytes, cancel: Event) -> str: ...
class RecoveryService:
    def __init__(self, control_root: Path): ...
    def status(self, operation_id: str) -> Mapping[str, object]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_recovery_copies.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_pending_recovery_copy_cannot_be_deleted():
    from tldw_chatbook.Backup_Recovery.recovery_copies import deletion_allowed
    assert deletion_allowed(pending_operation=True, active_hold=False, user_selected=True) is False
    assert deletion_allowed(pending_operation=False, active_hold=True, user_selected=True) is False
    assert deletion_allowed(pending_operation=False, active_hold=False, user_selected=False) is False
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_recovery_copies.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** List retained rollback artifacts with checked identities, coverage, sizes, operation associations, validation status, and pending holds. Passwords are not persisted. A missing password leaves encrypted recovery available for later unlocking, never silently deletes the artifact.

- [ ] **Step 4:** Later rollback validates the selected original recovery archive and previews replacement of post-restore changes. Capture those changes into a new verified encrypted safety copy before proceeding through the same replacement executor and new activation generation.

- [ ] **Step 5:** Explicit user deletion rechecks holds and unresolved journals under coordination; forbid age-based retention sweeps and unknown-file cleanup. Operation-owned unpublished staging alone may be cleaned automatically with verified identity.

- [ ] **Step 6:** Compose one RecoveryService facade over existing capture/reader/planner/executors at the app lifecycle boundary. Store immutable status/progress snapshots and sanitized issues; view navigation does not own/cancel the worker. Shutdown cancels safe preparation or leaves a durably recoverable handoff.

- [ ] **Step 7:** Test post-restore edits preserved before rollback, nested rollback failure, missing password, missing catalog, held copies, explicit orphan candidates, and app-close transitions. Verify old and new stored bytes rather than status strings alone.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
def deletion_allowed(*, pending_operation: bool, active_hold: bool,
                     user_selected: bool) -> bool:
    return user_selected and not pending_operation and not active_hold
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_recovery_copies.py -q
git diff --check
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): expose retained recovery copies and safe later rollback`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Users can inspect retained recovery copies and roll back later only after intervening changes are preserved.
- Pending evidence and held artifacts cannot be deleted or swept automatically.
- App-owned operation state survives navigation and closes without abandoning unsafe publication.
