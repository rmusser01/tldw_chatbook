# Coherent capture and qualified archives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verified local recovery archives and bounded inspection with explicit coverage and credentials.

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

- Task 11 owns `tldw_chatbook/Backup_Recovery/recovered_media.py`, `tldw_chatbook/Backup_Recovery/recovered_media_schema.py`.
- Task 12 owns `tldw_chatbook/Backup_Recovery/archive_models.py`, `tldw_chatbook/Backup_Recovery/archive_reader.py`, `tldw_chatbook/Backup_Recovery/limits.py`.
- Task 13 owns `tldw_chatbook/Backup_Recovery/sqlite_validation.py`.
- Task 14 owns `tldw_chatbook/Backup_Recovery/credentials.py`, `tldw_chatbook/Backup_Recovery/credential_policies.py`.
- Task 15 owns `tldw_chatbook/Backup_Recovery/capture.py`, `tldw_chatbook/Backup_Recovery/space.py`.
- Task 16 owns `tldw_chatbook/Backup_Recovery/archive_writer.py`.

<a id="task-11"></a>
## Task 11: Persist recovered media references and deletion tombstones

**Backlog:** [TASK-31994](../../../backlog/tasks/task-31994%20-%20Persist-recovered-media-references-and-deletion-tombstones.md) — To Do.

**Dependencies:** TASK-31986, TASK-31987, TASK-31992, TASK-31993.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/recovered_media.py`
- Create: `tldw_chatbook/Backup_Recovery/recovered_media_schema.py`
- Modify: `tldw_chatbook/Video_Generation/video_store.py`
- Modify: `tldw_chatbook/Widgets/chat_message_enhanced.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/backup-recovery-owner-inventory.md`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_recovered_media.py`
- Modify: `tldw_chatbook/Backup_Recovery/participants.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
class RecoveredMedia:
    def __init__(self, root: Path): ...
    def retain(self, source: Path, *, profile: str, message: str,
               slug: str, media_type: str) -> str: ...
    def resolve(self, asset_id: str) -> tuple[str, Path | None]: ...
    def delete(self, asset_id: str) -> None: ...
    def recovery_adapter(self) -> OwnerAdapter: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_recovered_media.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_deleted_asset_is_not_an_unexpected_missing_file(tmp_path):
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia
    source = tmp_path / "synthetic.webm"
    source.write_bytes(b"not decoded during registration")
    store = RecoveredMedia(tmp_path / "recovered")
    asset_id = store.retain(source, profile="p", message="m", slug="clip", media_type="video/webm")
    store.delete(asset_id)
    assert store.resolve(asset_id) == ("deleted", None)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_recovered_media.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Create a separate profile-scoped private catalog with schema version 1, asset digest/type/size, source profile/message/media-key identity, references, deletion tombstones, and operation recovery state. New schemas use numbered installed migrations; no main-store version bump unless its schema changes.

- [ ] **Step 4:** Publish verified regular payload bytes before committing references; journal file/catalog boundaries through the same private durable-write primitives. Do not decode media while registering or validating imported records.

- [ ] **Step 5:** Resolve recovered references before temporary-store lookup; known missing or deleted references never fall through by filename. Wire the actual image/video transcript consumers discovered from video_store callers and record them in the file inventory.

- [ ] **Step 6:** Delete writes a validated owner tombstone and retires payload bytes without erasing references still needed for Deleted rendering. Shared references/recovery holds prevent premature removal; explicit orphan cleanup rechecks them. Startup/TTL cleanup never removes recovered payloads.

- [ ] **Step 7:** Register this owner as durable baseline for re-backup independent of temporary-media opt-in. Test ID collisions, repeated slug across profiles, interrupted retain/delete, unexpected loss/digest corruption, shared references, tombstone round trips, and startup cleanup.

- [ ] **Step 8:** Register recovered-media file/catalog mutation admission and draining with participants.py; creation after the earlier participant census must not leave this new owner unfenced.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# resolve() consults the persisted row before any file/temporary lookup.
if row["deletion_state"] == "intentional":
    return "deleted", None
# Otherwise check the registered private payload's size/digest. A failed
# check returns ("missing", None), never another file with the same slug.
```

- [ ] **Step 9:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_recovered_media.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
```

- [ ] **Step 10:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 11:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): persist recovered media references and deletion tombstones`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Included temporary media becomes durable and resolves correctly across restart and subsequent backups.
- Intentional deletion survives as a tombstone without partial-backup status; unexpected missing required bytes still block completeness.
- Shared references, recovery holds, explicit cleanup, and interrupted publication/deletion preserve recoverable state.


<a id="task-12"></a>
## Task 12: Implement bounded immutable archive inspection

**Backlog:** [TASK-31995](../../../backlog/tasks/task-31995%20-%20Implement-bounded-immutable-archive-inspection.md) — To Do.

**Dependencies:** TASK-31984, TASK-31985, TASK-31986, TASK-31987.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/archive_models.py`
- Create: `tldw_chatbook/Backup_Recovery/archive_reader.py`
- Create: `tldw_chatbook/Backup_Recovery/limits.py`
- Create: `Tests/Backup_Recovery/test_archive_reader.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
@dataclass(frozen=True)
class ArchiveLimits:
    input_bytes: int = 2 * 1024**4
    decrypted_bytes: int = 2 * 1024**4
    expanded_bytes: int = 1024**4
    member_bytes: int = 256 * 1024**3
    members: int = 100_000
    manifest_bytes: int = 16 * 1024**2
    path_bytes: int = 1024
@dataclass(frozen=True)
class SealedArchive:
    path: Path
    digest: str
    manifest_bytes: bytes
def acquire(source: Path, work_root: Path, limits: ArchiveLimits,
            password: bytes | None, cancel: Event) -> SealedArchive: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_archive_reader.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_input_limit_applies_before_archive_parsing(tmp_path):
    import pytest
    from threading import Event
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    source = tmp_path / "oversized"
    source.write_bytes(b"x" * 1025)
    with pytest.raises(ValueError, match="input_limit"):
        acquire(source, tmp_path / "work", ArchiveLimits(input_bytes=1024), None, Event())
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_archive_reader.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Copy input to a private newly allocated staging artifact with actual streaming byte quotas and source-identity/stability checks. Close the writer and hash completed bytes; all later inspection/execution uses this artifact, never the original pathname. Encrypted inputs authenticate fully before ZIP parsing.

- [ ] **Step 4:** Define strict version-1 manifest models with Pydantic extra=forbid; freeze the parsed representation and persist canonical manifest bytes in SealedArchive. Enumerate file/directory logical IDs, parent relations, owner/schema capabilities, payload digests/sizes, exclusions, credential policy, and sanitized report. Original managed locators are allowed only in encrypted relocation metadata.

- [ ] **Step 5:** Allow only ZIP64 stored/deflated regular-file members. Validate central/local header agreement, ranges/overlap, duplicates, unsupported flags/encryption, path/case/Unicode/file-directory collisions, member counts including directory records, unreferenced payloads, expansion and streamed limits. Reject links/devices and never call extractall.

- [ ] **Step 6:** Require per-volume admission for input/decrypted/extracted bytes before and during work; hard limits are independent of compression ratios. Unusual compression requests expanded-byte review; local budget changes invalidate preflight.

- [ ] **Step 7:** Test mutation of source after preview, mutation of sealed staging, ZIP bombs, oversized headers/manifest, traversal/aliases, overlapping ZIP structures, truncation, partial groups, newer format, malicious terminal markup, and cancellation. Every failure leaves no completed output and no live-target write.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Apply to each streamed stage independently, before writing the chunk.
total += len(chunk)
if total > byte_limit:
    raise ValueError("input_limit")
destination.write(chunk)
# Use distinct counters/error codes for decrypted and expanded bytes.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_archive_reader.py -q
git diff --check
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): implement bounded immutable archive inspection`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Inspection uses completed immutable input bound to a digest and validates manifest/payload integrity under bounded resource use.
- Malformed, ambiguous, hostile, or unsupported archives cannot select destination paths or execute content.
- Limits apply during source copy and decryption before untrusted manifest information is available.


<a id="task-13"></a>
## Task 13: Validate and migrate imported SQLite under restricted owner policies

**Backlog:** [TASK-31996](../../../backlog/tasks/task-31996%20-%20Validate-and-migrate-imported-SQLite-under-restricted-owner-policies.md) — To Do.

**Dependencies:** TASK-31989, TASK-31990, TASK-31991, TASK-31994, TASK-31995.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/sqlite_validation.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Create: `Tests/Backup_Recovery/test_sqlite_validation.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def validate_candidate(owner: OwnerAdapter, candidate: Path,
                       cancel: Event, *, migrate: bool) -> tuple[str, ...]: ...
# private_sqlite.py owns the registered connection entry point:
def open_recovery_validation(owner_id: str, candidate: Path,
                             *, writable: bool) -> ContextManager[sqlite3.Connection]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_sqlite_validation.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_unknown_trigger_is_rejected_before_migration(tmp_path):
    import sqlite3
    from threading import Event
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    candidate = tmp_path / "hostile.db"
    with sqlite3.connect(candidate) as db:
        db.executescript("CREATE TABLE payload(x); CREATE TRIGGER surprise AFTER INSERT ON payload BEGIN DELETE FROM payload; END;")
    owner = core_adapters()[0]
    issues = validate_candidate(owner, candidate, Event(), migrate=True)
    assert "unsupported_schema" in issues
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_sqlite_validation.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Extend the registered private SQLite seam for read-only untrusted inspection and narrowly authorized staged writes; register explicit recovery owners and inventory entries. Set trusted_schema=OFF, disable extensions, omit side-effecting functions/modules, deny ATTACH/unauthorized operations with an authorizer, and install cancellation/progress/SQL/memory limits.

- [ ] **Step 4:** Compare sqlite_schema definitions, columns, indexes, triggers, views, virtual tables, and versions to each installed SchemaPolicy before any migration. Preserve legitimate FTS and recognized triggers through tested owner policies. Version equality alone never admits execution.

- [ ] **Step 5:** Run only supported installed migration steps against disposable staged candidates under the same restricted policy, then repeat schema/integrity/foreign-key/domain/asset validation. Do not instantiate ordinary repositories as a validation shortcut or retry through unrestricted connections.

- [ ] **Step 6:** Test a valid owner fixture altered with a malicious trigger/view/UDF/vtable as well as the small rejection case below; place a synthetic side-effect sentinel outside staging and assert it remains unchanged. Cover SQL budget interruption, absent security primitives, legitimate FTS round trips, newer schemas, supported older migrations, and failed migration rollback.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Registered private SQLite validation seam, before imported queries:
connection.enable_load_extension(False)
connection.execute("PRAGMA trusted_schema=OFF")
if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
    raise ValueError("sqlite_security_unavailable")
# Install the allowlisted authorizer and cancellation/SQL budgets next.
```

- [ ] **Step 7:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_sqlite_validation.py Tests/DB/test_private_sqlite_inventory.py -q
python -m pytest Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_private_sqlite_interop_owners.py -q
```

- [ ] **Step 8:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 9:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): validate and migrate imported sqlite under restricted owner policies`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Imported schema is qualified before migration and cannot activate unexpected SQL or application side effects.
- Valid supported SQLite/FTS stores migrate and validate under restricted connections.
- Unsupported capabilities, schemas, and resource failures remain explicit without unrestricted fallbacks.


<a id="task-14"></a>
## Task 14: Implement staged credential exclusion and encrypted credential recovery

**Backlog:** [TASK-31997](../../../backlog/tasks/task-31997%20-%20Implement-staged-credential-exclusion-and-encrypted-credential-recovery.md) — To Do.

**Dependencies:** TASK-31989, TASK-31990, TASK-31991, TASK-31992, TASK-31996.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/credentials.py`
- Create: `tldw_chatbook/Backup_Recovery/credential_policies.py`
- Modify: `tldw_chatbook/runtime_policy/server_credentials.py`
- Modify: `tldw_chatbook/Utils/config_encryption.py`
- Modify: `tldw_chatbook/Backup_Recovery/config_adapter.py`
- Create: `Tests/Backup_Recovery/test_credentials.py`
- Modify: `tldw_chatbook/Utils/sensitive_config_keys.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def sanitize_config(config: Mapping[str, object]) -> dict[str, object]: ...
def process_credentials(staging: Path, inventory: Inventory, *,
                        mode: Literal["exclude", "include", "rollback"],
                        encrypted: bool) -> tuple[str, ...]: ...
def restore_credential_values(staging: Path, scope_map: Mapping[str, str]) -> tuple[str, ...]: ...
def plan_credential_scopes(staging: Path) -> Mapping[str, str]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_credentials.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_config_secret_is_removed_without_mutating_source():
    from tldw_chatbook.Backup_Recovery.credentials import sanitize_config
    original = {"API": {"openai_api_key": "synthetic-secret-sentinel"}}
    sanitized = sanitize_config(original)
    assert "synthetic-secret-sentinel" not in repr(sanitized)
    assert original["API"]["openai_api_key"] == "synthetic-secret-sentinel"
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_credentials.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Build a typed owner policy inventory of actual known config/API/provider/server/connection secret fields, encrypted blobs, config histories, owned keyring scopes, and secret-bearing database columns. The sample key below must be verified against the current config adapter; include aliases accepted by config loading. Reuse Utils/sensitive_config_keys.py for recognized config-key aliases, with typed owner policies for semantic fields and scope; do not introduce another substring-only secret classifier.

- [ ] **Step 4:** Exclude values/blobs/live references in staged copies by default; retain only setup hints and environment variable names. Unknown credential-bearing formats block credential-excluded completeness until explicitly omitted or supported. Do not claim arbitrary prose, diagnostics, or external folders are sanitized.

- [ ] **Step 5:** For secret-bearing SQLite use qualified logical reconstruction/compaction preserving IDs and relationships, then scan entire resulting bytes for synthetic secret sentinels including formerly freed pages. Never append original sidecars or unprocessed histories afterward.

- [ ] **Step 6:** Include or rollback modes require encryption. Capture only readable supported Chatbook-owned values, never enumerate a whole keychain or export memory-only Sync keys. Require explicit coverage acknowledgement for unreadable/unexportable values; raw ciphertext is not usable credential coverage.

- [ ] **Step 7:** plan_credential_scopes reads only supported affected scopes and returns a checked old-to-new scope mapping without writes. Reuse a scope only while its value still matches; otherwise allocate a new nonconflicting scope. The executor journals that mapping before calling restore_credential_values and records its result afterward. Recheck scope identity/value at apply time; drift invalidates the plan. Unsupported remapping retains explicit encrypted recovery material. Never overwrite shared keyring entries or test remote authentication.

- [ ] **Step 8:** Test changed/deleted keyring entries after backup, collision across profiles, missing backend, revoked/expired fixtures, history aliases, encrypted config unlock failure, freed pages, and absence of sentinels from paths/logs/control reports.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# At process_credentials entry, before reading owned secret values:
if mode in {"include", "rollback"} and not encrypted:
    raise ValueError("credentials_require_encryption")
# Work on staged data only; exclusions rebuild secret-bearing SQLite bytes.
```

- [ ] **Step 9:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_credentials.py -q
git diff --check
```

- [ ] **Step 10:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 11:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): implement staged credential exclusion and encrypted credential recovery`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Default archives remove managed secrets from all supported staged locations, including SQLite remnants, without modifying sources.
- Credential inclusion and exact rollback require encryption and retain supported values with honest omission reporting.
- Restoration isolates credential scopes and never overwrites shared entries or claims remote authentication was recovered.


<a id="task-15"></a>
## Task 15: Capture a coherent final inventory with optional external content

**Backlog:** [TASK-31998](../../../backlog/tasks/task-31998%20-%20Capture-a-coherent-final-inventory-with-optional-external-content.md) — To Do.

**Dependencies:** TASK-31993, TASK-31994, TASK-31995, TASK-31996, TASK-31997.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/capture.py`
- Create: `tldw_chatbook/Backup_Recovery/space.py`
- Modify: `tldw_chatbook/Backup_Recovery/inventory.py`
- Modify: `tldw_chatbook/Model_Artifacts/recovery.py`
- Create: `Tests/Backup_Recovery/test_capture.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
@dataclass(frozen=True)
class CaptureResult:
    root: Path
    inventory: Inventory
    manifest_bytes: bytes
def capture(config_paths: tuple[Path, ...], approved_scope: str,
            destination: Path, *, options: Mapping[str, object],
            cancel: Event) -> CaptureResult: ...
def compare_scope(approved: Inventory, current: Inventory) -> tuple[str, ...]: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_capture.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_changed_source_mapping_invalidates_preview(tmp_path):
    from tldw_chatbook.Backup_Recovery.models import StorageItem, Inventory
    from tldw_chatbook.Backup_Recovery.capture import compare_scope
    old = Inventory((StorageItem("db", "main", tmp_path / "old", "included", ()),), True, "old-scope", ())
    new = Inventory((StorageItem("db", "main", tmp_path / "new", "included", ()),), True, "new-scope", ())
    assert "scope_changed" in compare_scope(old, new)
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_capture.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Implement the create flow over declared owners: preflight options/output/capacity, obtain participant coverage and maintenance, rediscover effective mappings/aliases/required assets under closed admission, and compare against approved scope. Scope or budget changes release locks and require a renewed preview; never expand a held lock set out of order.

- [ ] **Step 4:** Capture normal record/asset growth within approved owner scope under rechecked quotas. Snapshot databases and referenced assets as dependency groups, process credential policy in staging, validate captures, and reconcile manifest/completeness with the final fenced inventory before writers resume.

- [ ] **Step 5:** Use per-file before/after stability checks for explicitly selected external roots, with bounded retry and a per-file—not folder-wide—consistency label. Models require qualified dependency resolution; selected temporary media is captured with its message identity; diagnostics include arbitrary-content disclosure.

- [ ] **Step 6:** Calculate independent volume requirements for capture, encrypted/plain output, decrypt/staging, journals, and retained rollback. Runtime ENOSPC/quota/cancellation stops safely. Ordinary backup releases admission before packaging; return immutable CaptureResult with actual coverage/counts.

- [ ] **Step 7:** Test mutations between preview and capture, unknown owner/root appearance, aliases, disappeared assets, transaction/asset concurrency, optional omissions, tombstones, unstable external files, and insufficient capacity. Capture must never label an incomplete dependency group complete.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# Under the maintenance lease in capture():
current = discover(config_paths)
if current.scope_digest != approved_scope:
    raise ValueError("scope_changed")
# Raising unwinds the lease; renewed review happens after release.
# Asset/record growth is excluded from the scope hash and rechecks capacity.
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_capture.py -q
git diff --check
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): capture a coherent final inventory with optional external content`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- A completed capture reflects the final inventory under maintenance, including valid in-scope growth and coherent DB/asset dependencies.
- Changed scope or budget renews preview safely; partial and optional coverage is accurately reported.
- Ordinary writers resume after verified capture, before encryption or output transfer.


<a id="task-16"></a>
## Task 16: Write verify and publish recovery archives without overwrite

**Backlog:** [TASK-31999](../../../backlog/tasks/task-31999%20-%20Write-verify-and-publish-recovery-archives-without-overwrite.md) — To Do.

**Dependencies:** TASK-31985, TASK-31995, TASK-31998.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/archive_writer.py`
- Create: `Tests/Backup_Recovery/test_archive_writer.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def write_archive(capture: CaptureResult, destination: Path, *,
                  password: bytes | None, cancel: Event) -> SealedArchive: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_archive_writer.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_existing_backup_is_preserved_by_writer(tmp_path):
    import pytest
    from threading import Event
    from tldw_chatbook.Backup_Recovery.models import Inventory
    from tldw_chatbook.Backup_Recovery.capture import CaptureResult
    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
    output = tmp_path / "good.tldw-backup.zip"
    output.write_bytes(b"keep")
    capture = CaptureResult(tmp_path / "capture", Inventory((), True, "scope", ()), b"{}")
    with pytest.raises(FileExistsError):
        write_archive(capture, output, password=None, cancel=Event())
    assert output.read_bytes() == b"keep"
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_archive_writer.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Generate versioned manifest/report and regular ZIP64 members from sealed capture; store already-compressed media/model bytes and deflate eligible text. Apply the same reader limits, explicit directory topology, metadata policy, dependency checks, and secret/locator restrictions.

- [ ] **Step 4:** Validate output/source/control aliases before reading capture and again at publication; reject output within selected sources unless an explicit excluded output root was previewed. Allocate private temporary output on the destination volume, encrypt if requested/required, and run the actual reader verification before publication.

- [ ] **Step 5:** Publish with native_files.publish_new only after verification and flush; handle the race where another process creates the destination. Return the published artifact digest and verified manifest. Suffixes distinguish .tldw-backup.zip and .tldw-backup.zip.age without relying on suffix for input trust.

- [ ] **Step 6:** Exercise complete and explicit partial round trips, directory metadata, existing-file races, aliasing, output-in-source, interruption immediately before/after publication, corrupted output, wrong passwords, and runtime ENOSPC. Progress distinguishes capturing/packaging/encrypting/verifying/publishing; completion identifies the real file.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# After writing, flushing and actual reader verification:
publish_new(verified_output.path, destination)
# No exists/replace fallback. Record the completed published digest only
# after directory durability is established by the qualified primitive.
```

- [ ] **Step 7:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_archive_writer.py -q
git diff --check
```

- [ ] **Step 8:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 9:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): write verify and publish recovery archives without overwrite`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Archives created by the writer pass the same bounded reader and carry truthful coverage, directory, and credential metadata.
- Existing backups/source/control files remain unchanged under races and aliases.
- Failures and cancellation never expose incomplete output as a verified archive.
