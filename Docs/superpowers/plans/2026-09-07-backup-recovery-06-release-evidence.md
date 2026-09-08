# Recovery release qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrated complete/replace capabilities with native and product evidence.

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

- Task 26 owns `backlog/docs/backup-recovery-release-evidence.md`, `Docs/Backup-and-Recovery.md`, `.github/workflows/backup-recovery-qualification.yml`.

<a id="task-26"></a>
## Task 26: Qualify complete backup and replacement release capabilities

**Backlog:** [TASK-32009](../../../backlog/tasks/task-32009%20-%20Qualify-complete-backup-and-replacement-release-capabilities.md) — To Do.

**Dependencies:** TASK-31985, TASK-31993, TASK-31994, TASK-31995, TASK-31996, TASK-31997, TASK-31998, TASK-31999, TASK-32000, TASK-32001, TASK-32002, TASK-32003, TASK-32004, TASK-32005, TASK-32006, TASK-32007, TASK-32008.

**Files and ownership:**

- Create: `Tests/Backup_Recovery/test_complete_roundtrip.py`
- Create: `Tests/Backup_Recovery/test_recovery_release_gate.py`
- Create: `Tests/ProductionApp/test_backup_restore_end_to_end.py`
- Create: `backlog/docs/backup-recovery-release-evidence.md`
- Create: `Docs/Backup-and-Recovery.md`
- Modify: `Packaging/PACKAGING_CHECKLIST.md`
- Modify: `tldw_chatbook/Backup_Recovery/qualification.py`
- Create: `.github/workflows/backup-recovery-qualification.yml`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# qualification.py uses installed evidence/protocol support, not UI guesses.
def release_capability(*, helper: bool, owner_coverage: bool,
                       admission: bool, archive: bool, native_publish: bool,
                       restore: bool, product_flow: bool) -> bool: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_complete_roundtrip.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_complete_capability_requires_every_release_gate():
    from tldw_chatbook.Backup_Recovery.qualification import release_capability
    gates = dict(helper=True, owner_coverage=True, admission=True, archive=True,
                 native_publish=True, restore=True, product_flow=True)
    assert release_capability(**gates) is True
    for name in gates:
        incomplete = {**gates, name: False}
        assert release_capability(**incomplete) is False
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_complete_roundtrip.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Build a synthetic multi-profile installation using actual owner APIs with custom DBs, shared stores, durable files/empty roots, soft deletions, queued work, supported credentials, intentionally deleted recovered media, temporary media, selected models/external folders, and derived indexes. Include every supported owner-inventory row.

- [ ] **Step 4:** Exercise actual F9 create, inspect, isolated restore/reopen, replace, and later rollback preserving post-restore edits. Compare semantic data/identities/references and asset bytes; independently verify directory metadata and credential exclusion. Distinguish verified archive, validated installation, successful open, and Needs setup.

- [ ] **Step 5:** Run crash matrix on native qualified OS/filesystem combinations, including multiple volumes, pending fixed admission, lost control roots, output races, KDF limits, source/inventory drift, SQLite attacks, sidecars, index readiness, tombstone deletion interruption, and credentials changed after snapshot.

- [ ] **Step 6:** Enable complete capture and replacement independently only for demonstrated owner/protocol/platform capabilities. A generic boolean test below proves conjunction wiring, not native qualification: record command, commit, environment, actual result/artifact, and unavailable combinations in the evidence ledger.

- [ ] **Step 7:** Document recovery limitations/password loss/plaintext staging/no server recovery, upgrade interoperability, rollback retention, missing models/credentials, and explicit activation/rebuild workflow. Update packaging checklist and user help; complete live UI verification in a disposable profile. Run targeted feature and relevant inventory/lint guards only; a full suite requires a separate user request.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
def release_capability(*, helper: bool, owner_coverage: bool,
                       admission: bool, archive: bool, native_publish: bool,
                       restore: bool, product_flow: bool) -> bool:
    return all((helper, owner_coverage, admission, archive,
                native_publish, restore, product_flow))
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_complete_roundtrip.py Tests/Backup_Recovery/test_recovery_release_gate.py Tests/ProductionApp/test_backup_restore_end_to_end.py -q
python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q
python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): qualify complete backup and replacement release capabilities`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Complete and replacement capability labels are backed by end-to-end owner, archive, native, and product evidence.
- Both restore destinations and later rollback preserve expected data under ordinary and interrupted operations.
- User/release documentation states qualified platforms, exclusions, credential limits, and recovery actions without overstating guarantees.
