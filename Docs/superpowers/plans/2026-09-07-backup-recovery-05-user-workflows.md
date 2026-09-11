# User-facing backup and recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Usable F9, first-run, and damaged-installation recovery flows over the same services.

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

- Task 24 owns `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `tldw_chatbook/UI/Screens/backup_restore_state.py`.
- Task 25 owns `tldw_chatbook/Backup_Recovery/launcher.py`, `tldw_chatbook/Backup_Recovery/__main__.py`.

<a id="task-24"></a>
## Task 24: Expose backup and restore in canonical F9 Settings

**Backlog:** [TASK-32007](../../../backlog/tasks/task-32007%20-%20Expose-backup-and-restore-in-canonical-F9-Settings.md) — To Do.

**Dependencies:** TASK-31999, TASK-32004, TASK-32005, TASK-32006.

**Files and ownership:**

- Create: `tldw_chatbook/UI/Screens/backup_restore_screen.py`
- Create: `tldw_chatbook/UI/Screens/backup_restore_state.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/UI/test_backup_restore_screen.py`
- Create: `Tests/ProductionApp/test_backup_restore_composition.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
# Screen imports services only through its app-owned RecoveryService reference.
class BackupRestoreScreen(Screen): ...
def result_label(*, archive_verified: bool, restoration_validated: bool,
                 opened: bool, needs_setup: bool) -> str: ...
```

- [ ] **Step 1:** Add this first regression to `Tests/UI/test_backup_restore_screen.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_archive_verification_does_not_claim_profile_opened():
    from tldw_chatbook.UI.Screens.backup_restore_state import result_label
    label = result_label(archive_verified=True, restoration_validated=False,
                         opened=False, needs_setup=False)
    assert label == "Archive verified"
```

- [ ] **Step 2:** Run `python -m pytest Tests/UI/test_backup_restore_screen.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Add Backup & Restore entry to canonical F9 Settings and command palette with Create backup, Inspect/restore, Recovery copies, and Restored profiles. Do not modify deprecated settings surfaces or path Save semantics. Use existing widgets, tokens, pickers, footer/keybinding conventions.

- [ ] **Step 4:** Implement progressive forms for exact profile coverage, opt-in external/models/temporary/diagnostic choices, encryption/password confirmation, unavailable credential disclosure, output path and per-volume space/maintenance preview. Partial acknowledgement and credential-required encryption are enforced by service validation as well as UI.

- [ ] **Step 5:** Restore forms inspect first, choose mode, show immutable mappings/shared effects/retire/preserve sets/unsupported metadata/activation prerequisites, and collect rollback password for replacement. Recovery copies show explicit delete and later-rollback consequences; profile entries reopen in a new process.

- [ ] **Step 6:** Use exclusive background workers for operations over 100 ms, app-owned progress/cancellation, and view-generation guards. Escape/back navigation never cancels publication; cancellation changes to safe finish/rollback choices after the boundary. Never render archive text as markup.

- [ ] **Step 7:** Mount the real production hierarchy and CSS, drive actual buttons through real services with isolated fixtures, and assert visible archive/result paths, coverage, progress, errors, rollback prompt, tombstones, and Needs setup. Test narrow terminals and implemented keys; take a live TUI verification artifact before completion.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
def result_label(*, archive_verified: bool, restoration_validated: bool,
                 opened: bool, needs_setup: bool) -> str:
    if needs_setup:
        return "Needs setup"
    if opened:
        return "Opened successfully"
    if restoration_validated:
        return "Restoration validated"
    return "Archive verified" if archive_verified else "Not verified"
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/UI/test_backup_restore_screen.py Tests/ProductionApp/test_backup_restore_composition.py -q
python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): expose backup and restore in canonical f9 settings`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- Users can discover and execute create/inspect/both restore modes from F9 with truthful coverage, risks, and progress.
- Recovery copies and isolated profiles are inspectable and actionable through the canonical UI.
- Product-level mounted/live evidence verifies actual services and keyboard/navigation behavior without unintended execution.


<a id="task-25"></a>
## Task 25: Expose startup-independent recovery and first-run restore

**Backlog:** [TASK-32008](../../../backlog/tasks/task-32008%20-%20Expose-startup-independent-recovery-and-first-run-restore.md) — To Do.

**Dependencies:** TASK-31988, TASK-31995, TASK-32004, TASK-32005, TASK-32006, TASK-32007.

**Files and ownership:**

- Create: `tldw_chatbook/Backup_Recovery/launcher.py`
- Create: `tldw_chatbook/Backup_Recovery/__main__.py`
- Modify: `tldw_chatbook/cli.py`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Modify: `tldw_chatbook/UI/Wizards/first_run_recovery_dialog.py`
- Create: `Tests/Backup_Recovery/test_launcher.py`
- Create: `Tests/Wizards/test_first_run_backup_restore.py`

**Interfaces:** Consumes the named types/operations from the dependencies and produces
the following exact contracts. Factories return only installed owner declarations.

```python
def recovery_main(argv: Sequence[str] | None = None) -> int: ...
# Commands: inspect ARCHIVE; restore ARCHIVE --isolated;
# restore ARCHIVE --replace; recover OPERATION; profiles; copies.
# Optional local --control-root PATH never bypasses fixed admission checks.
```

- [ ] **Step 1:** Add this first regression to `Tests/Backup_Recovery/test_launcher.py`, then add the concrete
  fixtures/scenarios named in the implementation steps as their behavior is built.

```python
def test_recovery_help_does_not_import_normal_app(monkeypatch):
    import builtins
    import pytest
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in {"tldw_chatbook.app", "tldw_chatbook.config"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    from tldw_chatbook.Backup_Recovery.launcher import recovery_main
    with pytest.raises(SystemExit) as result:
        recovery_main(["--help"])
    assert result.value.code == 0
```

- [ ] **Step 2:** Run `python -m pytest Tests/Backup_Recovery/test_launcher.py -q`. Confirm the specified behavior
  fails; after adding importable structure, confirm a behavioral red assertion before
  proceeding. Do not count a missing optional dependency as the intended failure.

- [ ] **Step 3:** Provide python -m tldw_chatbook.Backup_Recovery and early CLI recovery dispatch using only the minimal recovery package until a profile is explicitly opened. Parse local paths/options safely; collect passwords interactively via a non-echoing prompt or owned UI, never argv/env.

- [ ] **Step 4:** Recovery inspection/isolated restore works with missing/malformed/encrypted-unavailable current configuration. Replacement still verifies current targets and rollback capture; inaccessible journal roots preserve the fixed fence and show explicit recovery actions.

- [ ] **Step 5:** First-run Restore a backup invokes the same app-owned services and dedicated recovery view. A failed bootstrap routes to minimal recovery before normal database constructors, migrations, catalog refresh, TTL cleanup, or defaults creation.

- [ ] **Step 6:** Expose recover operation, inspect copies, reopen profiles, bounded inert extraction for unsupported groups, and clear separate verified/validated/opened/needs-setup outcomes. Inert extraction retains archive path controls and never launches an imported DB schema.

- [ ] **Step 7:** Use clean subprocesses with import, network, subprocess, and filesystem sentinels to test every launch route, malformed config, inaccessible targets, wrong passwords, interrupted journals, and successful isolated first open. Native terminal QA uses only disposable synthetic profiles.

**Implementation invariant:** preserve this control flow while implementing the steps.

```python
# In Backup_Recovery/__main__.py:
from tldw_chatbook.Backup_Recovery.launcher import recovery_main

if __name__ == "__main__":
    raise SystemExit(recovery_main())
```

- [ ] **Step 8:** Run focused tests and applicable guards. Expected: named behavior
  and adversarial cases pass; no skips substituted for required release evidence.

```bash
python -m pytest Tests/Backup_Recovery/test_launcher.py Tests/Wizards/test_first_run_backup_restore.py -q
python -m pytest Tests/ProductionApp/test_service_composition_lifecycle.py -q
```

- [ ] **Step 9:** Run scoped lint/format checks from Execution discipline, review
  the complete diff and actual filesystem/process evidence, and update owner/user docs
  and this task's Implementation Notes with ADR-126 and exact results.

- [ ] **Step 10:** When all criteria below are demonstrated, check them in Backlog,
  mark the task Done using the verified CLI/file workflow, and commit only task-owned
  files with subject `feat(backup): expose startup-independent recovery and first-run restore`. Recheck task-ID collisions
  before merge and preserve unrelated work.

**Acceptance evidence:**

- First-run and damaged-installation users can inspect and restore without normal startup succeeding.
- Minimal recovery reuses qualified services and never bypasses admission, target verification, or activation gates.
- CLI and UI credentials stay out of process arguments, environment, logs, and persisted requests.
