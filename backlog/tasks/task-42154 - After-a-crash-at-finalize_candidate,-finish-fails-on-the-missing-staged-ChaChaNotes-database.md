---
id: TASK-42154
title: "After a crash at finalize_candidate, finish fails on the missing staged ChaChaNotes database"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 22:23
updated_date: 2026-11-22 06:20
labels:
- follow-up
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery/replacement.py
- tldw_chatbook/Backup_Recovery/publication.py
- tldw_chatbook/Backup_Recovery/journal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of fu/r18-petsdef (round 18, rev3), and confirmed on the branch's code; the Backup_Recovery recovery code there matches dev.

The probe:
- A Replace on a separate config folder, with no pet files (the harness's `[pets-none]` shape without pets: groups conversations, settings and workspaces).
- The process is killed at `publication.finalize_candidate` (`os._exit`), after publication has moved the candidate's artifacts into place.
- A fresh process then runs `replacement.recover_replacement(..., action=...)`.

What happens:
- `rollback` succeeds (`rolled_back`), and `startup_permission` is `(True, 'startup_allowed')`.
- `finish` fails with `FileNotFoundError: 'tldw_chatbook_ChaChaNotes.db'`: `recover_replacement` → `_acquired_source` → `publication._descriptor` → `journal.observe_artifact`. The descriptor check expects every candidate artifact to still sit in the stage, but publication has already moved them. Startup then stays refused (`recovery_pending`) until the user picks `rollback`.

`finish` is offered as a recovery action but cannot complete after this crash point.

Recipe (the round's scratch probe will be deleted; this is enough to write AC#1's test):
1. Build the case on `Tests/Backup_Recovery/test_created_owner_rollback.py`'s `_SCENARIO`:
   - `CASES={'control': {'groups': _PET_GROUPS, 'stores': (), 'extra': (), 'files': {}, 'delete': (), 'refusal': None}}`;
   - `BOARD_INSERT=_BOARD_INSERT`;
   - `LOCATION='separate'`, or `'default'` for the default-location variant.
2. Insert the crash point into the scenario's target role, just before `restored=reviewed(service,`:
   ```
   from tldw_chatbook.Backup_Recovery import publication as crash_publication
   crash_publication.finalize_candidate=lambda *args,**kwargs:os._exit(17)
   ```
   `replace()` imports `finalize_candidate` from `publication` inside the function, so the patch takes effect.
3. Run the source role, then the target role, each in a child process with `test_home_citation_retirement._run`'s environment: `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH=<root>/config/config.toml` under the role's root, plus `TLDW_TEST_MODE=1`. The target exits with 17, after publication.
4. In a fresh child with the target's environment:
   ```
   pending,_=bootstrap._records(bootstrap.default_bootstrap_root())
   operation=pending[0]['operation_id']
   recover_replacement(operation,control_root=Path.home()/'recovery-control',action=ACTION,rollback_password=b'rollback-test-password',cancel=Event())
   ```
   (from `tldw_chatbook.Backup_Recovery.replacement`).
5. Check `bootstrap.startup_permission(selector, bootstrap.default_bootstrap_root())` afterwards. With `ACTION='finish'` this raises today and startup stays `(False, 'recovery_pending')`; with `ACTION='rollback'` it returns `rolled_back` and startup is `(True, 'startup_allowed')`.

Priority medium: the app refuses to start until rollback is chosen, though rollback recovers it and nothing is lost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test kills a Replace at finalize_candidate (after publication) on a separate config folder and on the default one, then runs finish and rollback in fresh processes; finish is red before the fix
- [x] #2 finish completes after that crash, or the finish action is not offered when it cannot complete, and the notes say which and why
- [x] #3 After each recovery the app starts (startup permission and admission), and rollback keeps working
- [x] #4 The test runs on a PR or dev-push lane, with its time recorded
- [x] #5 (coordinator, 2026-11-21; job revised in review rev1) A dev-push job beside the Backup_Recovery owner-suite step runs test_replacement_recovery.py, test_default_service_container.py, the test_first_*binding* files and test_legacy_chat_context_roundtrip.py; any file that cannot pass reliably there is left off with the reason; each added file is measured with the step's flags, the job's total is recorded against its 20-minute step cap, and the Tests/CI pins are updated
<!-- AC:END -->

## Implementation Plan

1. Reproduce with the task's recipe (separate folder, crash at `finalize_candidate` after publication, `finish` in a fresh process): `FileNotFoundError` from `_acquired_source` -> `publication._descriptor`.
2. Add a test to `Tests/Backup_Recovery/test_created_owner_rollback.py` (already in the dev-push Backup_Recovery owner-suite lane), parametrized on the config location (separate, default): one source backup, then two targets each killed at `finalize_candidate`; one recovers with `finish`, the other with `rollback`, each in a fresh process, and each then checks startup permission and startup admission. Show `finish` red.
3. Fix in `recover_replacement`: re-derive the archive (`_acquired_source`, whose `_descriptor` needs every candidate artifact still staged) for the Personal Context key install only while publication has not started. `replace()` and `finish` both install the keys before `publication_started` is journaled, and `publish_candidate` already skips `_descriptor` once publication started.
4. Green at both locations; finish before publication (existing crash tests) and rollback still pass; owner suites, Architecture, preflight, ruff; record the test's time.

## Implementation Notes

**Cause.** TASK-40135 (0f7ded1620, 2026-11-13) added the Personal Context key install to `recover_replacement`'s finish path as `_install_personal_context_keys(_acquired_source(candidate, plan, cancel), ...)`, gated only on "not committed". `_acquired_source` runs `publication._descriptor`, which needs every candidate artifact still in its stage (and re-reads the archive). After a crash anywhere past `publication_started`, the staged files have moved, so finish raised `FileNotFoundError` (or `candidate_identity_changed`) before it reached `publish_candidate`/`finalize_candidate`. The archive was re-derived even when the plan has no Personal Context keys, because the argument is evaluated first.

**Fix (AC#2: finish completes).** That call now runs only while `publication_started` is not journaled, the same gate `publish_candidate` uses for its own `_descriptor` check. This is safe: `replace()` and finish both install the keys before `publish_candidate` journals `publication_started`, so once publication has started they are already installed and read back. A crash before publication still re-installs them, accepting the same value, and `test_restore_data_groups.py`'s Personal Context `finish` scenario (interrupted at `publish_candidate`) still passes. Rollback is unchanged. `finish` stays offered.

**Test (AC#1, AC#3).** `test_created_owner_rollback.py::test_a_replace_killed_after_publication_recovers_by_finish_or_rollback[separate|default]`. Each location runs one source backup, then two targets, each killed by `os._exit(17)` at `finalize_candidate` during a real `RecoveryService` Replace with the conversations, settings and workspaces groups and no pet files. Fresh processes then check that the app refuses to start (`recovery_pending`), that the journal shows `artifact_published` and no `committed`, and that the offered actions are `('finish', 'rollback')`. Then `finish` (→ `committed`), a start (startup permission `(True, 'startup_allowed')` plus `acquire_storage()`), a later rollback of the finished restore after credential review (→ `restoration_validated`) and another start. The other target runs `rollback` (→ `rolled_back`), then a start.

**Red/green.** Without the fix, both nodes failed at the finish recovery: `state: recovery_required`. The probe recipe named the underlying `FileNotFoundError: 'tldw_chatbook_ChaChaNotes.db'` from `_acquired_source` → `_descriptor` → `observe_artifact`. With the fix, `2 passed in 326.91s`.

**Already-red tests this fixes.** `test_replacement_recovery.py::test_fresh_process_reconciles_real_killed_publication[finish-*]` had six variants red at base dc371f0842 (`6 failed, 8 passed`). They kill publication at each boundary and then run finish. That file is in no CI lane, so the regression went unseen. With the fix the whole file passes (36 passed).

**Lane and time (AC#4).** The test sits in `Tests/Backup_Recovery/test_created_owner_rollback.py`, already a whole-file target of the dev-push "Backup_Recovery owner suites" job (`derived-artifacts.yml`, `BACKUP_OWNER_SUITE_TARGETS`), so no lane edit was needed. Local time at `-n 4`: 321.4 s [separate] and 321.8 s [default], with load 33-45 on 18 CPUs (one source plus two crashed targets each, about 15 child processes per node).

**Files.** `tldw_chatbook/Backup_Recovery/replacement.py` (`recover_replacement` only: one condition and a comment, +3 lines); `Tests/Backup_Recovery/test_created_owner_rollback.py`.

**Covering runs** (`-n 4 --timeout=1500`, through the wrapper, load 40-57). `test_replacement_recovery.py`: 36 passed. `test_restore_data_groups.py`, `test_publication_crashes.py`, `test_rollback_credentials.py`, `test_selected_owner_finish.py`, `test_settings_finish.py`, `test_replacement_publication_scopes.py`, `test_recovery_service.py` and `test_replacement_abort.py` together: 173 passed, 2 failed. The 2 failures are `test_replacement_publication_scopes.py::test_complete_cli_replacement_admits_mapped_sources[True|False]`: the child hit its fixed 90 s subprocess timeout. They fail the same way at base dc371f0842 with this fix reverted (`2 failed`, both `timed out after 90 s`, load 55-58), and that path never reaches the changed line (its only recovery is `abort`).

**Lane (AC#5, coordinator 2026-11-21, revised in review rev1).** Seven files that ran in no CI lane now run in a new dev-push job of `.github/workflows/derived-artifacts.yml`, `backup-recovery-binding-dev` ("Backup_Recovery recovery and first-binding suites (push to dev)"). The first commit (c3179e708d) put them in the owner-suite step. The review moved them to their own job because that step took 51:23 in CI at dc371f0842 and is estimated at 58-70 min on CI with this round's four created-owner nodes and backupdef's net seven (reviewer's estimate, review rev2). The new job copies the owner-suite job's trigger, `needs`, install, runner and flags (`-n 4 --timeout=300 --tb=short`). It is not in the aggregate's needs.

**Caps.** The new job's step cap is 20 min and the job's is 30. The seven files together with the step's flags took 159 nodes in 5.7 min at load 15-23. The owner-suite step's cap rises from 70 to 85 min and its job's from 75 to 90 (coordinator decision), with a dated comment citing the 51:23 CI time, this branch's local 56.8 min run and backupdef's additions. `BACKUP_BINDING_SUITE_TARGETS` and `test_backup_binding_suites_run_on_dev_pushes_only_and_are_never_required` (in `Tests/CI/test_ci_queue_pressure_contract.py`) pin the job. `test_derived_artifacts_workflow.py`'s job list names it. The seven files left `BACKUP_OWNER_SUITE_TARGETS`. ADR-103 records the new job and both cap changes in a dated 2026-11-21 round-19 paragraph (review rev2).

Each file alone, with `-n 4 --timeout=1500`, two passes, load 11-27:

| File | Nodes | Wall (pass 1 / pass 2) | Slowest node |
|---|---|---|---|
| `test_replacement_recovery.py` | 36 | 65 s / 93 s | 13 s |
| `test_default_service_container.py` | 40 (45 after review rev1) | 9 s / 15 s | |
| `test_first_binding_absent_sqlite.py` | 16 | 7 s / 10 s | |
| `test_first_binding_unused_scaffolds.py` | 12 | 5 s / 5 s | |
| `test_first_replacement_binding.py` | 25 | 266 s / 268 s | 65 s |
| `test_first_user_data_binding.py` | 24 | 12 s / 11 s | |
| `test_legacy_chat_context_roundtrip.py` | 1 | 72 s / 58 s | 66 s |

Run one at a time that is about 8 minutes (436 s and 460 s); run together, 5.7 min. Every node stays far under the step's `--timeout=300`, and none of these files has its own timeout marker.

**Left off:** `test_default_service_replacement.py`. Its children have fixed 150 s (F9 child) and 300 s (later-rollback child) budgets. Here it failed under load in three of three runs with this branch, and in one of two runs at dev dc371f0842. It is the same roundtrip as the step's already-deselected `test_f9_later_rollback_requires_explicit_credential_review`. `test_first_config_lock.py` does not match `test_first_*binding*` and was not added.

**Review rev1, minor items.** `test_created_owner_rollback.py` now asserts at import that `_SCENARIO` holds the crash anchor exactly once, so a missing anchor fails loudly. The `recover_replacement` comment now says the keys were installed before `publication_started` was journaled, and that once it is the staged files may have moved.

**Checks:** `Tests/CI` at `-n 4`: 1812 passed, 2 skipped. The crash and default-location tests pass, and so does preflight.

**First dev run (2026-11-22, coordinator).** The new job failed on its first dev push, run 44975330028 at fe84affafc: 19 setup errors in `test_first_replacement_binding.py` (`No module named build`, from the fixture that builds a wheel with `python -m build --no-isolation`) and one failure in `test_replacement_recovery.py::test_actual_native_projection_group_is_reversed_and_readable` (`No module named chromadb`). The local measurements above used the dev venv, which has both. The job now installs `setuptools>=77`, `build` and `chromadb==1.5.8` on top of the owner-suite line.
