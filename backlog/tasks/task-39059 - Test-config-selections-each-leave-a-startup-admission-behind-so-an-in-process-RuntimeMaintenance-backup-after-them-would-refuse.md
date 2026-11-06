---
id: TASK-39059
title: "Test config selections each leave a startup admission behind, so an in-process RuntimeMaintenance backup after them would refuse with runtime_startup_scope_unqualified"
status: To Do
created_date: 2026-11-06 10:32
labels:
- follow-up
- tests
- backup-recovery
priority: low
references:
- Tests/Backup_Recovery/config_test_support.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/runtime_maintenance.py
- tldw_chatbook/config.py
- Tests/conftest.py
- Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py
- Tests/Chat/test_local_server_discovery.py
- Tests/Wizards/test_first_run_protect_state.py
dependencies:
- TASK-38964
- TASK-38986
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down, by the storage review of TASK-38986; checked on dev b2c618185f, 2026-11-06, and on storage PR #3435 (head 3cc391fadc), which lands TASK-38986 and carries the endpoint PR's tests. Line numbers are dev's; ones marked #3435 are from that head. The helper and the admission code are the same on both.

**How a selection adds a startup.**
- `Tests/conftest.py`'s per-test isolation gives every test outside the bootstrap-profile list its own `HOME` (`:1783-1800`; #3435 :1829-1846). The bootstrap root is `Path.home()/.config/tldw_cli/recovery-bootstrap` (`bootstrap.py:128-129`, `profile_paths.py:21-22`), so each such test has its own root.
- `select_config_source` calls `install_config_source` (`Tests/Backup_Recovery/config_test_support.py:15-29`), which re-executes `tldw_chatbook.config`.
- Importing `config.py` calls `admit_startup()` (`config.py:4-6`). `_admit_startup` files a lease in `storage_admission._startups`, keyed by (pid, bootstrap root) (`storage_admission.py:1540-1560`).
- Nothing releases it at teardown; `_shutdown` releases every startup at exit (`atexit`, `:1595`).
- About a dozen test files pop their own startup by key instead. Examples: `bound_config` (`Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py:451`), `Tests/Backup_Recovery/test_mcp_recovery_review.py:45`, `Tests/Backup_Recovery/test_activation_model_artifacts.py:36`. The shared helpers do not.

**How many accumulate.** A probe plugin counted `_startups` at session end, in one process (`-p no:xdist`, run.sh):

| File | Tree | Passed | Startups held at the end |
|---|---|---|---|
| `Tests/Chat/test_local_server_discovery.py` | dev | 94 | 1 |
| `Tests/Chat/test_local_server_discovery.py` | #3435 | 282 | 110, one per TASK-38964 make-default node |
| `Tests/Wizards/test_first_run_protect_state.py` | dev | 73 | 15 |
| `Tests/test_config_private_bootstrap.py` | dev | 24 | 4 |
| `Tests/test_config_delete_settings.py` | dev | 46 | 1 |

**Why it matters.**
- `_LocalPause.retire_startup` (`storage_admission.py:314-322`) is called by the live-backup path (`runtime_maintenance.py:733`). It raises `RecoveryRequired("runtime_startup_scope_unqualified")` unless `_startups` holds exactly the current key. The probe read that condition as False after the discovery file and after the two dev files with more than one startup.
- So any later test in the same process that runs a real RuntimeMaintenance backup would refuse.
- No lane hits it today:
  - the tests that call `retire_startup` (`Tests/Backup_Recovery/test_runtime_startup_handoff.py`, `test_rag_projection_capture.py`) run their app in a child process (`_run`);
  - the dreams step's later backup file runs in sandboxed children.
- Production admits once per process.
- TASK-38986 made `--storage-lease-guard` subtract `_startups` (#3435 `Tests/conftest.py:1147`; its docstring calls them "each held until exit by design"). So the guard no longer reports these, which was right for the drain it protects but leaves this case unwatched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A config selection made through `select_config_source` or `install_config_source` releases, at teardown, the startup admission it added, and leaves in place any admission it did not add (the process's own, or a wider fixture's); a test is red before the change (after a selection in a fresh-HOME test, `_startups` has grown by one at teardown)
- [ ] #2 After `Tests/Chat/test_local_server_discovery.py` and `Tests/Wizards/test_first_run_protect_state.py` run in one process (`-p no:xdist`), exactly one startup admission remains (110 and 15 before)
- [ ] #3 The test files that pop their own startup today still pass, and the helper's own tests (`Tests/Backup_Recovery/test_config_test_support.py`) cover the release
- [ ] #4 `--storage-lease-guard` reports a startup admission that a test adds and leaves behind as its own finding, shown by a negative control (a selection without the release); or the notes record why that check is not added
<!-- AC:END -->
