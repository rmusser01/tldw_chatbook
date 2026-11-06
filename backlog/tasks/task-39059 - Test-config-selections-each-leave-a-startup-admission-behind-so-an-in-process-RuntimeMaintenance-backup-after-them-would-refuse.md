---
id: TASK-39059
title: "Test config selections each leave a startup admission behind, so an in-process RuntimeMaintenance backup after them would refuse with runtime_startup_scope_unqualified"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 10:32
updated_date: 2026-11-06 16:43
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
- [x] #1 A config selection made through `select_config_source` or `install_config_source` releases, at teardown, the startup admission it added, and leaves in place any admission it did not add (the process's own, or a wider fixture's); a test is red before the change (after a selection in a fresh-HOME test, `_startups` has grown by one at teardown)
- [x] #2 After `Tests/Chat/test_local_server_discovery.py` and `Tests/Wizards/test_first_run_protect_state.py` run in one process (`-p no:xdist`), exactly one startup admission remains (110 and 15 before)
- [x] #3 The test files that pop their own startup today still pass, and the helper's own tests (`Tests/Backup_Recovery/test_config_test_support.py`) cover the release
- [x] #4 `--storage-lease-guard` reports a startup admission that a test adds and leaves behind as its own finding, shown by a negative control (a selection without the release); or the notes record why that check is not added
<!-- AC:END -->

## Implementation Plan

1. Red tests in `Tests/Backup_Recovery/test_config_test_support.py`: a selection in a fresh-HOME test releases the one startup admission it added when its monkeypatch is undone; a nested selection and a bootstrap-profile selection add none and release none.
2. `install_config_source` records each startup admission its config import added and registers its release on the caller's monkeypatch, so it runs with the selection's own undo (before the lease guard's teardown check). The release closes the lease only while it is still the one in `_startups` under that key, so a test that popped its own (`bound_config`) is left alone.
3. `--storage-lease-guard` reports a startup admission a test adds and leaves behind as its own finding; negative control: the same selection without the release.
4. Count `_startups` after the discovery and first-run protect-state files in one process (`-p no:xdist`), and run the files that pop their own startup.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Selection release (AC#1).** `install_config_source` (`Tests/Backup_Recovery/config_test_support.py`), which `select_config_source` calls, snapshots `storage_admission._startups` around the config import. It enters each lease the import added in `_AddedStartups` through the caller's `monkeypatch.setitem`. Undoing a new key calls `del`, and the dict's `__delitem__` then pops and closes that lease, but only while `_startups` still holds it under its key. An admission the selection did not add is never touched: the process's own, a wider fixture's, or one that a fixture like `bound_config` already popped. The release runs with the selection's own undo (a `MonkeyPatch.context()` exit, or the test's `monkeypatch` teardown). It runs even when the import raised.

**Second source of leftover admissions (found while checking AC#2).** After the selection release, `test_first_run_protect_state.py` alone still left 3 admissions, not 1. A probe that wrapped `storage_admission._admit_startup` showed where the other two came from. Two production modules call `admit_startup()` at import, `tldw_chatbook/TTS/__init__.py` and `tldw_chatbook/MCP/server.py`. So the first fresh-HOME test to import each of them admits that test's own profile, which nothing uses again. Those two tests were `test_card_path_names_the_settings_encryption_card` and `test_summary_key_encryption_row_names_the_card[encrypted]`. `isolate_test_environment` (`Tests/conftest.py`) now releases, at teardown, every startup admission whose root is under the test's own fresh profile dir (`_release_profile_startups`). Bootstrap-profile tests are skipped.

**Counts in one process (AC#2).** `-p no:xdist`, through run.sh, with a probe plugin counting `_startups` at session end:

| Files | Base `2684e5453f` | This branch |
|---|---|---|
| `Tests/Chat/test_local_server_discovery.py` (282 passed) | 110 | 1 |
| `Tests/Wizards/test_first_run_protect_state.py` (73 passed) | 15 | 1 |
| both (355 passed) | 124 | 1 |

**Guard (AC#4).** `_StorageLeaseGuard` keeps leaving startup admissions out of its live-lease count, as the drain does. It now also reports a startup admission that the test added and still holds, as its own finding: "left N new startup admission(s) held, admission root(s): [...]". The release runs before the guard's check. Negative control: with the selection release removed, `Tests/Backup_Recovery/test_config_test_support.py --storage-lease-guard` reported "left 1 new startup admission(s) held" for `test_selected_config_uses_real_custody_and_refuses_later_retarget` and for the new release test. It is clean with the release. The Dreams boundary step (`--storage-lease-guard`) passed with the new finding: 906 passed, see TASK-39049.

**Tests (AC#3)** in `Tests/Backup_Recovery/test_config_test_support.py`.
- `test_selection_releases_only_the_startup_admission_it_added` runs in a fresh HOME. It makes an outer selection and a nested selection on the same root. It asserts that exactly one admission was added, that the inner undo kept the outer's, and that all are released at the end. Red before the change: `_startups` ended with one more key, the test HOME's `recovery-bootstrap` root.
- `test_selection_in_the_bootstrap_profile_keeps_the_process_admission` is bootstrap-profile: the selection adds nothing and the process's admission stays. It is green before the change. It was red under a temporary edit that released every admission present, not only the added ones: `{} != {(pid, '.../recovery-bootstrap'): <StorageLease>}`.

**Files that pop their own startup, and every helper user.** I ran 99 files that use `_startups`, `select_config_source` or `install_config_source`, plus the two AC#2 files, at `-n 4` (load 37-58). This branch: 119 failed, 17 errors, 3489 passed. Base: 121 failed, 17 errors, 3474 passed. Three nodes failed only here. Rerun on both trees, two of them (`test_actual_app_refuses_new_content_during_pause`, `test_ctrl_n_recovers_hidden_widget_focus_and_stops_at_summary`) behaved the same on both. The third, `test_raw_source_coordinator_io.py::test_remaining_source_entry_proofs_do_not_hold_coordinator`, failed 4 of 4 here at first, then 0 of 6, while base failed 4 of 6 ("the actual selected-source reader was never reached"). It is timing-sensitive on both trees. Of the 32 files that pop or read a startup by key, 22 pass whole here. These include `test_guarded_raw_evidence_reuse.py` (`bound_config`), `test_default_profile_selection.py`, `test_mcp_source_lifetimes.py`, `test_activation_model_artifacts.py`, and `test_activation_skills.py`, which fails one node at base. Nine fail the same number of nodes on both trees: `test_participant_lifetimes.py` 8, `test_raw_related_source_preparation.py` 7, `test_visual_identity_lifetimes.py` 5, `test_persona_visual_lifetimes.py` 5, `test_mcp_recovery_review.py` 4, `test_config_participant_lifetimes.py` 3, `test_rag_profile_admission.py` 3, `test_projection_generation.py` 2 and `test_projection_enhanced_recovery.py` 1. `test_activation_mcp.py` fails 1 node here and 2 at base.

**Files.** `Tests/Backup_Recovery/config_test_support.py`, `Tests/Backup_Recovery/test_config_test_support.py`, `Tests/conftest.py`.
<!-- SECTION:NOTES:END -->
