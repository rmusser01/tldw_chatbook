---
id: TASK-38989
title: "Architecture census rows and two guards are out of date on dev, and none of the 23 red Architecture files runs on a PR lane"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 07:30
labels:
- follow-up
- tests
- architecture
- ci
priority: medium
references:
- Tests/Architecture
- tldw_chatbook/provider_registry.py
- tldw_chatbook/Agents/hook_permissions.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b on 2026-11-05: a serial `Tests/Architecture` run had 41 failures in 1044 s at load 7-10. This task covers 13 of them. Every one is a census or inventory out of date, or a guard flagging something it should not. No PR lane runs any of the 23 failing files, and the branches due to merge before this one do not touch them.

**Library export cluster (3 nodes: `test_library_cluster_membership_census.py`, and `test_library_export_wiring.py` ×2).**
- The cluster census has an unaccounted `_project_library_export_canvas`.
- Both export-wiring nodes fail on `_run_library_export_via_service`, which is neither a delegator nor class-forwarding.

**Census rows out of date, and guard false positives (7 nodes: the framework armed-clock inventory, `test_no_undefined_module_globals.py` ×2, `test_app_extracted_patch_targets.py`, `test_persona_buddy_boundary.py`, and `test_backup_owner_inventory.py` ×2).**
- The framework clock guard flags 45 `ProviderRecord(…auto_refresh=…)` rows. That is a catalog field (`tldw_chatbook/provider_registry.py:184`), not a clock.
- The undefined-globals guard flags `_bind_store_lock`. That is a class-body helper (`tldw_chatbook/Agents/hook_permissions.py:230`), called at `:284` and deleted at `:285`. The same guard keeps an exemption row for the deleted `Local_Ingestion/API_Endpoint_Sample.py`.
- `test_app_extracted_patch_targets`: `Tests/UI/test_app_instance_warning.py:165` patches `app.get_subscriptions_db_path` on its own.
- `test_persona_buddy_boundary.py:153`: a call now has 2 arguments where the test expects 1.
- The backup owner inventory (2 nodes) shows producer census drift. `chatbook_creator` writes went from 1 to 2, and `_connect_registered_sqlite` from 1 to 2. The SQLite registry also gained `personal_context.interview_drafts`, `plugins.registry`, `library.legacy_recovery`, `recovery.operations.guardian` and `recovery.operations.dreams`.

**Profile and legacy-downloader inventories (3 nodes: `test_profile_owned_path_inventory.py` ×2, and `test_task2062_3_legacy_downloader_retirement.py`).**
- The profile-path exception rules are out of sort order.
- The profile-path walker (`rglob("*.py")`, `:313`) fails to parse a tracked QA snippet, `Docs/superpowers/qa/2026-11-01-prompt-mix-ins/ci-rebase/hosted-5a8/response-rules/original-consume-visible-handoff.py`, with an IndentationError. That file came in with bffa2bffed on 2026-11-02.
- The task2062_3 inventory has new readers: `Backup_Recovery/config_adapter.py`, `inventory.py` and `detached_layout.py`.

**Not in this task.**
- The 3 timer-census nodes belong to TASK-35059 and TASK-36191.
- The RecoveryRequired node belongs to TASK-38111.
- The guarded-modal quit census is fixed by TASK-38747.

**Lane cost.** The whole directory takes 1044 s serially for 4146 tests. The dreams-boundary step has about 13 minutes free under its 20-minute cap, and the PR Fast Lane is at its cap (TASK-38716). Laning these files therefore needs per-file times and an owner decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 13 nodes pass on dev, each drift either recorded as a reviewed census row or reverted in product code (red before: 13 of 13 fail at 29592ecc9b)
- [x] #2 The ProviderRecord and class-body-helper false positives are fixed in the guards, and each guard still fails on a real hand-armed auto_refresh clock and on a real undefined private helper (negative controls)
- [x] #3 The profile-path walker skips non-runtime Python under Docs/, or that QA snippet stops being a .py file, and the walker still catches a new runtime path (negative control)
- [ ] #4 Per-file serial times for the 23 red Architecture files are recorded, and the files run on a PR lane with their added time measured, or the notes record the owner's decision to keep them off
<!-- AC:END -->

## Implementation Plan

1. Re-confirm the 13 reds at base e8eeae7db9 and list each drift in full: several censuses stop at the first cluster or row they find.
2. Revert product drift where the census caught a real defect. Otherwise record a reviewed census row.
3. Guards: the ProviderRecord catalog field and class-body helper scope, each with negative controls. The profile-path walker skips Docs/.
4. Record per-file serial times for the 23 files. Lane the ones that pass and need no git history on the dreams-boundary step (appended at the end), and time the addition 3 times with the step's flags.

## Implementation Notes

**Status: 11 of the 13 nodes pass. AC#1 is not met for the 2 backup owner inventory nodes; see the last section.**

**Library export (3 nodes): reverted in product code.**
- `LibraryScreen._run_library_export_via_service` was a stale 97-line duplicate, not a delegator. b5251e9a6e (TASK-32628) put a full body back on the screen. That body carried the lifetime span (`operation` / `run_async`) but lacked task-32232's empty-export copy and artifact-read receipt counts, which live only in the controller. Production called the screen copy, so both task-32232 features had been absent since 2026-09-16.
- The controller body now runs inside the lifetime span, and the screen method is a one-line class-forwarding delegator again. `test_library_export_roundtrip.py` (temporary `bootstrap_profile` mark): at base, `…collecting_nothing_reports_failure` (wrong message) and `…receipt_counts_come_from_the_written_manifest` (`None == 2`) fail; both pass here.
- The User Guide (`library/import-and-export.md`) already describes the restored behaviour, so it needs no change.
- Library export controller ratchet 1492 -> 1501 under the owner waiver (dated row). library_screen.py drops 79 lines.
- The cluster census also had drift that the export failure hid. Reviewed stayer rows:
  - `_project_library_export_canvas`: task-31249's deferred projection (3cee32b3d0).
  - `retain_library_collection_annotation_draft`: a screen `@on` handler that forwards to a differently named controller method (0f69e0ba96).
  - `_library_conversation_visible_link_receipt`: ef610f3a77's receipt projection.

**Guards (AC#2).**
- **Framework clock.** `NON_CLOCK_AUTO_REFRESH_CALLEES = {"ProviderRecord"}` skips the 45 catalog rows. `test_non_clock_auto_refresh_fields_stay_plain_bools` pins `ProviderRecord` as a non-widget dataclass whose `auto_refresh` is a bool. Negative controls:
  - With the exemption removed, the new test and the package node are red.
  - A real `widget.auto_refresh = 0.5` added to a package file is flagged.
  - `SomeWidget(auto_refresh=True)` stays flagged in the new synthetic test.
- **Undefined globals.** A call made directly in a class body now resolves in that class's own namespace (`_scope_level_names`, shared with module scope). Methods, lambdas and comprehensions still never see class names.
  - `test_class_body_helper_resolves_only_in_its_own_class_body` pins both sides, plus a deleted module helper.
  - Negative controls: with class scope disabled, hook_permissions and the new test are red; an over-broad "any enclosing class" check makes the new test red.
  - The stale `API_Endpoint_Sample.py` exemption is dropped; that file is gone.

**Profile paths (AC#3).**
- The walker skips `Docs/` (`NON_RUNTIME_PYTHON_ROOTS`). Negative control: a temporary `tldw_chatbook/_r10a_negctrl.py` reading `config.BASE_DATA_DIR_CLI` is caught.
- `APPROVED_EXCEPTIONS` is stable-sorted by identity, and each rule moves with its comments. It is a pure block move: the ADR-256 group, legacy_catalogs and settings_speech_tts.

**Other census rows, reviewed.**
- **Patch targets.** `test_app_instance_warning.py` now patches `get_subscriptions_db_path` on every app module through `set_app_global`.
- **Persona Buddy boundary.** TASK-32901's six controller logs interpolated the class name into free text. They are now the boundary's fixed `persona_buddy_*` categories, with the class bound as `exception_type` (bounded by a regex, as `preferences.py` does). `test_controller_failure_logging.py` reads the bound field.
- **Legacy downloader.** `Backup_Recovery/config_adapter.py`, `detached_layout.py` and `inventory.py` name `model_download_dir` only to classify it as an inert external location. None of them reads or writes the directory.
- **Diagnostic inventory.** The Buddy and export log changes moved it, and the full branch run caught that. Every row was reviewed with `--statements` before `--write` (separate commit).

**Not done: backup owner inventory (2 nodes).**
- The filing saw only the first lines of a truncated diff. At e8eeae7db9 the producer census differs from `backup-recovery-owner-inventory.md` (last edited 2026-10-11) by:
  - **644** added rows;
  - **194** gone rows;
  - **12** changed counts;
  - the SQLite registry has **13** new members and **8** gone.
- New SQLite members: `chat.launch_wake`, `chat.trace_maintenance`, `db.chunking_lab`, `db.workspace_project_index`, `library.legacy_recovery`, `personal_context.interview_drafts`, `personal_context.repository`, `plugins.registry`, `recovery.operations.dreams`, `recovery.operations.guardian`, `tests.store_kernel`, `utils.db_upgrade_notice`, `workflows.local`.
- Gone SQLite members: `recovery.operations.receipts` and seven `settings.*` stores.
- Each new row needs an owner, a classification and a backup cohort. That is a backup-coverage review. Stamping 644 rows with defaults would defeat what the census is for, so the rows are left for a dedicated task.

**Lane (AC#4).**
- **Per-file serial times**, summed `--durations=0`, base e8eeae7db9 at load 6-15 / this branch at load 13-20:

| File | Base | Branch |
| --- | ---: | ---: |
| no_undefined_module_globals | 194.1 | 307.1 |
| profile_owned_path_inventory | 54.3 | 154.2 |
| framework_armed_clock_inventory | 25.1 | 31.7 |
| persistent_terminal_privacy_boundary | 15.9 | 30.0 |
| worker_exclusive_group_inventory | 17.2 | 21.8 |
| timer_path_static_update_inventory | 23.6 | 21.6 |
| progress_widget_clock_guard | 12.8 | 20.6 |
| recovery_entrypoints | 110.5 | 18.9 |
| backup_owner_inventory | 13.2 | 15.6 |
| app_extracted_patch_targets | 12.2 | 13.3 |
| task2062_3 | 1.9 | 2.8 |
| console_wave6_inventory | 1.9 | 2.3 |
| console_wave6_closeout | 1.8 | 2.1 |
| guarded_modal_quit_census | 1.4 | 1.6 |
| the other 9 | ~0 each | ~0 each |

  The 23 files sum to 488.5 s at base and 643.5 s on this branch. Base recovery_entrypoints included 4 × 20 s timeouts.
- **Laned: 14 files on the dreams-boundary step,** appended at the end after `test_vllm_profiles.py`. The step has headroom: its job ran 6.2-7.1 of 20 min over the 6 newest successful PR runs (2026-11-04..05; boundary step 2.0-2.4 min, admission step 3.3-3.8 min).
  - Files: actor_pack_boundary, app_extracted_patch_targets, bare_chat_screen_shells, console_fleet_ownership, framework_armed_clock_inventory, library_cluster_membership_census, library_export_wiring, persona_buddy_boundary, persona_visual_runtime_boundary, profile_owned_path_inventory, progress_widget_clock_guard, recovery_entrypoints, task2062_3, worker_exclusive_group_inventory.
  - The pins `DREAMS_BOUNDARY_TARGETS` and the critical-gate argv pin were updated; Tests/CI: 1211 passed.
  - The additions alone, 3 runs with the step's flags: 335.8 s, 250.5 s and 266.7 s (148 passed each, load ~30 / 19-27 / 13-23 on 18 CPUs shared with other agents).
  - The full step via dreamstep.sh: 625 passed in 590 s at load ~23.
  - Measured at base load, these 14 files sum to about 150 s.
- **Kept off, owner decision needed:**
  - `test_no_undefined_module_globals.py`: 2,943 parametrized nodes, which mostly cost per-test conftest overhead. 194-307 s serial; with it in, the additions run took 754 s at load ~34. Proposed follow-up: collapse it to one test, then lane it.
  - 4 files that `git show` frozen revisions and would fail on this job's shallow checkout: console_realtime_controller_boundary, console_review_selection_controller_boundary, console_wave6_inventory, console_wave6_closeout_inventory.
  - Red files owned elsewhere: timer_path_static_update_inventory (TASK-35059/36191) and persistent_terminal_privacy_boundary (TASK-38111). backup_owner_inventory is red pending the review above.
  - guarded_modal_quit_census is already on this step.

**Evidence.**
- Red before: all 13 nodes failed in the serial base run (40 failed, 4106 passed, 979 s).
- Full serial Architecture run on this branch: 7 failed, 4146 passed (1414 s, load ~15):
  - the 2 backup nodes;
  - persistent_terminal_privacy (TASK-38111);
  - the 3 timer-census nodes (TASK-35059/36191);
  - the diagnostic inventory, now regenerated and verified.

AC#1 stays open for the 2 backup nodes. AC#4 is met for the 14 laned files; the 9 kept off need the owner decisions listed above. The task stays In Progress.
